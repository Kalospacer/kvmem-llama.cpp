#pragma once

// Host spill for attention KV, indexed by logical block then layer.
//
// Product path: ordered K sum (F32) at first write, mean-K computed on read;
// packed GPU-format K/V (q8_0 etc)
// copied at stage-out. Restore is memcpy; orig pos does not need unrotated K.
// `write_layer_k_rows` (unrotated token-major K) remains for tests only.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace kvmem {

class NvmeKvTier;
class SnapshotWriter;
class SnapshotReader;
struct SnapshotBuffer;

struct RawKvStoreConfig {
    uint32_t n_layer = 0;
    uint32_t n_embd_k = 0;
    uint32_t n_embd_v = 0;
    uint32_t block_tokens = 128;
    uint64_t k_row_bytes = 0;      // 0 = FP16 (n_embd_k * 2); else opaque K row
    uint64_t k_gpu_row_bytes = 0;  // 0 = no packed GPU K; else bytes per token row
    uint64_t v_gpu_row_bytes = 0;  // 0 = no packed GPU V; else bytes per token row
    uint64_t nvme_bytes = 0;
    std::string nvme_dir;
    std::string nvme_file = "kvmem_raw_k.bin";
};

class RawKvStore {
public:
    explicit RawKvStore(RawKvStoreConfig cfg);
    ~RawKvStore();

    RawKvStore(const RawKvStore &) = delete;
    RawKvStore &operator=(const RawKvStore &) = delete;

    const RawKvStoreConfig &config() const { return cfg_; }
    bool nvme_enabled() const;

    void ensure_blocks(uint32_t block_count);

    // Raw K and packed GPU K are separate representations and may coexist;
    // raw-K writes do not invalidate packed K. Replay invalidates explicitly.
    // A non-null V replaces the packed-V representation in each touched block,
    // clearing its valid count. Switching back requires refilling packed rows.
    void write_layer_tokens(uint32_t pos0, uint32_t n, uint32_t il,
                            const float * k, const float * v);
    void write_layer_tokens_f16(uint32_t pos0, uint32_t n, uint32_t il,
                                const uint16_t * k, const uint16_t * v);
    // Opaque unrotated K rows (`k_row_bytes`). Optional F32 is mean only.
    void write_layer_k_rows(uint32_t pos0, uint32_t n, uint32_t il,
                            const uint8_t * k, const float * k_f32);
    void write_layer_v_gpu(uint32_t pos0, uint32_t n, uint32_t il, const uint8_t * v);
    // Packed GPU-format K (RoPE+Hadamard+quant already applied). RAM only.
    void write_layer_k_gpu(uint32_t pos0, uint32_t n, uint32_t il, const uint8_t * k);
    // Prefill mean-K only (no raw-K rows). k is token-major F32, n_embd_k per token.
    void write_layer_mean_k(uint32_t pos0, uint32_t n, uint32_t il, const float * k);
    // Decode running-sum: `sum` is already reduced over `n` tokens in one block.
    void write_layer_mean_sum(uint32_t pos0, uint32_t n, uint32_t il, const float * sum);

    bool has_block(uint32_t block_id) const;
    bool has_k(uint32_t block_id, uint32_t il) const;
    bool has_k_gpu(uint32_t block_id, uint32_t il, uint32_t n = 1) const;
    bool has_v(uint32_t block_id, uint32_t il) const;
    bool has_v_gpu(uint32_t block_id, uint32_t il, uint32_t n = 1) const;
    uint32_t n_tokens(uint32_t block_id) const;

    bool copy_k(uint32_t block_id, uint32_t il, float * out) const;
    bool copy_v(uint32_t block_id, uint32_t il, float * out) const;
    bool copy_k_rows(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;
    bool copy_k_gpu(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;
    bool copy_v_gpu(uint32_t block_id, uint32_t il, uint8_t * out, uint32_t n) const;

    // Normalize the stored sum by mean_tokens; absent statistics return zeros.
    void mean_k(uint32_t block_id, uint32_t il, float * out) const;

    // Legacy per-store logical sizes (shared payload counted in each store),
    // plus NVMe write counters; not resident capacity or unique memory.
    size_t bytes_k() const;
    size_t bytes_v() const;
    size_t allocated_bytes() const;
    uint64_t capacity_bytes(uint32_t tokens, uint32_t populated_layers = UINT32_MAX) const;
    void snapshot_write(SnapshotWriter & out);
    void snapshot_read(SnapshotReader & in, uint32_t max_blocks);
    // Caller freezes the detached store until all bindings have been restored.
    void snapshot_buffers(std::vector<SnapshotBuffer> & buffers);

    // Append allocation identity/bytes; deduplicate identities across stores.
    // Counts sizeof(*this), vector capacities (including private block/layer
    // metadata, raw K/V, k_sum and scratch/queued I/O), packed vector objects
    // and their byte capacities, and heap storage of config strings.
    // Excludes allocator/shared_ptr control-block overhead, NVMe tier internals,
    // disk bytes, worker-local in-flight I/O and thread/OS resources.
    // Entries do not own memory. Quiesce harvest/writers and retain all stores
    // throughout a multi-store snapshot; this call only locks this store.
    void append_allocations(std::vector<std::pair<const void*, size_t>>& out) const;

    // Immutable packed leases for a completed GPU capture. These keep payload
    // identity alive and force later writers through COW. Caller fences harvest.
    bool packed_refs(uint32_t block_id, uint32_t il, uint32_t rows,
                     std::shared_ptr<const std::vector<uint8_t>> & k,
                     std::shared_ptr<const std::vector<uint8_t>> & v) const;
    // A graph will overwrite this block starting at keep_rows. Other blocks'
    // backing copies remain valid, including nonresident historical blocks.
    void invalidate_packed_block(uint32_t block_id, uint32_t keep_rows);

    uint64_t nvme_bytes_written() const;
    uint64_t nvme_syscalls() const;
    uint64_t nvme_wait_ns() const;

    void wait_writes();
    void clear();
    // Preserve only the valid prefix, including a partial last block.
    void truncate_to(uint32_t token_pos);
    // Share each layer's packed K/V; metadata, raw K/V and k_sum stay private.
    // RAM only (both clone methods throw for NVMe). Caller must complete
    // asynchronous harvest before publishing a branch; shared_ptr is not a fence.
    std::unique_ptr<RawKvStore> clone_prefix(uint32_t token_pos) const;
    std::unique_ptr<RawKvStore> clone_prefix_deep(uint32_t token_pos) const;
    void invalidate_packed_from(uint32_t token_pos);
    // In-process tail checkpoint: per layer, valid count followed by the F32 sum.
    std::vector<float> mean_checkpoint(uint32_t token_pos) const;
    void restore_mean_checkpoint(uint32_t token_pos, const std::vector<float> & state);

private:
    using PackedPayload = std::shared_ptr<std::vector<uint8_t>>;
    static const std::vector<uint8_t> & payload(const PackedPayload & p);
    static std::vector<uint8_t> & writable_payload(PackedPayload & p);
    void ensure_blocks_locked(uint32_t block_count);
    struct LayerBlk {
        uint32_t n_tokens = 0;
        uint32_t k_gpu_tokens = 0;
        uint32_t v_gpu_tokens = 0;
        uint32_t mean_tokens = 0;
        std::vector<uint8_t> k;
        std::vector<uint16_t> v;
        PackedPayload k_gpu;
        PackedPayload v_gpu;
        std::vector<float> k_sum;
        bool k_on_nvme = false;
        bool v_on_nvme = false;
        bool k_gpu_fmt = false;
        bool v_gpu_fmt = false;
        bool k_flushing = false;
        bool v_flushing = false;
    };
    struct BlockRaw {
        std::vector<LayerBlk> layers;
    };

    uint32_t nvme_key(uint32_t block_id, uint32_t il, bool is_v) const;
    uint64_t k_row_bytes() const;
    uint64_t k_slot_bytes() const;
    uint64_t v_slot_bytes() const;
    uint64_t v_gpu_slot_bytes() const;
    bool k_is_f16() const;
    void maybe_flush_k(uint32_t block_id, uint32_t il);
    void maybe_flush_v(uint32_t block_id, uint32_t il);
    void capture_mean_f16(LayerBlk & lb) const;
    void add_mean_f32(LayerBlk & lb, uint32_t off, uint32_t take, const float * k);
    bool load_k_nvme(uint32_t block_id, uint32_t il, uint8_t * dst) const;
    bool load_v_nvme(uint32_t block_id, uint32_t il, uint16_t * dst) const;
    bool load_v_gpu_nvme(uint32_t block_id, uint32_t il, uint8_t * dst) const;
    void enqueue_flush(uint32_t key, std::vector<uint8_t> && data, uint64_t bytes,
                       uint32_t block_id, uint32_t il, bool is_v);
    void io_loop();
    bool io_sync_inline() const;

    struct IoJob {
        uint32_t key = 0;
        uint32_t block_id = 0;
        uint32_t il = 0;
        bool is_v = false;
        uint64_t bytes = 0;
        std::vector<uint8_t> data;
    };

    RawKvStoreConfig cfg_;
    std::vector<BlockRaw> blocks_;
    std::unique_ptr<NvmeKvTier> nvme_;
    mutable std::vector<uint16_t> io_;
    mutable std::vector<uint8_t> io8_;
    size_t nvme_k_bytes_ = 0;
    size_t nvme_v_bytes_ = 0;
    uint64_t nvme_syscalls_ = 0;
    uint64_t nvme_wait_ns_ = 0;
    mutable std::mutex mu_;
    mutable std::condition_variable cv_;
    std::vector<IoJob> q_;
    size_t inflight_ = 0;
    std::thread io_thread_;
    std::atomic<bool> stop_io_{false};
};

} // namespace kvmem
