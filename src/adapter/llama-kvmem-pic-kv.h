#pragma once

#include "llama.h"
#include "pic-rope.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class llama_memory_kvmem;
struct llama_kvmem_stash;

namespace kvmem_pic {

// Persist with the independently evaluated segment, not inferred from the
// destination or environment at reuse time. Epoch covers model/template/codec
// and segment-construction rules. No exact-ancestor entry may masquerade as this.
struct kv_segment_identity {
    const llama_model * model = nullptr;
    const llama_model * mtp_model = nullptr;
    uint64_t epoch = 0;
    bool independently_evaluated = false;
};

struct kv_layer_codec {
    bool mtp = false;
    uint32_t raw_layer = 0;   // Target: model layer; MTP: raw layer 0.
    uint32_t graph_layer = 0;
    packed_k_rope_config k;
    ggml_type v_type = GGML_TYPE_F16;
    uint32_t v_elements = 0;
    int v_hadamard_nrot = 0; // Explicit codec identity even though V is memcpy.
};

struct kv_splice_request {
    kv_segment_identity source_identity;
    kv_segment_identity destination_identity;
    std::vector<kv_layer_codec> source_codecs;
    std::vector<kv_layer_codec> destination_codecs;
    uint32_t source_begin = 0;
    uint32_t destination_begin = 0; // Must equal the current live row count.
    std::vector<llama_token> tokens; // Body only; checked token-for-token against stash.
    std::vector<rope_position> positions; // Destination body, one record per token.
    size_t max_plan_bytes = 0; // Required cap on retained plan bytes, not peak scratch.
};

struct kv_splice_layer {
    kv_layer_codec codec;
    // Complete valid destination block, including any untouched prefix rows.
    // No padding rows are published as valid. K has destination RoPE; V is exact.
    std::vector<uint8_t> packed_k;
    std::vector<uint8_t> packed_v;
    // Ordered pre-RoPE K sum over all valid rows, not a normalized mean.
    // New rows are reconstructed from quantized K and are APPROXIMATE.
    std::vector<float> k_sum;
    // Original target-only prefix sum, retained to reject a stale mean snapshot.
    std::vector<float> prefix_k_sum;
};

struct kv_splice_block {
    uint32_t block_id = 0;
    uint32_t prefix_rows = 0;
    uint32_t valid_rows = 0;
    int32_t previous_slot = -1; // Commit revalidates; never an allocated reservation.
    std::vector<kv_splice_layer> layers;
};

struct kv_splice_plan {
    const llama_memory_kvmem * destination = nullptr;
    kv_segment_identity identity;
    uint64_t attention_epoch = 0;
    uint32_t source_begin = 0;
    uint32_t destination_begin = 0;
    uint32_t destination_end = 0;
    uint32_t block_tokens = 0;
    uint32_t slots_needed = 0;
    bool approximate = true; // Propagate to resulting raw/Q/checkpoint provenance.
    std::vector<llama_token> tokens;
    std::vector<rope_position> positions;
    std::vector<kv_splice_block> blocks;
    LLAMA_API size_t bytes() const;
};

// HOST PREPARATION ONLY. Caller synchronizes both contexts, commits all target/
// MTP harvest (including a partial destination tail), and excludes concurrent
// writes throughout preparation/validation/commit. Source stash must be detached
// and immutable. No decode, tensor upload, slot allocation or live raw writes.
// Identity and full codec metadata are trusted capture-time bindings supplied
// by the segment builder/adapter; stash alone cannot prove their authenticity.
// Commit must match identity.model/mtp_model/epoch against the live context.
//
// Arbitrary source/destination block offsets are supported. Text-only seq0,
// non-transposed V, complete FA+MTP codec coverage, RAM stores, and spare slots
// without eviction are required. Unsupported pic-rope configurations fail closed.
// Existing target tail sums must be checkpointed; MTP tail sums are reconstructed.
// Mean reconstruction relocates source K to position zero, then dequantizes and
// removes Hadamard. This adds quantization loss; original FP32 sums are unrecoverable.
// No logits, recurrent/conv composition, MTP carry or pending-token handling here.
// False or allocation exception leaves out and all stores unchanged.
LLAMA_API bool pic_kv_prepare(const llama_kvmem_stash & source, llama_memory_kvmem & destination,
                             const kv_splice_request & request, kv_splice_plan & out, std::string & error);

// Recheck a prepared, unmodified plan immediately before commit, under the same
// exclusion boundary. This verifies epoch/row count/slots, sizes and prefix KV;
// it is not a substitute for the adapter's private row_positions/cell checks.
LLAMA_API bool pic_kv_validate(llama_memory_kvmem & destination,
                              const kv_splice_plan & plan, std::string & error);

// Adapter integration contract (Avicenna; not declared as an unimplemented stub):
// bool llama_memory_kvmem::pic_kv_commit(const kvmem_pic::kv_splice_plan &, std::string &);
// It must validate again, preallocate host metadata and reserve every required
// slot, preserve first-block prefix metadata, append tokens/four-axis positions,
// write ALL target/MTP packed blocks + write_layer_mean_sum(block_start,valid_rows),
// register runtime rows, occupy seq0 cells in matching target/MTP slots, invalidate
// resident reuse tags and advance attention_epoch. Publish success only after all
// transfers finish. Never invoke llama_decode to manufacture the skipped body.
// An outer transaction must restore a clean KV + recurrent/conv + MTP carry + Q
// checkpoint on any exception/failure after mutation; truncation alone cannot undo
// overwritten tail cells/means. Resume logits via explicitly evaluated seam/tail.

} // namespace kvmem_pic
