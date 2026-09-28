#include "kvmem/raw_kv_store.hpp"
#include "kvmem/snapshot.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

static void check(bool ok) { if (!ok) throw std::runtime_error("snapshot test failed"); }
template<class F> static void rejects(F f) {
    try { f(); } catch (const std::exception &) { return; }
    throw std::runtime_error("expected snapshot failure");
}

int main() {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 3; cfg.n_embd_k = 64; cfg.n_embd_v = 64; cfg.block_tokens = 64;
    cfg.k_gpu_row_bytes = 68; cfg.v_gpu_row_bytes = 128;
    kvmem::RawKvStore source(cfg), restored(cfg);
    constexpr uint32_t tokens = 9021;
    std::vector<uint8_t> k(tokens*68), v(tokens*128);
    std::vector<float> means(tokens*64);
    for (size_t i = 0; i < k.size(); ++i) k[i] = uint8_t(i*19+7);
    for (size_t i = 0; i < v.size(); ++i) v[i] = uint8_t(i*13+3);
    for (size_t i = 0; i < means.size(); ++i) means[i] = float(i % 37)-18.5f;
    for (uint32_t layer : {0u, 2u}) {
        source.write_layer_k_gpu(0, tokens, layer, k.data());
        source.write_layer_v_gpu(0, tokens, layer, v.data());
        source.write_layer_mean_k(0, tokens, layer, means.data());
    }
    const auto write = [&](kvmem::SnapshotWriter & out) { source.snapshot_write(out); };
    kvmem::SnapshotWriter size; write(size);
    std::vector<uint8_t> archive;
    size_t max_chunk = 0;
    kvmem::SnapshotWriter out([&](const void * p, size_t n) {
        max_chunk = std::max(max_chunk, n);
        const auto * begin = (const uint8_t *)p; archive.insert(archive.end(), begin, begin+n);
    });
    write(out);
    check(size.bytes() == archive.size() && out.bytes() == size.bytes());
    check(size.bytes() > kvmem::snapshot_chunk && max_chunk <= kvmem::snapshot_chunk);
    auto read = [&](const std::vector<uint8_t> & data, kvmem::RawKvStore & raw) {
        size_t offset = 0;
        kvmem::SnapshotReader in([&](void * p, size_t n) { std::memcpy(p, data.data()+offset, n); offset += n; }, data.size());
        raw.snapshot_read(in, (tokens+63)/64);
        check(in.remaining() == 0); return in.hash();
    };
    check(read(archive, restored) == out.hash());
    for (uint32_t block = 0; block < (tokens+63)/64; ++block) {
        const uint32_t n = std::min(64u, tokens-block*64);
        check(!restored.has_k_gpu(block, 1));
        for (uint32_t layer : {0u, 2u}) {
            std::vector<uint8_t> got_k(n*68), got_v(n*128);
            check(restored.copy_k_gpu(block, layer, got_k.data(), n));
            check(restored.copy_v_gpu(block, layer, got_v.data(), n));
            check(!std::memcmp(got_k.data(), k.data()+block*64*68, got_k.size()));
            check(!std::memcmp(got_v.data(), v.data()+block*64*128, got_v.size()));
            float a[64], b[64]; source.mean_k(block, layer, a); restored.mean_k(block, layer, b);
            check(!std::memcmp(a, b, sizeof(a)));
        }
    }
    check(source.allocated_bytes() <= source.capacity_bytes(tokens, 2));
    const auto before = restored.allocated_bytes();
    auto truncated = archive; truncated.resize(truncated.size()-7);
    rejects([&] { read(truncated, restored); });
    check(restored.allocated_bytes() == before); // transactional raw load
    auto corrupt = archive; corrupt[0] ^= 1;
    rejects([&] { read(corrupt, restored); });
    auto bad_length = archive;
    // Header is 4 uint32 + block uint32 + 3 uint64, then the block count.
    const uint64_t insane = UINT64_MAX;
    std::memcpy(bad_length.data()+4*5+3*8, &insane, sizeof(insane));
    rejects([&] { read(bad_length, restored); });

    std::cout << "snapshot roundtrip, bounds and checksum passed\n";
}
