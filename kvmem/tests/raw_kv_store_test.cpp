#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"
#include "kvmem/nvme_kv_tier.hpp"
#include <filesystem>

#include <algorithm>
#include <utility>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__,         \
                         __LINE__, #cond);                                     \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Compare lazy normalization with the previous ordered FP32 sum + cached mean.
// Cancellation and non-power-of-two tails catch mean-to-sum reconstruction.
static void test_sum_only(uint32_t block_tokens) {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 3; // The other two layers model uncaptured recurrent layers.
    cfg.n_embd_k = 1024;
    cfg.n_embd_v = 1024;
    cfg.block_tokens = block_tokens;
    cfg.k_row_bytes = 1088; // Quantized product path has no raw-K fallback.
    kvmem::RawKvStore raw(cfg);
    const uint32_t count = 2 * block_tokens + 7;
    std::vector<float> input(count * cfg.n_embd_k);
    const float values[] = {1e7f, .125f, -1e7f, .3f, -7.1f, .001f, 9.7f};
    for (uint32_t t = 0; t < count; ++t) {
        for (uint32_t d = 0; d < cfg.n_embd_k; ++d) {
            input[t * cfg.n_embd_k + d] = values[(t + d) % 7];
        }
    }
    const auto verify = [&](uint32_t end) {
        const uint32_t blocks = (end + block_tokens - 1) / block_tokens;
        // Exactly one F32 vector per populated block/layer, with no lazy cache.
        CHECK(raw.bytes_k() == size_t(blocks) * cfg.n_embd_k * sizeof(float));
        for (uint32_t b = 0; b < blocks; ++b) {
            CHECK(raw.has_block(b));
            const uint32_t start = b * block_tokens;
            const uint32_t n = std::min(block_tokens, end - start);
            std::vector<float> expected(cfg.n_embd_k, 0.0f), got(cfg.n_embd_k);
            for (uint32_t t = start; t < start + n; ++t) {
                for (uint32_t d = 0; d < cfg.n_embd_k; ++d) {
                    expected[d] += input[t * cfg.n_embd_k + d];
                }
            }
            const float inv = 1.0f / static_cast<float>(n);
            for (auto & x : expected) x *= inv;
            for (int repeat = 0; repeat < 3; ++repeat) {
                raw.mean_k(b, 1, got.data());
                CHECK(std::memcmp(got.data(), expected.data(), got.size() * sizeof(float)) == 0);
            }
        }
        CHECK(raw.bytes_k() == size_t(blocks) * cfg.n_embd_k * sizeof(float));
    };
    raw.write_layer_mean_k(0, 5, 1, input.data());
    verify(5);
    const auto checkpoint = raw.mean_checkpoint(5);
    CHECK(checkpoint.size() == cfg.n_layer * (1 + cfg.n_embd_k));
    CHECK(checkpoint[0] == 0 && checkpoint[1 + cfg.n_embd_k] == 5);
    CHECK(raw.mean_checkpoint(block_tokens).empty());
    // Continue in uneven batches across block boundaries.
    for (uint32_t pos = 5; pos < count;) {
        const uint32_t n = std::min(11u, count - pos);
        raw.write_layer_mean_k(pos, n, 1, input.data() + pos * cfg.n_embd_k);
        pos += n;
        verify(pos);
    }
    raw.truncate_to(5);
    std::vector<float> missing(cfg.n_embd_k, 123.0f);
    raw.mean_k(0, 1, missing.data());
    for (float x : missing) CHECK(x == 0.0f);
    raw.restore_mean_checkpoint(5, checkpoint);
    verify(5);
    // Reads/rollback must not round-trip a mean back into an approximate sum.
    raw.write_layer_mean_k(5, count - 5, 1, input.data() + 5 * cfg.n_embd_k);
    verify(count);
    for (const auto & where : {std::pair<uint32_t, uint32_t>{0, 0}, {99, 1}, {0, 99}}) {
        std::fill(missing.begin(), missing.end(), 123.0f);
        raw.mean_k(where.first, where.second, missing.data());
        for (float x : missing) CHECK(x == 0.0f);
    }
    raw.truncate_to(0);
    CHECK(raw.bytes_k() == 0 && !raw.has_block(0));
}

using Allocations = std::map<const void *, size_t>;

static Allocations allocations(const kvmem::RawKvStore & raw) {
    std::vector<std::pair<const void *, size_t>> entries;
    raw.append_allocations(entries);
    const auto count = entries.size();
    raw.append_allocations(entries); // Appends; identities are stable across reads.
    CHECK(entries.size() == 2 * count);
    Allocations result;
    for (const auto & entry : entries) {
        CHECK(entry.first && entry.second);
        const auto inserted = result.emplace(entry);
        CHECK(inserted.second || inserted.first->second == entry.second);
    }
    CHECK(result.size() == count);
    return result;
}

static size_t allocation_bytes(const Allocations & entries) {
    size_t bytes = 0;
    for (const auto & entry : entries) bytes += entry.second;
    return bytes;
}

static Allocations shared_allocations(const kvmem::RawKvStore & a, const kvmem::RawKvStore & b) {
    const auto aa = allocations(a);
    const auto bb = allocations(b);
    Allocations shared;
    for (const auto & entry : aa) {
        const auto it = bb.find(entry.first);
        if (it != bb.end()) {
            CHECK(it->second == entry.second);
            shared.insert(entry);
        }
    }
    return shared;
}

static void check_packed(const kvmem::RawKvStore & raw, uint32_t bid, uint32_t il,
                         bool is_k, const std::vector<uint8_t> & expected) {
    const auto row = is_k ? raw.config().k_gpu_row_bytes : raw.config().v_gpu_row_bytes;
    CHECK(row && expected.size() % row == 0);
    const auto n = static_cast<uint32_t>(expected.size() / row);
    std::vector<uint8_t> got(expected.size());
    CHECK(is_k ? raw.copy_k_gpu(bid, il, got.data(), n) : raw.copy_v_gpu(bid, il, got.data(), n));
    CHECK(got == expected);
}

static void test_packed_cow() {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd_k = cfg.n_embd_v = 3;
    cfg.block_tokens = 4;
    cfg.k_gpu_row_bytes = 7;
    cfg.v_gpu_row_bytes = 11;
    kvmem::RawKvStore parent(cfg);
    std::vector<uint8_t> k(9 * 7, 17), v(9 * 11, 29);
    std::vector<float> means(9 * 3, 2.0f);
    for (uint32_t il = 0; il < 2; ++il) {
        parent.write_layer_k_gpu(0, 9, il, k.data());
        parent.write_layer_v_gpu(0, 9, il, v.data());
        parent.write_layer_mean_k(0, 6, il, means.data());
    }
    const auto checkpoint = parent.mean_checkpoint(6);
    for (uint32_t il = 0; il < 2; ++il) parent.write_layer_mean_k(6, 3, il, means.data());
    const auto parent_ids = allocations(parent);
    auto child = parent.clone_prefix(6);
    auto sibling = parent.clone_prefix(9);
    auto deep = parent.clone_prefix_deep(6);
    CHECK(allocations(parent) == parent_ids);
    CHECK(shared_allocations(parent, *deep).empty());
    CHECK(shared_allocations(*child, *deep).empty());
    // Both the vector object and its actual byte allocation are shared, for
    // every retained block/layer/K-or-V, including the clipped tail.
    const size_t one_k = sizeof(std::vector<uint8_t>) + 4 * 7;
    const size_t one_v = sizeof(std::vector<uint8_t>) + 4 * 11;
    const auto shared = shared_allocations(parent, *child);
    CHECK(shared.size() == 2 * 2 * 2 * 2);
    CHECK(allocation_bytes(shared) == 2 * 2 * (one_k + one_v));
    CHECK(shared_allocations(parent, *sibling).size() == 3 * 2 * 2 * 2);
    auto unique = parent_ids;
    const auto child_ids = allocations(*child);
    unique.insert(child_ids.begin(), child_ids.end());
    CHECK(allocation_bytes(unique) == allocation_bytes(parent_ids) + allocation_bytes(child_ids)
                                      - allocation_bytes(shared));
    CHECK(child->n_tokens(1) == 2 && parent.n_tokens(1) == 4);
    CHECK(!child->has_k_gpu(1, 0, 3) && !child->has_v_gpu(1, 0, 3));
    check_packed(*child, 1, 0, true, std::vector<uint8_t>(2 * 7, 17));
    check_packed(*deep, 1, 0, false, std::vector<uint8_t>(2 * 11, 29));

    // Checkpoint/mean mutations never detach packed storage or affect siblings.
    child->restore_mean_checkpoint(6, checkpoint);
    std::vector<float> sum(3, 8.0f), got(3);
    child->write_layer_mean_sum(6, 1, 0, sum.data());
    child->mean_k(1, 0, got.data());
    CHECK(got[0] == 4.0f);
    parent.mean_k(1, 0, got.data());
    CHECK(got[0] == 2.0f);
    sibling->mean_k(1, 0, got.data());
    CHECK(got[0] == 2.0f);
    CHECK(shared_allocations(parent, *child) == shared);
    CHECK(!child->has_k_gpu(1, 0, 3));
    child->truncate_to(6);
    child->restore_mean_checkpoint(6, checkpoint);

    // An overwrite detaches just this layer's K; V and other layers stay shared.
    std::vector<uint8_t> other_k(4 * 7, 91), other_v(4 * 11, 103);
    child->write_layer_k_gpu(0, 4, 0, other_k.data());
    CHECK(allocation_bytes(shared_allocations(parent, *child)) == allocation_bytes(shared) - one_k);
    check_packed(parent, 0, 0, true, std::vector<uint8_t>(4 * 7, 17));
    check_packed(*child, 0, 0, true, other_k);
    const auto detached = allocations(*child);
    child->write_layer_k_gpu(1, 1, 0, other_k.data());
    CHECK(allocations(*child) == detached); // Unique payload reused on hot writes.
    sibling->write_layer_v_gpu(0, 4, 1, other_v.data());
    CHECK(allocation_bytes(shared_allocations(parent, *sibling)) == 3 * 2 * (one_k + one_v) - one_v);
    check_packed(parent, 0, 1, false, std::vector<uint8_t>(4 * 11, 29));
    check_packed(*child, 0, 1, false, std::vector<uint8_t>(4 * 11, 29));

    // Replay invalidation is branch-local. Refill crosses the partial tail and
    // allocates a new block without making old suffix bytes valid again.
    const auto before_invalidate = shared_allocations(parent, *child);
    child->invalidate_packed_from(5);
    CHECK(shared_allocations(parent, *child) == before_invalidate);
    CHECK(!child->has_k_gpu(1, 0, 2) && !child->has_v_gpu(1, 0, 2));
    CHECK(parent.has_k_gpu(1, 0, 4) && sibling->has_v_gpu(1, 0, 4));
    child->write_layer_k_gpu(5, 4, 0, other_k.data());
    child->write_layer_v_gpu(5, 4, 0, other_v.data());
    std::vector<uint8_t> tail_k(4 * 7, 91), tail_v(4 * 11, 103);
    std::fill_n(tail_k.begin(), 7, 17);
    std::fill_n(tail_v.begin(), 11, 29);
    check_packed(*child, 1, 0, true, tail_k);
    check_packed(*child, 1, 0, false, tail_v);
    check_packed(parent, 1, 0, true, std::vector<uint8_t>(4 * 7, 17));
    check_packed(*sibling, 1, 0, false, std::vector<uint8_t>(4 * 11, 29));
    check_packed(*child, 2, 0, true, std::vector<uint8_t>(7, 91));
    CHECK(!child->has_k_gpu(2, 0, 2));

    // Mutating the original after publication must detach too.
    parent.write_layer_k_gpu(3, 4, 1, other_k.data());
    check_packed(*sibling, 0, 1, true, std::vector<uint8_t>(4 * 7, 17));
    check_packed(*child, 0, 1, true, std::vector<uint8_t>(4 * 7, 17));
    sibling->truncate_to(4);
    CHECK(!sibling->has_block(1) && parent.has_block(1) && child->has_block(1));
    parent.clear();
    sibling.reset();
    check_packed(*child, 1, 0, true, tail_k);
    check_packed(*child, 0, 1, false, std::vector<uint8_t>(4 * 11, 29));
    const auto last_owner = allocations(*child);
    child->write_layer_v_gpu(0, 1, 1, other_v.data());
    CHECK(allocations(*child) == last_owner);
    check_packed(*deep, 1, 0, true, std::vector<uint8_t>(2 * 7, 17));
    child->clear();
    CHECK(child->bytes_k() == 0 && child->bytes_v() == 0);
    CHECK(allocation_bytes(allocations(*child)) < allocation_bytes(last_owner));
    // Retained vector capacity remains charged after clear.
    CHECK(allocation_bytes(allocations(*child)) > sizeof(kvmem::RawKvStore));
}

static void test_cow_boundaries() {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 1;
    cfg.block_tokens = 128;
    cfg.k_gpu_row_bytes = cfg.v_gpu_row_bytes = 3;
    kvmem::RawKvStore parent(cfg);
    std::vector<uint8_t> rows(256 * 3, 42), replacement(2 * 3, 99);
    parent.write_layer_k_gpu(0, 256, 0, rows.data());
    parent.write_layer_v_gpu(0, 256, 0, rows.data());
    for (uint32_t pos : {0u, 1u, 127u, 128u, 129u, 256u}) {
        auto branch = parent.clone_prefix(pos);
        const auto kept = (pos + 127) / 128;
        CHECK(shared_allocations(parent, *branch).size() == kept * 4);
        CHECK(branch->n_tokens(pos / 128) == pos % 128);
        branch->write_layer_k_gpu(pos, 2, 0, replacement.data());
        branch->write_layer_v_gpu(pos, 2, 0, replacement.data());
        check_packed(parent, 0, 0, true, std::vector<uint8_t>(128 * 3, 42));
        check_packed(parent, 1, 0, false, std::vector<uint8_t>(128 * 3, 42));
        for (uint32_t bid = 0; bid * 128 < pos + 2; ++bid) {
            const uint32_t n = std::min(128u, pos + 2 - bid * 128);
            std::vector<uint8_t> expected(n * 3);
            for (uint32_t t = 0; t < n; ++t) {
                std::fill_n(expected.begin() + t * 3, 3, bid * 128 + t < pos ? 42 : 99);
            }
            check_packed(*branch, bid, 0, true, expected);
            check_packed(*branch, bid, 0, false, expected);
        }
    }
}

static void test_legacy_and_mean_isolation() {
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd_k = cfg.n_embd_v = 3;
    cfg.block_tokens = 4;
    cfg.k_gpu_row_bytes = cfg.v_gpu_row_bytes = 7;
    kvmem::RawKvStore parent(cfg);
    std::vector<float> ones(6 * 3, 1.0f), twos(6 * 3, 2.0f), got(4 * 3);
    std::vector<uint16_t> halves(6 * 3, 0x4000); // 2.0
    std::vector<uint8_t> packed(6 * 7, 31);
    parent.write_layer_tokens(0, 6, 0, ones.data(), ones.data());
    parent.write_layer_k_gpu(0, 6, 0, packed.data());
    parent.write_layer_v_gpu(0, 6, 0, packed.data());
    auto a = parent.clone_prefix(6);
    auto b = parent.clone_prefix(6);
    a->write_layer_tokens(3, 3, 0, twos.data(), twos.data());
    b->write_layer_tokens_f16(3, 3, 0, halves.data(), halves.data());
    for (auto * branch : {a.get(), b.get()}) {
        CHECK(branch->copy_k(0, 0, got.data()));
        CHECK(got[0] == 1.0f && got[9] == 2.0f);
        CHECK(branch->copy_v(1, 0, got.data()) && got[0] == 2.0f);
        CHECK(!branch->has_v_gpu(0, 0));
        CHECK(shared_allocations(parent, *branch).size() == 2 * 2); // Only packed K remains shared.
        // Switching packed V -> F32/F16 V -> packed V must not resurrect the
        // old valid count. Exercise both a full block and a partial tail.
        for (uint32_t bid : {0u, 1u}) {
            const uint32_t old_rows = bid == 0 ? 4 : 2;
            const uint32_t pos = bid * cfg.block_tokens;
            std::vector<uint8_t> replacement(old_rows * 7, 99), copied(old_rows * 7, 211);
            branch->write_layer_v_gpu(pos, 1, 0, replacement.data());
            CHECK(branch->n_tokens(bid) == old_rows);
            CHECK(branch->has_v_gpu(bid, 0, 1));
            CHECK(!branch->has_v_gpu(bid, 0, 2));
            CHECK(!branch->copy_v_gpu(bid, 0, copied.data(), old_rows));
            CHECK(copied == std::vector<uint8_t>(old_rows * 7, 211));
            check_packed(*branch, bid, 0, false, std::vector<uint8_t>(7, 99));
            check_packed(parent, bid, 0, false, std::vector<uint8_t>(old_rows * 7, 31));
            branch->write_layer_v_gpu(pos + 1, old_rows - 1, 0, replacement.data() + 7);
            check_packed(*branch, bid, 0, false, replacement);
            check_packed(parent, bid, 0, false, std::vector<uint8_t>(old_rows * 7, 31));
        }
    }
    CHECK(parent.copy_k(0, 0, got.data()) && got[9] == 1.0f);
    check_packed(parent, 1, 0, false, std::vector<uint8_t>(2 * 7, 31));
    a->truncate_to(1);
    a->mean_k(0, 0, got.data());
    CHECK(got[0] == 1.0f);
    parent.mean_k(0, 0, got.data());
    CHECK(got[0] == 1.0f);
    b->write_layer_mean_k(0, 4, 0, twos.data());
    b->mean_k(0, 0, got.data());
    CHECK(got[0] == 2.0f);
    parent.mean_k(0, 0, got.data());
    CHECK(got[0] == 1.0f);

    // Opaque raw rows and their statistics are copied privately as well.
    cfg.k_row_bytes = 5;
    kvmem::RawKvStore opaque(cfg);
    std::vector<uint8_t> rows(6 * 5, 13), changed(3 * 5, 77), copied(4 * 5);
    opaque.write_layer_k_rows(0, 6, 1, rows.data(), ones.data());
    auto c = opaque.clone_prefix(6);
    CHECK(shared_allocations(opaque, *c).empty());
    c->write_layer_k_rows(3, 3, 1, changed.data(), twos.data());
    CHECK(c->copy_k_rows(0, 1, copied.data(), 4) && copied[0] == 13 && copied[15] == 77);
    CHECK(opaque.copy_k_rows(0, 1, copied.data(), 4) && copied[15] == 13);
    opaque.mean_k(0, 1, got.data());
    CHECK(got[0] == 1.0f);
}

int main() {
    test_packed_cow();
    test_cow_boundaries();
    test_legacy_and_mean_isolation();
    test_sum_only(32);
    test_sum_only(128);
    kvmem::RawKvStoreConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd_k = 4;
    cfg.n_embd_v = 4;
    cfg.block_tokens = 4;
    kvmem::RawKvStore raw(cfg);

    std::vector<float> k(8, 0.0f);
    std::vector<float> v(8, 1.0f);
    for (int i = 0; i < 8; ++i) {
        k[i] = static_cast<float>(i);
    }
    raw.write_layer_tokens(0, 2, 0, k.data(), nullptr);
    CHECK(raw.n_tokens(0) == 2);
    CHECK(raw.has_k(0, 0));
    CHECK(!raw.has_v(0, 0));
    // The K sum is captured at write so RAM scoring does not copy_k.
    std::vector<float> got(8, -1.0f);
    CHECK(raw.copy_k(0, 0, got.data()));
    CHECK(got[0] == 0.0f);
    CHECK(got[5] == 5.0f);
    CHECK(raw.bytes_v() == 0);
    CHECK(raw.bytes_k() > 0);

    raw.write_layer_tokens(0, 2, 0, nullptr, v.data());
    CHECK(raw.has_v(0, 0));
    std::vector<float> gv(8, 0.0f);
    CHECK(raw.copy_v(0, 0, gv.data()));
    CHECK(std::fabs(gv[0] - 1.0f) < 1e-3f);

    std::vector<float> mean(4, 0.0f);
    raw.mean_k(0, 0, mean.data());
    CHECK(std::fabs(mean[0] - 2.0f) < 1e-3f); // (0+4)/2
    CHECK(std::fabs(mean[1] - 3.0f) < 1e-3f); // (1+5)/2

    std::vector<float> k2(8, 0.0f);
    for (int i = 0; i < 8; ++i) {
        k2[i] = static_cast<float>(i + 8);
    }
    raw.write_layer_tokens(2, 2, 0, k2.data(), nullptr);
    CHECK(raw.n_tokens(0) == 4);
    raw.mean_k(0, 0, mean.data());
    CHECK(std::fabs(mean[0] - 6.0f) < 1e-3f); // (0+4+8+12)/4
    CHECK(std::fabs(mean[1] - 7.0f) < 1e-3f); // (1+5+9+13)/4

    kvmem::RopeConfig rc;
    rc.n_rot = 4;
    rc.n_embd_head = 4;
    rc.n_head_kv = 1;
    rc.freq_base = 10000.0f;
    std::vector<float> src(4, 0.0f);
    src[0] = 1.0f;
    src[2] = 1.0f;
    std::vector<float> dst(4, 0.0f);
    kvmem::rope_neox_apply(rc, src.data(), 1, 0, dst.data());
    // pos=0 → identity rotation
    CHECK(std::fabs(dst[0] - 1.0f) < 1e-5f);
    CHECK(std::fabs(dst[2] - 1.0f) < 1e-5f);

    kvmem::rope_neox_apply(rc, src.data(), 1, 1, dst.data());
    CHECK(std::fabs(dst[0] - src[0]) > 1e-6f || std::fabs(dst[2] - src[2]) > 1e-6f);

    // Hybrid: only attention layers are captured; layer 0 may stay empty.
    kvmem::RawKvStore raw_h(cfg);
    raw_h.write_layer_tokens(0, 2, 1, k.data(), nullptr);
    CHECK(raw_h.has_block(0));
    CHECK(raw_h.n_tokens(0) == 2);
    CHECK(!raw_h.has_k(0, 0));
    CHECK(raw_h.has_k(0, 1));
    std::vector<float> hgot(8, -1.0f);
    CHECK(raw_h.copy_k(0, 1, hgot.data()));
    CHECK(hgot[0] == 0.0f);

#if KVMEM_ENABLE_NVME
    kvmem::RawKvStoreConfig ncfg = cfg;
    ncfg.nvme_dir = (std::filesystem::temp_directory_path() / "kvmem_raw_k_test").string();
    ncfg.nvme_file = "raw.bin";
    ncfg.nvme_bytes = 4ull * 1024ull * 1024ull;
    kvmem::RawKvStore rawn(ncfg);
    CHECK(rawn.nvme_enabled());
    for (bool deep : {false, true}) {
        bool rejected = false;
        try {
            auto clone = deep ? rawn.clone_prefix_deep(1) : rawn.clone_prefix(1);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        CHECK(rejected);
    }
    std::vector<float> kfull(16, 0.0f);
    for (int i = 0; i < 16; ++i) {
        kfull[i] = static_cast<float>(i);
    }
    rawn.write_layer_tokens(0, 4, 0, kfull.data(), nullptr);
    CHECK(rawn.has_k(0, 0));
    std::vector<float> nout(16, -1.0f);
    CHECK(rawn.copy_k(0, 0, nout.data()));
    CHECK(nout[0] == 0.0f);
    CHECK(nout[15] == 15.0f);
    std::vector<float> nmean(4, 0.0f);
    rawn.mean_k(0, 0, nmean.data());
    CHECK(std::fabs(nmean[0] - 6.0f) < 1e-2f); // (0+4+8+12)/4

#endif

    // F16 write: 0x3c00 is 1.0 in IEEE half.
    kvmem::RawKvStore raw16(cfg);
    std::vector<uint16_t> ones(8, 0x3c00);
    raw16.write_layer_tokens_f16(0, 2, 0, ones.data(), nullptr);
    std::vector<float> f16out(8, 0.0f);
    CHECK(raw16.copy_k(0, 0, f16out.data()));
    CHECK(std::fabs(f16out[0] - 1.0f) < 1e-3f);
    CHECK(std::fabs(f16out[7] - 1.0f) < 1e-3f);
    std::vector<float> m16(4, 0.0f);
    raw16.mean_k(0, 0, m16.data());
    CHECK(std::fabs(m16[0] - 1.0f) < 1e-3f);
    CHECK(std::fabs(m16[3] - 1.0f) < 1e-3f);

    kvmem::RawKvStoreConfig gcfg = cfg;
    gcfg.v_gpu_row_bytes = 6;
    kvmem::RawKvStore rawg(gcfg);
    std::vector<uint8_t> packed(12);
    for (int i = 0; i < 12; ++i) {
        packed[static_cast<size_t>(i)] = static_cast<uint8_t>(i + 1);
    }
    rawg.write_layer_v_gpu(0, 2, 0, packed.data());
    CHECK(rawg.has_v(0, 0));
    CHECK(rawg.has_v_gpu(0, 0));
    CHECK(!rawg.has_block(0));
    kvmem::RawKvStoreConfig kgcfg = cfg;
    kgcfg.k_gpu_row_bytes = 6;
    kvmem::RawKvStore rawkg(kgcfg);
    rawkg.write_layer_k_gpu(0, 2, 0, packed.data());
    CHECK(rawkg.has_k_gpu(0, 0));
    CHECK(rawkg.has_block(0));
    CHECK(!rawkg.has_k(0, 0));
    std::vector<uint8_t> kgout(12, 0);
    CHECK(rawkg.copy_k_gpu(0, 0, kgout.data(), 2));
    CHECK(kgout[0] == 1);
    CHECK(kgout[11] == 12);

    kvmem::RawKvStore rawm(cfg);
    rawm.write_layer_mean_k(0, 2, 0, k.data());
    CHECK(rawm.has_block(0));
    CHECK(!rawm.has_k(0, 0));
    CHECK(!rawm.has_k_gpu(0, 0));
    std::vector<float> mm(4, 0.0f);
    rawm.mean_k(0, 0, mm.data());
    CHECK(std::fabs(mm[0] - 2.0f) < 1e-3f);
    rawm.clear();
    CHECK(!rawm.has_block(0));
    rawm.write_layer_mean_k(0, 1, 0, k.data());
    std::vector<float> mm0(4, 0.0f);
    rawm.mean_k(0, 0, mm0.data());
    CHECK(std::fabs(mm0[0] - k[0]) < 1e-3f);
    rawm.write_layer_mean_k(4, 1, 0, k.data() + 4);
    CHECK(rawm.has_block(1));
    rawm.truncate_to(4);
    CHECK(rawm.has_block(0));
    CHECK(!rawm.has_block(1));
    kvmem::RawKvStore raws(cfg);
    raws.write_layer_mean_k(0, 1, 0, k.data());
    raws.write_layer_mean_k(1, 1, 0, k.data() + 4);
    std::vector<float> ms(4, 0.0f);
    raws.mean_k(0, 0, ms.data());
    CHECK(std::fabs(ms[0] - 2.0f) < 1e-3f);
    auto sumcfg = cfg;
    sumcfg.k_gpu_row_bytes = 6;
    kvmem::RawKvStore rawsum(sumcfg);
    std::vector<float> ksum(4, 0.0f);
    for (int d = 0; d < 4; ++d) {
        ksum[static_cast<size_t>(d)] = k[static_cast<size_t>(d)] + k[static_cast<size_t>(d + 4)];
    }
    rawsum.write_layer_mean_sum(0, 2, 0, ksum.data());
    std::vector<float> msum(4, 0.0f);
    rawsum.mean_k(0, 0, msum.data());
    CHECK(std::fabs(msum[0] - 2.0f) < 1e-3f);
    // Packed rows can run ahead of captured statistics: normalize by mean_tokens.
    const std::vector<uint8_t> sum_packed(24, 0);
    rawsum.write_layer_k_gpu(0, 4, 0, sum_packed.data());
    rawsum.mean_k(0, 0, msum.data());
    CHECK(msum[0] == 2.0f);
    rawsum.write_layer_mean_sum(2, 1, 0, k2.data());
    rawsum.mean_k(0, 0, msum.data());
    CHECK(msum[0] == (ksum[0] + k2[0]) * (1.0f / 3.0f));
    std::vector<uint8_t> gout(12, 0);
    CHECK(rawg.copy_v_gpu(0, 0, gout.data(), 2));
    CHECK(gout[0] == 1);
    CHECK(gout[11] == 12);
    std::vector<float> no_f32(8, 0.0f);
    CHECK(!rawg.copy_v(0, 0, no_f32.data()));

    kvmem::RawKvStoreConfig ngcfg = cfg;
    ngcfg.v_gpu_row_bytes = 6;
    ngcfg.nvme_dir = (std::filesystem::temp_directory_path() / "kvmem_raw_vgpu_test").string();
    ngcfg.nvme_file = "raw_vgpu.bin";
    ngcfg.nvme_bytes = 4ull * 1024ull * 1024ull;
    std::vector<uint8_t> packed4(24);
    for (int i = 0; i < 24; ++i) {
        packed4[static_cast<size_t>(i)] = static_cast<uint8_t>(i + 1);
    }
    std::vector<uint8_t> gout4(24, 0);
#if KVMEM_ENABLE_NVME
    kvmem::RawKvStore rawgn(ngcfg);
    CHECK(rawgn.nvme_enabled());
    rawgn.write_layer_v_gpu(0, 4, 0, packed4.data());
    CHECK(rawgn.has_v(0, 0));
    CHECK(rawgn.copy_v_gpu(0, 0, gout4.data(), 4));
    CHECK(gout4[0] == 1);
    CHECK(gout4[23] == 24);
    CHECK(!rawgn.copy_v(0, 0, no_f32.data()));

#endif

    kvmem::RawKvStoreConfig kcfg = cfg;
    kcfg.k_row_bytes = 6;
    kvmem::RawKvStore rawk(kcfg);
    rawk.write_layer_k_rows(0, 2, 0, packed.data(), k.data());
    CHECK(rawk.has_k(0, 0));
    std::vector<uint8_t> kout(12, 0);
    CHECK(rawk.copy_k_rows(0, 0, kout.data(), 2));
    CHECK(kout[0] == 1);
    CHECK(kout[11] == 12);
    CHECK(!rawk.copy_k(0, 0, no_f32.data()));
    std::vector<float> mk(4, 0.0f);
    rawk.mean_k(0, 0, mk.data());
    CHECK(std::fabs(mk[0] - 2.0f) < 1e-3f);

#if KVMEM_ENABLE_NVME
    kvmem::RawKvStoreConfig nkcfg = cfg;
    nkcfg.k_row_bytes = 6;
    nkcfg.nvme_dir = (std::filesystem::temp_directory_path() / "kvmem_raw_krow_test").string();
    nkcfg.nvme_file = "raw_krow.bin";
    nkcfg.nvme_bytes = 4ull * 1024ull * 1024ull;
    kvmem::RawKvStore rawkn(nkcfg);
    CHECK(rawkn.nvme_enabled());
    std::vector<float> krowf(16, 0.0f);
    for (int i = 0; i < 16; ++i) {
        krowf[static_cast<size_t>(i)] = static_cast<float>(i);
    }
    rawkn.write_layer_k_rows(0, 4, 0, packed4.data(), krowf.data());
    CHECK(rawkn.has_k(0, 0));
    std::vector<uint8_t> kout4(24, 0);
    CHECK(rawkn.copy_k_rows(0, 0, kout4.data(), 4));
    CHECK(kout4[0] == 1);
    CHECK(kout4[23] == 24);
    CHECK(!rawkn.copy_k(0, 0, no_f32.data()));
    std::vector<float> krowmean(4, 0.0f);
    rawkn.mean_k(0, 0, krowmean.data());
    CHECK(std::fabs(krowmean[0] - 6.0f) < 1e-3f);

#endif

    // Replacing a suffix preserves prefix bytes and rejects stale packed rows,
    // even if the mean-K capture has already extended the logical block.
    for (bool nvme : {false, true}) {
        if (nvme && !KVMEM_ENABLE_NVME) continue;
        auto tail_cfg = ngcfg;
        tail_cfg.k_gpu_row_bytes = 6;
        if (!nvme) tail_cfg.nvme_bytes = 0;
        kvmem::RawKvStore tail_store(tail_cfg);
        tail_store.write_layer_mean_k(0, 2, 0, k.data());
        const auto checkpoint = tail_store.mean_checkpoint(2);
        tail_store.write_layer_mean_k(2, 2, 0, k2.data());
        tail_store.write_layer_k_gpu(0, 4, 0, packed4.data());
        tail_store.write_layer_v_gpu(0, 4, 0, packed4.data());
        tail_store.truncate_to(2);
        tail_store.restore_mean_checkpoint(2, checkpoint);
        CHECK(tail_store.n_tokens(0) == 2);
        CHECK(!tail_store.copy_k_gpu(0, 0, gout4.data(), 4));
        CHECK(!tail_store.copy_v_gpu(0, 0, gout4.data(), 4));
        CHECK(tail_store.copy_k_gpu(0, 0, gout4.data(), 2));
        CHECK(std::memcmp(gout4.data(), packed4.data(), 12) == 0);
        std::vector<float> replacement(8, 20.0f);
        tail_store.write_layer_mean_k(2, 2, 0, replacement.data());
        CHECK(!tail_store.has_k_gpu(0, 0, 4));
        CHECK(!tail_store.has_v_gpu(0, 0, 4));
        tail_store.mean_k(0, 0, mean.data());
        CHECK(std::fabs(mean[0] - 11.0f) < 1e-3f);
        std::vector<uint8_t> new_bytes(12, 231);
        tail_store.write_layer_k_gpu(2, 2, 0, new_bytes.data());
        tail_store.write_layer_v_gpu(2, 2, 0, new_bytes.data());
        for (bool is_k : {false, true}) {
            CHECK(is_k ? tail_store.copy_k_gpu(0, 0, gout4.data(), 4)
                       : tail_store.copy_v_gpu(0, 0, gout4.data(), 4));
            CHECK(std::memcmp(gout4.data(), packed4.data(), 12) == 0);
            CHECK(std::memcmp(gout4.data() + 12, new_bytes.data(), 12) == 0);
        }
        tail_store.invalidate_packed_from(3);
        CHECK(!tail_store.has_k_gpu(0, 0, 4));
        CHECK(!tail_store.has_v_gpu(0, 0, 4));
        CHECK(tail_store.copy_k_gpu(0, 0, gout4.data(), 3));
        CHECK(tail_store.copy_v_gpu(0, 0, gout4.data(), 3));
    }
    {
        // clone_prefix: shared first rows, partial tail block kept.
        kvmem::RawKvStoreConfig ccfg;
        ccfg.n_layer = 1;
        ccfg.n_embd_k = 4;
        ccfg.n_embd_v = 4;
        ccfg.block_tokens = 2;
        ccfg.k_gpu_row_bytes = 3;
        ccfg.v_gpu_row_bytes = 3;
        kvmem::RawKvStore src(ccfg);
        std::vector<uint8_t> rows(15);
        for (int i = 0; i < 15; ++i) rows[static_cast<size_t>(i)] = static_cast<uint8_t>(i + 1);
        for (uint32_t pos = 0; pos < 5; pos += 2) {
            const uint32_t n = pos + 2 <= 5 ? 2 : 1;
            src.write_layer_k_gpu(pos, n, 0, rows.data() + pos * 3);
            src.write_layer_v_gpu(pos, n, 0, rows.data() + pos * 3);
        }
        auto clone = src.clone_prefix(3);
        CHECK(clone->has_k_gpu(0, 0, 2) && clone->has_v_gpu(0, 0, 2));
        CHECK(clone->has_k_gpu(1, 0, 1) && !clone->has_k_gpu(1, 0, 2));
        CHECK(!clone->has_k_gpu(2, 0, 1));
        CHECK(src.has_k_gpu(1, 0, 2) && src.has_k_gpu(2, 0, 1));
        std::vector<uint8_t> got(6, 0);
        CHECK(clone->copy_k_gpu(1, 0, got.data(), 1));
        CHECK(got[0] == 7 && got[2] == 9);
        std::vector<uint8_t> other(6, 200);
        clone->write_layer_k_gpu(0, 2, 0, other.data());
        CHECK(src.copy_k_gpu(0, 0, got.data(), 2));
        CHECK(got[0] == 1 && got[5] == 6);
        CHECK(src.clone_prefix(0)->bytes_k() == 0);
    }
    return 0;
}
