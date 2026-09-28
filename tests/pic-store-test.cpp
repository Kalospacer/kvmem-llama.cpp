#include "../src/adapter/llama-kvmem-pic-store.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace {

using namespace kvmem_pic;

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

template<typename F>
void rejects(F && fn) {
    try { fn(); } catch (const std::exception &) { return; }
    throw std::runtime_error("expected rejection");
}

segment_key exact_key() {
    segment_key key;
    key.model = "test-model";
    key.configuration = "test-layout-rope";
    key.model_generation = 3;
    key.configuration_generation = 7;
    key.origin = provenance::exact;
    return key;
}

std::vector<int32_t> tokens(size_t size, int32_t offset = 0) {
    std::vector<int32_t> result(size);
    for (size_t i = 0; i < size; ++i) result[i] = offset + int32_t(i);
    return result;
}

struct raw_data {
    explicit raw_data(size_t rows) : values(rows * 2, 0.25f) {}
    std::vector<float> values;
};

data_owner raw_owner(const std::shared_ptr<raw_data> & raw) {
    data_owner owner;
    owner.owner = raw;
    owner.allocations = [](const void * ptr, allocation_list & out) {
        const auto & raw = *static_cast<const raw_data *>(ptr);
        out.emplace_back(ptr, sizeof(raw));
        out.emplace_back(raw.values.data(), raw.values.capacity() * sizeof(float));
        // Duplicate identities from multiple paths must not increase the bill.
        out.emplace_back(raw.values.data(), raw.values.capacity() * sizeof(float));
        return true;
    };
    return owner;
}

void fill(segment_builder & build, size_t rows, bool hidden = false,
        const std::shared_ptr<raw_data> & shared_raw = {}) {
    auto & data = build.data();
    for (int32_t il : {0, 2}) {
        segment_layer layer;
        layer.layer = il;
        layer.transition.d = 2;
        layer.transition.heads = 1;
        layer.transition.sequences = 1;
        layer.transition.t = {1, 0, 0, 1};
        layer.transition.u = {0.1f, 0.2f, 0.3f, 0.4f};
        layer.conv_end.ne = {3, 6, 1, 1};
        layer.conv_end.data.assign(18, float(il + 1));
        data.layers.push_back(std::move(layer));
    }
    if (hidden) {
        data.target_hidden.ne = {2, int64_t(rows), 1, 1};
        data.target_hidden.data.assign(rows * 2, 0.5f);
    }
    data.raw = raw_owner(shared_raw ? shared_raw : std::make_shared<raw_data>(rows));
    data.raw_rows = rows;
}

publish_result publish(store & cache, const std::vector<int32_t> & full, token_range range,
        bool hidden = false, const std::shared_ptr<raw_data> & raw = {}, segment_key key = exact_key()) {
    auto build = cache.begin(std::move(key), full, range, {0, 2}, hidden);
    fill(build, range.end - range.begin, hidden, raw);
    return cache.publish(std::move(build));
}

void test_payload_and_publication() {
    store cache;
    const auto full = tokens(1100);
    const auto key = exact_key();
    auto build = cache.begin(key, full, {20, 1044}, {2, 0}, true);
    require(cache.match(key, full, {{0, full.size()}}).empty(), "unfilled build became visible");
    fill(build, 1024, true);
    require(cache.match(key, full, {{0, full.size()}}).empty(), "unpublished product became visible");
    auto result = cache.publish(std::move(build));
    require(bool(result) && result.evicted == 0 && result.reason.empty(), "complete publication failed");
    rejects([&] { build.data(); });
    auto matches = cache.match(key, full, {{0, full.size()}}, true);
    require(matches.size() == 1 && matches[0].range.begin == 20 && matches[0].range.end == 1044,
            "full-token source range was not preserved");
    const auto & entry = *matches[0].entry;
    require(entry.id == result.entry->id && entry.tokens.size() == 1024 && entry.data.raw_rows == 1024,
            "entry lost tokens or raw rows");
    require(entry.data.layers[1].layer == 2 && entry.data.layers[1].transition.u[3] == 0.4f &&
            entry.data.layers[1].conv_end.data[0] == 3 && entry.data.target_hidden.data.size() == 2048,
            "entry contains metadata without the constructed products");
    const auto raw = std::static_pointer_cast<const raw_data>(entry.data.raw.owner);
    require(raw->values.size() == 2048 && raw->values[0] == 0.25f, "raw data owner was lost");
    std::vector<int32_t> shifted(13, -1);
    shifted.insert(shifted.end(), entry.tokens.begin(), entry.tokens.end());
    const auto shifted_hits = cache.match(key, shifted, {{0, shifted.size()}}, true);
    require(shifted_hits.size() == 1 && shifted_hits[0].range.begin == 13,
            "matching depended on original request position");
}

void test_invalid_products() {
    store cache;
    const auto full = tokens(512);
    const auto key = exact_key();
    rejects([&] { cache.begin(key, full, {0, 511}, {0, 2}); });
    rejects([&] { cache.begin(key, full, {4, 513}, {0, 2}); });
    rejects([&] { cache.begin(key, full, {100, 50}, {0, 2}); });
    rejects([&] { cache.begin(key, full, {0, 512}, {}); });
    rejects([&] { cache.begin(key, full, {0, 512}, {0, 0}); });
    rejects([&] { cache.begin(key, full, {0, 512}, {-1}); });
    rejects([&] { cache.begin({}, full, {0, 512}, {0}); });
    auto invalid = [&](const std::function<void(segment_data &)> & damage) {
        auto build = cache.begin(key, full, {0, 512}, {0, 2}, true);
        fill(build, 512, true);
        damage(build.data());
        rejects([&] { cache.publish(std::move(build)); });
        require(cache.size() == 0 && cache.bytes() == 0, "invalid product partially published");
    };
    invalid([](segment_data & d) { d.layers.pop_back(); });
    invalid([](segment_data & d) { d.layers[1].layer = 0; });
    invalid([](segment_data & d) { d.layers[0].transition.t.pop_back(); });
    invalid([](segment_data & d) { d.layers[0].transition.u[0] = std::numeric_limits<float>::quiet_NaN(); });
    invalid([](segment_data & d) { d.layers[0].transition.d = INT64_MAX; });
    invalid([](segment_data & d) { d.layers[0].transition.sequences = 2; });
    invalid([](segment_data & d) { d.layers[0].conv_end.data.pop_back(); });
    invalid([](segment_data & d) { d.raw_rows = 511; });
    invalid([](segment_data & d) { d.raw.owner.reset(); });
    invalid([](segment_data & d) {
        static int borrowed;
        d.raw.owner = std::shared_ptr<const void>(std::shared_ptr<const void>{}, &borrowed);
    });
    invalid([](segment_data & d) { d.target_hidden = {}; });
    invalid([](segment_data & d) { d.target_hidden.ne[1] = 511; });
    invalid([](segment_data & d) { d.raw.allocations = [](const void *, allocation_list &) { return false; }; });
    invalid([](segment_data & d) { d.raw.allocations = [](const void *, allocation_list &) -> bool {
        throw std::runtime_error("allocation collection failed");
    }; });
    invalid([](segment_data & d) { d.raw.allocations = [](const void *, allocation_list &) { return true; }; });
    invalid([](segment_data & d) { d.raw.allocations = [](const void *, allocation_list & out) {
        out.emplace_back(nullptr, 1024); return true;
    }; });
    auto valid = cache.begin(key, full, {0, 512}, {0, 2});
    fill(valid, 512);
    for (auto & layer : valid.data().layers) {
        layer.conv_end.ne[0] = 0;
        layer.conv_end.data.clear();
    }
    require(bool(cache.publish(std::move(valid))), "zero-history convolution was rejected");
    require(cache.match(key, full, {{0, 512}}, true).empty(), "missing MTP hidden was accepted");
}

void test_longest_ranges_and_collisions() {
    store cache;
    const auto full = tokens(2400);
    const auto key = exact_key();
    require(bool(publish(cache, full, {0, 700})), "publish short segment");
    require(bool(publish(cache, full, {200, 1200})), "publish long segment");
    require(bool(publish(cache, full, {1200, 1800})), "publish adjacent segment");
    require(bool(publish(cache, full, {1800, 2400})), "publish last segment");
    auto hits = cache.match(key, full, {{1800, 2400}, {0, 1800}, {0, 1800}});
    require(hits.size() == 3 && hits[0].range.begin == 200 && hits[0].range.end == 1200 &&
            hits[1].range.begin == 1200 && hits[2].range.begin == 1800, "longest non-overlapping selection failed");
    require(cache.match(key, full, {{0, 400}, {400, 800}}).empty(), "match crossed an allowed-range boundary");
    require(cache.match(key, full, {}).empty(), "empty allowlist granted matches");
    rejects([&] { cache.match(key, full, {{0, 2401}}); });
    rejects([&] { cache.match(key, full, {{9, 8}}); });

    store repeated;
    const auto block = tokens(512);
    require(bool(publish(repeated, block, {0, 512})), "publish repeated block");
    auto request = block;
    request.insert(request.end(), block.begin(), block.end());
    auto two = repeated.match(key, request, {{0, 1024}});
    require(two.size() == 2 && two[0].entry == two[1].entry && two[1].range.begin == 512,
            "same segment could not match twice at disjoint positions");

    store collision;
    std::vector<int32_t> a(512, 19);
    a[0] = 1; a[1] = 0;
    auto b = a;
    b[0] = 0; b[1] = 257;
    // Both anchor and full polynomial hashes collide: 1*257 + 0 == 0*257 + 257.
    require(bool(publish(collision, a, {0, 512})), "publish collision fixture");
    require(collision.match(key, b, {{0, 512}}).empty(), "hash collision bypassed token verification");
    require(collision.match(key, a, {{0, 512}}).size() == 1, "collision handling lost the real match");
}

void test_key_and_provenance_isolation() {
    store cache;
    const auto full = tokens(512);
    const auto key = exact_key();
    require(bool(publish(cache, full, {0, 512})), "publish exact");
    auto miss = [&](segment_key other) { require(cache.match(other, full, {{0, 512}}).empty(), "key isolation failed"); };
    auto other = key; other.model += "-other"; miss(other);
    other = key; ++other.model_generation; miss(other);
    other = key; other.configuration += "-other"; miss(other);
    other = key; ++other.configuration_generation; miss(other);
    other = key; ++other.format_generation; miss(other);
    other = key; other.origin = provenance::approximate; other.approximation = "source-a:method-v1"; miss(other);
    require(bool(publish(cache, full, {0, 512}, false, {}, other)), "publish approximate");
    auto hits = cache.match(other, full, {{0, 512}});
    require(hits.size() == 1 && hits[0].entry->key.origin == provenance::approximate, "approximate mixed with exact");
    hits = cache.match(key, full, {{0, 512}});
    require(hits.size() == 1 && hits[0].entry->key.origin == provenance::exact, "exact mixed with approximate");
    other.approximation = "source-b:method-v1"; miss(other);
    other.approximation.clear();
    rejects([&] { cache.match(other, full, {{0, 512}}); });
    other = key; other.approximation = "not-exact";
    rejects([&] { cache.begin(other, full, {0, 512}, {0, 2}); });
}

uint64_t entry_bytes() {
    store cache;
    const auto full = tokens(512);
    require(bool(publish(cache, full, {0, 512})), "sizing publication failed");
    return cache.bytes();
}

void test_lru_budget_and_pins() {
    require(store().budget() == maximum_store_bytes, "default budget differs from 2 GiB");
    rejects([] { store oversized(maximum_store_bytes + 1); });
    const auto a = tokens(512), b = tokens(512, 2000), c = tokens(512, 4000);
    const auto key = exact_key();
    const auto cost = entry_bytes();
    store cache(2 * cost);
    require(bool(publish(cache, a, {0, 512})), "publish A");
    require(bool(publish(cache, b, {0, 512})), "publish B");
    require(cache.match(key, a, {{0, 512}}).size() == 1, "touch A");
    auto third = publish(cache, c, {0, 512});
    require(bool(third) && third.evicted == 1 && cache.size() == 2 && cache.bytes() <= cache.budget(), "LRU budget failure");
    require(cache.match(key, b, {{0, 512}}).empty(), "LRU did not evict untouched B");
    require(cache.match(key, a, {{0, 512}}).size() == 1, "LRU evicted recently used A");

    store pinned(cost);
    auto first = publish(pinned, a, {0, 512});
    require(bool(first), "publish pinned A");
    auto rejected = publish(pinned, b, {0, 512});
    require(!rejected && !rejected.reason.empty() && rejected.evicted == 0 && pinned.size() == 1,
            "budget rejection evicted a live pinned entry");
    require(pinned.match(key, a, {{0, 512}}).size() == 1, "failed publication changed index");
    pinned.clear();
    require(pinned.size() == 0 && pinned.bytes() == cost && pinned.match(key, a, {{0, 512}}).empty(),
            "evicted pinned allocation disappeared from accounting");
    require(!publish(pinned, b, {0, 512}), "pinned allocation did not enforce sub-budget");
    first.entry.reset();
    require(pinned.bytes() == 0 && bool(publish(pinned, b, {0, 512})), "released pin did not free budget");

    store disabled(0);
    require(!publish(disabled, a, {0, 512}) && disabled.size() == 0, "zero budget accepted publication");
    store too_small(cost - 1);
    require(!publish(too_small, a, {0, 512}), "oversized candidate accepted");

    store unchanged(2 * cost);
    require(bool(publish(unchanged, a, {0, 512})) && bool(publish(unchanged, b, {0, 512})), "prepare atomic rejection");
    auto large_raw = std::make_shared<raw_data>(512);
    large_raw->values.reserve(size_t(cost));
    require(!publish(unchanged, c, {0, 512}, false, large_raw) && unchanged.size() == 2 &&
            unchanged.match(key, a, {{0, 512}}).size() == 1 && unchanged.match(key, b, {{0, 512}}).size() == 1,
            "rejected oversized candidate evicted existing unpinned entries");
}

void test_unique_allocations_and_owner_lifetime() {
    store cache;
    auto shared = std::make_shared<raw_data>(512);
    std::weak_ptr<raw_data> weak = shared;
    const auto a = tokens(512), b = tokens(512, 1000);
    auto first = publish(cache, a, {0, 512}, false, shared);
    const auto one = cache.bytes();
    auto second = publish(cache, b, {0, 512}, false, shared);
    const auto raw_bytes = sizeof(raw_data) + shared->values.capacity() * sizeof(float);
    require(cache.bytes() == 2 * one - raw_bytes, "shared raw allocation was counted twice");
    {
        store tight(2 * one - raw_bytes);
        require(bool(publish(tight, a, {0, 512}, false, shared)) && bool(publish(tight, b, {0, 512}, false, shared)),
                "shared allocation was double-charged during admission");
        require(tight.size() == 2 && tight.bytes() == tight.budget(), "unique-byte admission evicted a shared entry");
    }
    std::unordered_map<const void *, size_t> unique;
    uint64_t sum = 0;
    for (const auto & allocation : cache.allocations()) {
        require(unique.emplace(allocation).second, "allocation callback exported duplicate identities");
        sum += allocation.second;
    }
    require(sum == cache.bytes() && unique.count(shared->values.data()) == 1, "allocation snapshot disagrees with budget");
    shared.reset();
    cache.clear();
    first.entry.reset();
    require(!weak.expired() && cache.bytes() == one, "shared owner died while second lease survived");
    second.entry.reset();
    require(weak.expired() && cache.bytes() == 0, "raw owner leaked after last lease");
}

// Keep the actual stash type opaque; only the supplied deleter sees this fixture.
struct stash_fixture { bool * freed; };
void free_stash_fixture(llama_kvmem_stash * ptr) {
    auto * fixture = reinterpret_cast<stash_fixture *>(ptr);
    *fixture->freed = true;
    delete fixture;
}
bool stash_fixture_allocations(const llama_kvmem_stash * ptr, allocation_list & out) {
    out.emplace_back(ptr, sizeof(stash_fixture));
    return true;
}

void test_stash_deleter() {
    bool freed = false;
    {
        auto * fixture = new stash_fixture{ &freed };
        auto owner = own_stash(reinterpret_cast<llama_kvmem_stash *>(fixture), free_stash_fixture, stash_fixture_allocations);
        auto lease = owner;
        owner = {};
        require(!freed, "stash released before final owner");
        allocation_list allocations;
        require(lease.allocations(lease.owner.get(), allocations) && allocations.size() == 1, "stash allocation bridge failed");
    }
    require(freed, "stash hooks deleter not called");
    freed = false;
    rejects([&] { own_stash(reinterpret_cast<llama_kvmem_stash *>(new stash_fixture{ &freed }), free_stash_fixture, nullptr); });
    require(freed, "stash leaked when callback validation failed");
}

void test_atomic_visibility() {
    store cache;
    const auto full = tokens(512);
    const auto key = exact_key();
    std::atomic<bool> done{false};
    std::atomic<bool> invalid{false};
    std::thread reader([&] {
        try {
            while (!done.load()) {
                for (const auto & hit : cache.match(key, full, {{0, 512}}, true)) {
                    if (hit.entry->data.layers.size() != 2 || hit.entry->data.target_hidden.data.size() != 1024 ||
                            !hit.entry->data.raw.owner) invalid = true;
                }
            }
        } catch (...) { invalid = true; }
    });
    try {
        for (int i = 0; i < 20; ++i) {
            auto build = cache.begin(key, full, {0, 512}, {0, 2}, true);
            std::this_thread::yield();
            fill(build, 512, true);
            require(bool(cache.publish(std::move(build))), "concurrent publication failed");
            cache.clear();
        }
    } catch (...) {
        done = true;
        reader.join();
        throw;
    }
    done = true;
    reader.join();
    require(!invalid.load(), "reader observed an incomplete publication");
}

} // namespace

int main() {
    try {
        test_payload_and_publication();
        test_invalid_products();
        test_longest_ranges_and_collisions();
        test_key_and_provenance_isolation();
        test_lru_budget_and_pins();
        test_unique_allocations_and_owner_lifetime();
        test_stash_deleter();
        test_atomic_visibility();
        std::puts("PIC store: passed");
        return 0;
    } catch (const std::exception & ex) {
        std::fprintf(stderr, "PIC store: %s\n", ex.what());
        return 1;
    }
}
