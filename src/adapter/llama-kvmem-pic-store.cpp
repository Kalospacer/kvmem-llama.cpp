#include "llama-kvmem-pic-store.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace kvmem_pic {
namespace {

void require(bool ok, const char * message) {
    if (!ok) throw std::invalid_argument(message);
}

void check_key(const segment_key & key) {
    require(!key.model.empty() && !key.configuration.empty() && key.format_generation != 0,
            "PIC store requires model, configuration and format keys");
    require((key.origin == provenance::exact && key.approximation.empty()) ||
            (key.origin == provenance::approximate && !key.approximation.empty()),
            "PIC exact and approximate provenance must be explicit and separate");
}

void check_range(token_range range, size_t count) {
    require(range.begin <= range.end && range.end <= count, "PIC token range is out of bounds");
}

size_t product(std::initializer_list<int64_t> dims) {
    size_t n = 1;
    for (int64_t d : dims) {
        require(d > 0 && uint64_t(d) <= SIZE_MAX / sizeof(float), "PIC tensor dimension is invalid");
        require(n <= SIZE_MAX / sizeof(float) / size_t(d), "PIC tensor dimensions overflow");
        n *= size_t(d);
    }
    return n;
}

void check_finite(const std::vector<float> & values) {
    require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }),
            "PIC segment contains non-finite data");
}

bool has_hidden(const segment & entry) {
    return !entry.data.target_hidden.data.empty();
}

void validate(segment & entry) {
    require(entry.data.raw.owner && entry.data.raw.owner.use_count() > 0 &&
            entry.data.raw.allocations && entry.data.raw_rows == entry.tokens.size(),
            "PIC segment requires owned raw data for every token");
    auto & layers = entry.data.layers;
    require(layers.size() == entry.recurrent_layers.size(), "PIC segment is missing recurrent layers");
    std::sort(layers.begin(), layers.end(), [](const segment_layer & a, const segment_layer & b) {
        return a.layer < b.layer;
    });
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto & layer = layers[i];
        require(layer.layer == entry.recurrent_layers[i], "PIC recurrent layer IDs differ from the build spec");
        const auto & tr = layer.transition;
        require(tr.sequences == 1, "PIC store segments must contain one sequence");
        const auto count = product({ tr.d, tr.d, tr.heads, tr.sequences });
        require(tr.t.size() == count && tr.u.size() == count, "PIC T/U data is incomplete");
        check_finite(tr.t);
        check_finite(tr.u);
        const auto & conv = layer.conv_end;
        require(conv.ne[0] >= 0 && conv.ne[1] > 0 && conv.ne[2] == 1 && conv.ne[3] == 1,
                "PIC convolution boundary shape is invalid");
        const size_t conv_count = conv.ne[0] == 0 ? 0 : product({ conv.ne[0], conv.ne[1] });
        require(conv.data.size() == conv_count, "PIC convolution boundary data is incomplete");
        check_finite(conv.data);
    }
    const auto & hidden = entry.data.target_hidden;
    if (entry.require_hidden || has_hidden(entry)) {
        require(hidden.ne[1] > 0 && uint64_t(hidden.ne[1]) == entry.tokens.size() &&
                hidden.ne[2] == 1 && hidden.ne[3] == 1, "PIC target hidden must cover the full segment");
        require(hidden.data.size() == product({ hidden.ne[0], hidden.ne[1] }), "PIC target hidden data is incomplete");
        check_finite(hidden.data);
    } else {
        require(hidden.ne == std::array<int64_t, 4>{}, "PIC absent target hidden must have an empty shape");
    }
}

// Unsigned arithmetic intentionally wraps modulo 2^64. Hashes only filter.
constexpr uint64_t hash_base = 257;

uint64_t token_value(int32_t token) { return uint64_t(uint32_t(token)) + 1; }

uint64_t token_hash(const std::vector<int32_t> & tokens, size_t count) {
    uint64_t value = 0;
    for (size_t i = 0; i < count; ++i) value = value * hash_base + token_value(tokens[i]);
    return value;
}

struct token_hashes {
    explicit token_hashes(const std::vector<int32_t> & tokens) : prefix(tokens.size() + 1), power(tokens.size() + 1, 1) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            prefix[i + 1] = prefix[i] * hash_base + token_value(tokens[i]);
            power[i + 1] = power[i] * hash_base;
        }
    }
    uint64_t get(size_t begin, size_t length) const { return prefix[begin + length] - prefix[begin] * power[length]; }
    std::vector<uint64_t> prefix;
    std::vector<uint64_t> power;
};

template<typename T>
void append_vector(allocation_list & out, const std::vector<T> & values) {
    if (values.capacity()) out.emplace_back(values.data(), values.capacity() * sizeof(T));
}

void append_string(allocation_list & out, const std::string & value) {
    const auto address = reinterpret_cast<uintptr_t>(value.data());
    const auto object = reinterpret_cast<uintptr_t>(&value);
    // Inline string storage is already included in sizeof(segment).
    if (address < object || address >= object + sizeof(value)) out.emplace_back(value.data(), value.capacity() + 1);
}

allocation_list entry_allocations(const segment & entry) {
    allocation_list out;
    out.emplace_back(&entry, sizeof(entry));
    append_string(out, entry.key.model);
    append_string(out, entry.key.configuration);
    append_string(out, entry.key.approximation);
    append_vector(out, entry.tokens);
    append_vector(out, entry.recurrent_layers);
    append_vector(out, entry.data.layers);
    append_vector(out, entry.data.target_hidden.data);
    for (const auto & layer : entry.data.layers) {
        append_vector(out, layer.transition.t);
        append_vector(out, layer.transition.u);
        append_vector(out, layer.conv_end.data);
    }
    allocation_list raw;
    if (!entry.data.raw.allocations(entry.data.raw.owner.get(), raw)) {
        throw std::runtime_error("PIC raw allocation callback failed");
    }
    require(std::any_of(raw.begin(), raw.end(), [](const std::pair<const void *, size_t> & a) {
        return a.first && a.second;
    }), "PIC raw owner has no reported allocation");
    out.insert(out.end(), raw.begin(), raw.end());
    return out;
}

struct allocation_union {
    void add(const allocation_list & list) {
        for (const auto & item : list) {
            if (!item.second) continue;
            require(item.first != nullptr, "PIC allocation identity is null");
            auto & size = values[item.first];
            size = std::max(size, item.second);
        }
    }
    uint64_t bytes() const {
        uint64_t total = 0;
        for (const auto & item : values) {
            if (item.second > UINT64_MAX - total) throw std::overflow_error("PIC allocation byte count overflow");
            total += item.second;
        }
        return total;
    }
    allocation_list list() const { return allocation_list(values.begin(), values.end()); }
    std::unordered_map<const void *, size_t> values;
};

} // namespace

bool segment_key::operator==(const segment_key & other) const {
    return model == other.model && configuration == other.configuration &&
            model_generation == other.model_generation && configuration_generation == other.configuration_generation &&
            format_generation == other.format_generation && origin == other.origin && approximation == other.approximation;
}

data_owner own_stash(llama_kvmem_stash * stash, void (*free_stash)(llama_kvmem_stash *),
        bool (*stash_allocations)(const llama_kvmem_stash *, allocation_list &)) {
    require(stash && free_stash, "PIC stash and deleter must be non-null");
    std::shared_ptr<llama_kvmem_stash> owned(stash, free_stash);
    require(stash_allocations != nullptr, "PIC stash allocation callback must be non-null");
    data_owner result;
    result.owner = std::move(owned);
    result.allocations = [stash_allocations](const void * ptr, allocation_list & out) {
        return stash_allocations(static_cast<const llama_kvmem_stash *>(ptr), out);
    };
    return result;
}

segment_builder::segment_builder(std::unique_ptr<segment> value) : value(std::move(value)) {}
segment_builder::segment_builder(segment_builder &&) noexcept = default;
segment_builder & segment_builder::operator=(segment_builder &&) noexcept = default;
segment_builder::~segment_builder() = default;

segment_data & segment_builder::data() {
    require(bool(value), "PIC builder has been consumed");
    return value->data;
}

struct store::impl {
    struct record {
        segment_ptr entry;
        uint64_t full_hash;
        uint64_t anchor_hash;
    };
    using records = std::list<record>;
    using index_type = std::unordered_multimap<uint64_t, records::iterator>;
    struct live_record {
        segment_ptr entry;
        allocation_list allocations;
    };

    explicit impl(uint64_t limit) : limit(limit) {}
    uint64_t limit;
    uint64_t next_id = 1;
    mutable std::mutex mutex;
    records lru;
    index_type index;
    mutable std::vector<std::weak_ptr<const segment>> tracked;

    std::vector<live_record> live() const {
        std::vector<live_record> result;
        for (auto it = tracked.begin(); it != tracked.end();) {
            if (auto entry = it->lock()) {
                auto allocations = entry_allocations(*entry);
                result.push_back({ std::move(entry), std::move(allocations) });
                ++it;
            } else {
                it = tracked.erase(it);
            }
        }
        return result;
    }

    static allocation_union account(const std::vector<live_record> & live, const allocation_list & extra,
            const std::unordered_set<const segment *> & excluded = {}) {
        allocation_union result;
        result.add(extra);
        for (const auto & item : live) {
            if (!excluded.count(item.entry.get())) result.add(item.allocations);
        }
        return result;
    }

    void erase(records::iterator victim) {
        const auto range = index.equal_range(victim->anchor_hash);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == victim) {
                index.erase(it);
                break;
            }
        }
        lru.erase(victim);
    }
};

store::store(uint64_t budget_bytes) {
    require(budget_bytes <= maximum_store_bytes, "PIC sub-budget cannot exceed 2 GiB");
    state.reset(new impl(budget_bytes));
}
store::~store() = default;

segment_builder store::begin(segment_key key, const std::vector<int32_t> & full_tokens,
        token_range source, std::vector<int32_t> recurrent_layers, bool require_hidden) const {
    check_key(key);
    check_range(source, full_tokens.size());
    require(source.end - source.begin >= minimum_segment_tokens, "PIC segments require at least 512 tokens");
    std::sort(recurrent_layers.begin(), recurrent_layers.end());
    require(!recurrent_layers.empty() && recurrent_layers.front() >= 0 &&
            std::adjacent_find(recurrent_layers.begin(), recurrent_layers.end()) == recurrent_layers.end(),
            "PIC build requires unique recurrent layer IDs");
    auto entry = std::unique_ptr<segment>(new segment);
    entry->key = std::move(key);
    entry->source = source;
    entry->tokens.assign(full_tokens.begin() + source.begin, full_tokens.begin() + source.end);
    entry->recurrent_layers = std::move(recurrent_layers);
    entry->require_hidden = require_hidden;
    return segment_builder(std::move(entry));
}

publish_result store::publish(segment_builder build) {
    require(bool(build.value), "PIC builder has been consumed");
    validate(*build.value);
    const auto full_hash = token_hash(build.value->tokens, build.value->tokens.size());
    const auto anchor_hash = token_hash(build.value->tokens, minimum_segment_tokens);
    const auto added = entry_allocations(*build.value);
    std::lock_guard<std::mutex> lock(state->mutex);
    auto live = state->live();
    std::vector<impl::records::iterator> victims;
    std::unordered_set<const segment *> excluded;
    auto total = impl::account(live, added).bytes();
    for (auto it = state->lru.rbegin(); total > state->limit && it != state->lru.rend(); ++it) {
        // One reference in LRU and one in live(). Any further owner pins it.
        if (it->entry.use_count() != 2) continue;
        victims.push_back(std::prev(it.base()));
        excluded.insert(it->entry.get());
        total = impl::account(live, added, excluded).bytes();
    }
    if (total > state->limit) return { nullptr, 0, "PIC sub-budget exhausted by candidate or pinned allocations" };
    if (state->next_id == UINT64_MAX) throw std::overflow_error("PIC entry ID overflow");
    build.value->id = state->next_id;
    segment_ptr entry(std::move(build.value));
    // Complete all potentially throwing index allocations before any eviction.
    state->tracked.reserve(state->tracked.size() + 1);
    state->lru.push_front({ entry, full_hash, anchor_hash });
    try {
        state->index.emplace(anchor_hash, state->lru.begin());
    } catch (...) {
        state->lru.pop_front();
        throw;
    }
    state->tracked.emplace_back(entry);
    ++state->next_id;
    for (auto victim : victims) state->erase(victim);
    return { std::move(entry), victims.size(), {} };
}

std::vector<segment_match> store::match(const segment_key & key, const std::vector<int32_t> & full_tokens,
        const std::vector<token_range> & allowed, bool require_hidden) {
    check_key(key);
    for (auto range : allowed) check_range(range, full_tokens.size());
    if (full_tokens.size() < minimum_segment_tokens || allowed.empty()) return {};
    const token_hashes hashes(full_tokens);
    std::lock_guard<std::mutex> lock(state->mutex);
    struct candidate {
        token_range range;
        impl::records::iterator record;
    };
    std::vector<candidate> candidates;
    for (auto range : allowed) {
        if (range.end - range.begin < minimum_segment_tokens) continue;
        for (size_t pos = range.begin; pos <= range.end - minimum_segment_tokens; ++pos) {
            const auto bucket = state->index.equal_range(hashes.get(pos, minimum_segment_tokens));
            for (auto it = bucket.first; it != bucket.second; ++it) {
                const auto record = it->second;
                const auto & entry = *record->entry;
                const auto length = entry.tokens.size();
                if (!(entry.key == key) || (require_hidden && !has_hidden(entry)) || length > range.end - pos ||
                        hashes.get(pos, length) != record->full_hash) continue;
                if (std::equal(entry.tokens.begin(), entry.tokens.end(), full_tokens.begin() + pos)) {
                    candidates.push_back({ { pos, pos + length }, record });
                }
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        const auto al = a.range.end - a.range.begin, bl = b.range.end - b.range.begin;
        if (al != bl) return al > bl;
        if (a.range.begin != b.range.begin) return a.range.begin < b.range.begin;
        return a.record->entry->id < b.record->entry->id;
    });
    std::map<size_t, size_t> occupied;
    std::vector<segment_match> result;
    for (const auto & candidate : candidates) {
        const auto next = occupied.lower_bound(candidate.range.begin);
        if (next != occupied.end() && next->first < candidate.range.end) continue;
        if (next != occupied.begin() && std::prev(next)->second > candidate.range.begin) continue;
        occupied.emplace(candidate.range.begin, candidate.range.end);
        result.push_back({ candidate.range, candidate.record->entry });
        state->lru.splice(state->lru.begin(), state->lru, candidate.record);
    }
    std::sort(result.begin(), result.end(), [](const segment_match & a, const segment_match & b) {
        return a.range.begin < b.range.begin;
    });
    return result;
}

allocation_list store::allocations() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    return impl::account(state->live(), {}).list();
}

uint64_t store::bytes() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    return impl::account(state->live(), {}).bytes();
}

size_t store::size() const {
    std::lock_guard<std::mutex> lock(state->mutex);
    return state->lru.size();
}

uint64_t store::budget() const { return state->limit; }

void store::clear() {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->index.clear();
    state->lru.clear();
}

} // namespace kvmem_pic
