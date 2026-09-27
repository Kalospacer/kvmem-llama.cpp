#pragma once

#include "llama-kvmem-pic-math.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct llama_kvmem_stash;

namespace kvmem_pic {

constexpr size_t minimum_segment_tokens = 512;
constexpr uint64_t maximum_store_bytes = uint64_t(2) * 1024 * 1024 * 1024;

using allocation_list = std::vector<std::pair<const void *, size_t>>;
using allocation_callback = std::function<bool(const void *, allocation_list &)>;

// Retained allocations must be immutable. The callback reports allocation base
// identities and capacity bytes, not logical slice sizes. It must not reenter store.
struct data_owner {
    std::shared_ptr<const void> owner;
    allocation_callback allocations;
};

// With a valid non-null deleter, transfers ownership even on exception. Pass the hooks
// llama_kvmem_stash_free and llama_kvmem_stash_allocations. Never call consuming
// stash_put on this owner; use stash_fork when installing a cached segment.
data_owner own_stash(llama_kvmem_stash * stash,
        void (*free_stash)(llama_kvmem_stash *),
        bool (*stash_allocations)(const llama_kvmem_stash *, allocation_list &));

enum class provenance { exact, approximate };

struct segment_key {
    std::string model;
    std::string configuration;
    uint64_t model_generation = 0;
    uint64_t configuration_generation = 0;
    uint32_t format_generation = 1;
    provenance origin = provenance::approximate;
    // Required for approximate data; empty for exact. Include source lineage
    // and approximation method/version. Token equality alone is not exactness.
    std::string approximation;

    bool operator==(const segment_key & other) const;
};

struct token_range {
    size_t begin = 0;
    size_t end = 0; // exclusive, in the fully tokenized request
};

struct host_tensor {
    std::array<int64_t, 4> ne = {};
    std::vector<float> data; // F32, ne[0] fastest
};

struct segment_layer {
    int32_t layer = -1;
    transition_data transition;
    host_tensor conv_end; // [history, channels, 1, 1]
};

struct segment_data {
    std::vector<segment_layer> layers;
    host_tensor target_hidden; // optional [n_embd, segment_tokens, 1, 1]
    data_owner raw;            // complete segment-local raw K/V, owned independently
    size_t raw_rows = 0;
};

struct segment {
    uint64_t id = 0;
    segment_key key;
    token_range source;
    std::vector<int32_t> tokens;
    std::vector<int32_t> recurrent_layers;
    bool require_hidden = false;
    segment_data data;
};

using segment_ptr = std::shared_ptr<const segment>;

class segment_builder {
public:
    segment_builder(segment_builder &&) noexcept;
    segment_builder & operator=(segment_builder &&) noexcept;
    ~segment_builder();
    segment_data & data();

private:
    explicit segment_builder(std::unique_ptr<segment> value);
    std::unique_ptr<segment> value;
    friend class store;
};

struct publish_result {
    segment_ptr entry;
    size_t evicted = 0;
    std::string reason; // nonempty on budget rejection; no existing entries evicted
    explicit operator bool() const { return bool(entry); }
};

struct segment_match {
    token_range range;
    segment_ptr entry;
};

// Thread-safe index/publication. Caller must freeze payloads and external raw
// owners, and snapshot external allocation owners on the inference thread.
// Budget counts unique entry/payload/raw allocations, including evicted entries
// pinned by match/publication handles. Index nodes, shared_ptr control blocks,
// callback closure allocations and allocator overhead are excluded.
class store {
public:
    explicit store(uint64_t budget_bytes = maximum_store_bytes);
    ~store();
    store(const store &) = delete;
    store & operator=(const store &) = delete;

    // An invisible, single-sequence build. recurrent_layers is the complete model
    // recurrent layer set. Configuration keys must cover layout/position rules.
    segment_builder begin(segment_key key, const std::vector<int32_t> & full_tokens,
            token_range source, std::vector<int32_t> recurrent_layers, bool require_hidden = false) const;
    // Consumes the builder. Invalid/incomplete products or accounting failures
    // throw without modifying the index. All prior mutable aliases must be retired.
    publish_result publish(segment_builder build);

    // Whole segments only; each match must fit within one allowed range. Empty
    // ranges grant nothing. Longest-first greedy selection, earliest offset then
    // oldest entry ID on ties; returns disjoint matches in request order.
    // Hash matches ALWAYS undergo full token comparison. Hits update LRU.
    std::vector<segment_match> match(const segment_key & key, const std::vector<int32_t> & full_tokens,
            const std::vector<token_range> & allowed, bool require_hidden = false);

    // Deduplicated allocation identities, suitable for a coordinator's global
    // union with live/stash allocation reports. Keep leases while using snapshots.
    allocation_list allocations() const;
    uint64_t bytes() const;
    size_t size() const; // searchable entries, excluding evicted pinned handles
    uint64_t budget() const;
    void clear(); // remove all matches; pinned allocations remain accounted

private:
    struct impl;
    std::unique_ptr<impl> state;
};

} // namespace kvmem_pic
