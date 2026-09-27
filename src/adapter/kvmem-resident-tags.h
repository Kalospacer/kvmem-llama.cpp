#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

// Non-owning GPU contents cache. Only the live runtime allocates/frees slots.
// Strong host leases prove ancestry and prevent allocator-address reuse (ABA).
// Serialized by the inference thread; publishing requires completed GPU/host IO.
class kvmem_resident_tags {
public:
    using payload = std::shared_ptr<const std::vector<uint8_t>>;
    using payloads = std::vector<payload>;

    void resize(size_t slots) {
        if (slots != tags_.size()) clear();
        tags_.resize(slots);
    }

    void invalidate(size_t slot) noexcept {
        if (slot >= tags_.size()) return;
        auto & t = tags_[slot];
        t.ready = false;
        t.refs.clear();
        if (t.write_epoch == std::numeric_limits<uint64_t>::max()) {
            clear();
            for (auto & tag : tags_) tag.write_epoch = 0;
        }
        ++t.write_epoch;
    }

    void clear() noexcept {
        for (auto & t : tags_) {
            t.ready = false;
            t.refs.clear();
        }
    }

    bool matches(size_t slot, uint32_t orig, uint32_t rows, const payloads & refs) const noexcept {
        if (slot >= tags_.size() || rows == 0 || refs.empty()) return false;
        const auto & t = tags_[slot];
        return t.ready && t.published_epoch == t.write_epoch && t.orig == orig &&
               t.rows == rows && t.refs == refs;
    }

    // Caller keeps its own write barrier until publication. No pending tag is
    // visible if vector allocation or a transfer failed before this call.
    void publish(size_t slot, uint32_t orig, uint32_t rows, payloads refs) noexcept {
        if (slot >= tags_.size()) return;
        auto & t = tags_[slot];
        t.refs.swap(refs);
        t.orig = orig;
        t.rows = rows;
        t.published_epoch = t.write_epoch;
        t.ready = rows != 0 && !t.refs.empty();
    }

    void append_allocations(std::vector<std::pair<const void *, size_t>> & out) const {
        if (tags_.capacity()) out.emplace_back(tags_.data(), tags_.capacity() * sizeof(tag));
        for (const auto & t : tags_) {
            if (t.refs.capacity()) out.emplace_back(t.refs.data(), t.refs.capacity() * sizeof(payload));
            for (const auto & p : t.refs) {
                if (!p) continue;
                out.emplace_back(p.get(), sizeof(*p));
                if (p->capacity()) out.emplace_back(p->data(), p->capacity());
            }
        }
    }

private:
    struct tag {
        uint64_t write_epoch = 0;
        uint64_t published_epoch = 0;
        uint32_t orig = 0;
        uint32_t rows = 0;
        bool ready = false;
        payloads refs;
    };
    std::vector<tag> tags_;
};
