#include "llama-kvmem-pic-kv.h"

#include "llama-memory-kvmem.h"
#include "llama-memory-kvmem-mtp.h"
#include "llama-kvmem-quant.h"
#include "llama-kvmem-diag.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kvmem_pic {
namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}

size_t product(size_t a, size_t b) {
    require(!b || a <= SIZE_MAX / b, "PIC KV size overflow");
    return a * b;
}

void add_size(size_t & total, size_t bytes) {
    require(bytes <= SIZE_MAX - total, "PIC KV size overflow");
    total += bytes;
}

bool same_identity(const kv_segment_identity & a, const kv_segment_identity & b) {
    return a.model == b.model && a.mtp_model == b.mtp_model && a.epoch == b.epoch;
}

bool same_rope(const packed_k_rope_config & a, const packed_k_rope_config & b) {
    return a.type == b.type && a.head_dim == b.head_dim && a.heads == b.heads &&
        a.n_rot == b.n_rot && a.mode == b.mode && a.sections == b.sections &&
        a.n_ctx_orig == b.n_ctx_orig && a.freq_base == b.freq_base &&
        a.freq_scale == b.freq_scale && a.ext_factor == b.ext_factor &&
        a.attn_factor == b.attn_factor && a.beta_fast == b.beta_fast &&
        a.beta_slow == b.beta_slow && a.n_rot_offset == b.n_rot_offset &&
        a.freq_factors == b.freq_factors && a.hadamard_nrot == b.hadamard_nrot;
}

bool same_codec(const kv_layer_codec & a, const kv_layer_codec & b) {
    return a.mtp == b.mtp && a.raw_layer == b.raw_layer && a.graph_layer == b.graph_layer &&
        same_rope(a.k, b.k) && a.v_type == b.v_type && a.v_elements == b.v_elements &&
        a.v_hadamard_nrot == b.v_hadamard_nrot;
}

size_t row_bytes(ggml_type type, uint64_t elements) {
    require(type >= 0 && type < GGML_TYPE_COUNT, "invalid PIC KV type");
    const auto * traits = ggml_get_type_traits(type);
    require(traits && traits->blck_size > 0 && traits->type_size > 0 && elements > 0 &&
            elements <= UINT32_MAX && elements % traits->blck_size == 0, "invalid PIC KV row shape");
    return product(static_cast<size_t>(elements / traits->blck_size), traits->type_size);
}

uint32_t k_elements(const kv_layer_codec & codec) {
    require(codec.k.head_dim > 0 && codec.k.heads > 0, "invalid PIC K head shape");
    const uint64_t n = uint64_t(codec.k.head_dim) * codec.k.heads;
    require(n <= UINT32_MAX, "PIC K width overflow");
    return static_cast<uint32_t>(n);
}

void text_position(const rope_position & p) {
    require(p[0] >= 0 && std::all_of(p.begin(), p.end(), [&](int32_t x) { return x == p[0]; }),
            "PIC KV splice currently requires nonnegative scalar text positions");
}

const kvmem::RawKvStore & live_raw(llama_memory_kvmem & dst, bool mtp) {
    if (mtp) {
        require(dst.mtp_follower() != nullptr, "missing destination MTP follower");
        return dst.mtp_follower()->raw();
    }
    return dst.raw();
}

void check_raw(const kvmem::RawKvStore & raw, const kv_layer_codec & codec) {
    const auto & cfg = raw.config();
    require(!raw.nvme_enabled() && !cfg.nvme_bytes, "PIC KV splice rejects NVMe stores");
    require(cfg.block_tokens > 0 && codec.raw_layer < cfg.n_layer, "invalid PIC raw layer/block size");
    require(cfg.n_embd_k == k_elements(codec) && cfg.n_embd_v == codec.v_elements &&
            cfg.k_gpu_row_bytes == row_bytes(codec.k.type, cfg.n_embd_k) &&
            cfg.v_gpu_row_bytes == row_bytes(codec.v_type, cfg.n_embd_v), "PIC raw codec mismatch");
}

void check_codecs(llama_memory_kvmem & dst, const std::vector<kv_layer_codec> & codecs) {
    require(dst.get_kv() && !dst.v_trans(), "PIC KV requires a non-transposed target cache");
    auto target_ids = dst.get_kv()->get_layer_ids();
    std::sort(target_ids.begin(), target_ids.end());
    std::vector<uint32_t> mtp_ids;
    if (dst.mtp_follower()) {
        require(dst.mtp_follower()->target() == &dst, "MTP follower belongs to another target");
        mtp_ids = dst.mtp_follower()->get_kv()->get_layer_ids();
        require(mtp_ids.size() == 1, "PIC KV currently supports one MTP raw layer");
    }
    require(!target_ids.empty() && codecs.size() == target_ids.size() + mtp_ids.size(),
            "PIC codecs must cover every target FA and MTP layer exactly once");
    for (size_t i = 0; i < codecs.size(); ++i) {
        const auto & codec = codecs[i];
        const bool mtp = i >= target_ids.size();
        const uint32_t graph = mtp ? mtp_ids[0] : target_ids[i];
        require(codec.mtp == mtp && codec.graph_layer == graph && codec.raw_layer == (mtp ? 0u : graph),
                "PIC codec order must be sorted target FA layers then MTP");
        const auto & raw = live_raw(dst, mtp);
        check_raw(raw, codec);
        require(raw.config().block_tokens == dst.block_tokens(), "destination raw block size mismatch");
        if (!mtp) {
            const auto & rope = dst.rope();
            require(codec.k.n_rot == rope.n_rot && codec.k.head_dim == rope.n_embd_head &&
                    codec.k.heads == rope.n_head_kv && codec.k.freq_base == rope.freq_base &&
                    codec.k.freq_scale == rope.freq_scale, "PIC target RoPE metadata differs from live adapter");
        }
        require(codec.v_hadamard_nrot == 0 ||
                (codec.v_hadamard_nrot == 64 && ggml_is_quantized(codec.v_type) && codec.v_elements % 64 == 0),
                "invalid PIC V Hadamard codec metadata");
        auto * cache = mtp ? dst.mtp_follower()->get_kv() : dst.get_kv();
        auto * kt = cache->get_k_storage(graph);
        auto * vt = cache->get_v_storage(graph);
        require(cache->get_n_stream() == 1 && kt && vt &&
                cache->type_k() == codec.k.type && cache->type_v() == codec.v_type &&
                kt->ne[0] == k_elements(codec) && vt->ne[0] == codec.v_elements &&
                kt->nb[1] == row_bytes(codec.k.type, k_elements(codec)) &&
                vt->nb[1] == row_bytes(codec.v_type, codec.v_elements),
                "PIC requires matching token-major target/MTP tensor layouts");
        // Delegate ALL supported RoPE checks, including no-op relocations.
        std::vector<uint8_t> unused;
        std::string reason;
        const auto status = relocate_packed_k(codec.k, {}, {}, nullptr, 0, unused, reason);
        require(status == rope_status::ok,
                reason.empty() ? "unsupported PIC RoPE codec" : reason.c_str());
    }
}

// Raw exposes prefix reads; slice each touched source block without assuming
// source and destination offsets or block lengths coincide.
std::vector<uint8_t> read_rows(const kvmem::RawKvStore & raw, uint32_t layer,
                               uint32_t begin, uint32_t count, bool is_k) {
    const auto & cfg = raw.config();
    require(cfg.block_tokens > 0, "zero source block size");
    const size_t row = static_cast<size_t>(is_k ? cfg.k_gpu_row_bytes : cfg.v_gpu_row_bytes);
    require(row > 0, "missing packed PIC rows");
    std::vector<uint8_t> result(product(count, row));
    for (uint32_t done = 0; done < count;) {
        const uint32_t pos = begin + done, bid = pos / cfg.block_tokens, off = pos % cfg.block_tokens;
        const uint32_t take = std::min(count - done, cfg.block_tokens - off);
        std::vector<uint8_t> prefix(product(off + take, row));
        require(is_k ? raw.copy_k_gpu(bid, layer, prefix.data(), off + take)
                     : raw.copy_v_gpu(bid, layer, prefix.data(), off + take), "incomplete PIC packed K/V span");
        std::memcpy(result.data() + product(done, row), prefix.data() + product(off, row), product(take, row));
        done += take;
    }
    return result;
}

std::vector<float> content_rows(const kv_layer_codec & codec, const std::vector<uint8_t> & k,
                                const std::vector<rope_position> & positions) {
    std::vector<uint8_t> at_zero;
    std::vector<rope_position> zeros(positions.size(), rope_position{});
    std::string reason;
    const auto status = relocate_packed_k(codec.k, positions, zeros, k.data(), k.size(), at_zero, reason);
    require(status == rope_status::ok, reason.c_str());
    const size_t width = k_elements(codec), row = row_bytes(codec.k.type, width);
    std::vector<float> result(product(positions.size(), width));
    std::vector<uint64_t> aligned((row + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    for (size_t t = 0; t < positions.size(); ++t) {
        std::memcpy(aligned.data(), at_zero.data() + t * row, row);
        require(kvmem_cache_unpack_rows(codec.k.type, aligned.data(), result.data() + t * width, 1, width),
                "PIC mean dequantization failed");
    }
    if (codec.k.hadamard_nrot) {
        hadamard_rows_fast(result.data(), positions.size(), codec.k.heads, codec.k.head_dim, codec.k.hadamard_nrot);
    }
    require(std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); }),
            "non-finite reconstructed PIC mean rows");
    return result;
}

void add_rows(std::vector<float> & sum, const std::vector<float> & rows, uint32_t count) {
    require(rows.size() == product(sum.size(), count), "PIC mean row shape mismatch");
    for (uint32_t t = 0; t < count; ++t) {
        for (size_t d = 0; d < sum.size(); ++d) sum[d] += rows[t * sum.size() + d];
    }
    require(std::all_of(sum.begin(), sum.end(), [](float x) { return std::isfinite(x); }), "PIC mean sum overflow");
}

void check_live(llama_memory_kvmem & dst, uint32_t begin, uint32_t end) {
    require(dst.block_tokens() > 0 && begin == dst.store_n_tokens() && end > begin && end <= INT32_MAX,
            "PIC splice must append a nonempty span at the current logical end");
    require(!dst.replay(), "cannot splice during query replay");
    std::string reason;
    const bool fits = dst.can_append(end, 0, false, reason);
    require(fits, reason.c_str());
}

} // namespace

size_t kv_splice_plan::bytes() const {
    size_t result = sizeof(*this);
    add_size(result, product(tokens.capacity(), sizeof(llama_token)));
    add_size(result, product(positions.capacity(), sizeof(rope_position)));
    add_size(result, product(blocks.capacity(), sizeof(kv_splice_block)));
    for (const auto & block : blocks) {
        add_size(result, product(block.layers.capacity(), sizeof(kv_splice_layer)));
        for (const auto & layer : block.layers) {
            add_size(result, layer.packed_k.capacity());
            add_size(result, layer.packed_v.capacity());
            add_size(result, product(layer.k_sum.capacity(), sizeof(float)));
            add_size(result, product(layer.prefix_k_sum.capacity(), sizeof(float)));
            add_size(result, product(layer.codec.k.freq_factors.capacity(), sizeof(float)));
        }
    }
    return result;
}

bool pic_kv_prepare(const llama_kvmem_stash & source, llama_memory_kvmem & dst,
                    const kv_splice_request & request, kv_splice_plan & out, std::string & error) {
    const int64_t pt0 = ggml_time_us();
    int64_t us_read = 0, us_reloc = 0, us_mean = 0;
    try {
        require(request.source_identity.model && request.source_identity.epoch &&
                request.source_identity.independently_evaluated &&
                same_identity(request.source_identity, request.destination_identity),
                "PIC source must be an independent segment with matching model/config epoch");
        require(bool(dst.mtp_follower()) == bool(request.source_identity.mtp_model) &&
                bool(source.mtp_raw) == bool(dst.mtp_follower()), "PIC MTP source/destination coverage differs");
        require(!request.tokens.empty() && request.tokens.size() == request.positions.size() &&
                request.tokens.size() <= INT32_MAX && request.max_plan_bytes > 0, "invalid PIC body or plan budget");
        const uint64_t src_end = uint64_t(request.source_begin) + request.tokens.size();
        const uint64_t dst_end = uint64_t(request.destination_begin) + request.tokens.size();
        require(source.raw && source.runtime && src_end <= source.rows && src_end <= source.row_positions.size() &&
                src_end <= source.runtime->store().total_tokens() && dst_end <= INT32_MAX, "PIC span out of bounds");
        require(std::all_of(source.runtime->store().blocks().begin(), source.runtime->store().blocks().end(),
                [](const kvmem::KvMemBlock & b) { return b.gpu_slot < 0 && !b.in_flight; }),
                "PIC source stash is not detached/quiescent");
        check_live(dst, request.destination_begin, static_cast<uint32_t>(dst_end));
        check_codecs(dst, request.destination_codecs);
        require(request.source_codecs.size() == request.destination_codecs.size(), "PIC source codec coverage differs");
        for (size_t i = 0; i < request.source_codecs.size(); ++i) {
            const auto & codec = request.source_codecs[i];
            require(same_codec(codec, request.destination_codecs[i]), "PIC source/destination codec differs");
            check_raw(codec.mtp ? *source.mtp_raw : *source.raw, codec);
        }
        for (size_t i = 0; i < request.tokens.size(); ++i) {
            const auto & src = source.row_positions[request.source_begin + i];
            // row.spatial mirrors is_pos_2d() (n_pos >= 3) and is set for every
            // row of an M-RoPE model, including scalar text. The real constraint
            // is nonnegative tokens and four equal text position axes.
            if (src.token < 0 || src.token != request.tokens[i]) {
                char detail[192];
                snprintf(detail, sizeof(detail),
                        "PIC src token mismatch i=%zu spatial=%d src=%d want=%d pos=%d",
                        i, (int) src.spatial, (int) src.token, (int) request.tokens[i], (int) src.pos[0]);
                require(false, detail);
            }
            text_position(src.pos);
            text_position(request.positions[i]);
        }

        const int64_t pt_checks = ggml_time_us();
        kv_splice_plan plan;
        plan.destination = &dst;
        plan.identity = request.destination_identity;
        plan.attention_epoch = dst.attention_view(false).epoch;
        plan.source_begin = request.source_begin;
        plan.destination_begin = request.destination_begin;
        plan.destination_end = static_cast<uint32_t>(dst_end);
        plan.block_tokens = dst.block_tokens();
        const uint32_t first = plan.destination_begin / plan.block_tokens;
        const uint32_t last = (plan.destination_end - 1) / plan.block_tokens;
        // Conservative retained-size preflight before copying any body payload.
        size_t estimate = sizeof(plan);
        add_size(estimate, product(request.tokens.size(), sizeof(llama_token) + sizeof(rope_position)));
        size_t per_block = sizeof(kv_splice_block);
        for (const auto & codec : request.destination_codecs) {
            add_size(per_block, sizeof(kv_splice_layer));
            add_size(per_block, product(plan.block_tokens, row_bytes(codec.k.type, k_elements(codec))));
            add_size(per_block, product(plan.block_tokens, row_bytes(codec.v_type, codec.v_elements)));
            add_size(per_block, product(k_elements(codec), sizeof(float)));
            if (!codec.mtp) add_size(per_block, product(k_elements(codec), sizeof(float)));
        }
        add_size(estimate, product(uint64_t(last) - first + 1, per_block));
        require(estimate <= request.max_plan_bytes, "PIC retained plan exceeds budget");
        plan.tokens = request.tokens;
        plan.positions = request.positions;
        plan.blocks.reserve(last - first + 1);
        const auto tail_state = dst.raw().mean_checkpoint(plan.destination_begin);
        for (uint32_t bid = first; bid <= last; ++bid) {
            kv_splice_block block;
            block.block_id = bid;
            const uint32_t block_start = bid * plan.block_tokens;
            block.prefix_rows = bid == first ? plan.destination_begin % plan.block_tokens : 0;
            block.valid_rows = std::min(plan.block_tokens, plan.destination_end - block_start);
            if (bid < dst.store().block_count()) block.previous_slot = dst.store().blocks()[bid].gpu_slot;
            if (block.previous_slot < 0) ++plan.slots_needed;
            const uint32_t take = block.valid_rows - block.prefix_rows;
            const uint32_t body_off = block_start + block.prefix_rows - plan.destination_begin;
            std::vector<rope_position> source_pos(take), destination_pos(take);
            for (uint32_t t = 0; t < take; ++t) {
                source_pos[t] = source.row_positions[request.source_begin + body_off + t].pos;
                destination_pos[t] = request.positions[body_off + t];
            }
            block.layers.reserve(request.destination_codecs.size());
            for (const auto & codec : request.destination_codecs) {
                kv_splice_layer layer;
                layer.codec = codec;
                const auto & src_raw = codec.mtp ? *source.mtp_raw : *source.raw;
                const auto & dst_raw = live_raw(dst, codec.mtp);
                int64_t lt = ggml_time_us();
                auto source_k = read_rows(src_raw, codec.raw_layer, request.source_begin + body_off, take, true);
                auto source_v = read_rows(src_raw, codec.raw_layer, request.source_begin + body_off, take, false);
                us_read += ggml_time_us() - lt;
                lt = ggml_time_us();
                std::vector<uint8_t> moved_k;
                std::string reason;
                const auto status = relocate_packed_k(codec.k, source_pos, destination_pos,
                        source_k.data(), source_k.size(), moved_k, reason);
                require(status == rope_status::ok, reason.c_str());
                us_reloc += ggml_time_us() - lt;
                lt = ggml_time_us();
                layer.k_sum.assign(k_elements(codec), 0.0f);
                if (block.prefix_rows) {
                    layer.packed_k = read_rows(dst_raw, codec.raw_layer, block_start, block.prefix_rows, true);
                    layer.packed_v = read_rows(dst_raw, codec.raw_layer, block_start, block.prefix_rows, false);
                    if (!codec.mtp) {
                        const size_t stride = 1 + layer.k_sum.size(), offset = product(codec.raw_layer, stride);
                        require(offset + stride <= tail_state.size() && tail_state[offset] == float(block.prefix_rows),
                                "target tail mean checkpoint missing or incomplete; flush before PIC prepare");
                        std::copy_n(tail_state.begin() + offset + 1, layer.k_sum.size(), layer.k_sum.begin());
                        layer.prefix_k_sum = layer.k_sum;
                    } else {
                        // MTP never captures pre-RoPE K. Recover its text prefix
                        // from packed rows too, without touching target statistics.
                        std::vector<rope_position> prefix_pos(block.prefix_rows);
                        const auto & cells = dst.get_kv()->get_cells(0);
                        for (uint32_t t = 0; t < block.prefix_rows; ++t) {
                            const auto cell = uint64_t(block.previous_slot) * plan.block_tokens + t;
                            require(block.previous_slot >= 0 && cell < cells.size() && !cells.is_empty(cell) &&
                                    cells.ext_get(cell).tok >= 0 && cells.ext_get(cell).logical_pos == llama_pos(block_start + t),
                                    "MTP mean reconstruction requires a resident text destination tail");
                            const auto p = dst.model_pos(block_start + t);
                            require(p >= 0 && cells.pos_get(cell) == p &&
                                    (cells.ext_get(cell).x == 0 || cells.ext_get(cell).x == p) &&
                                    (cells.ext_get(cell).y == 0 || cells.ext_get(cell).y == p),
                                    "MTP destination tail has unsupported spatial positions");
                            prefix_pos[t] = {p, p, p, p};
                        }
                        add_rows(layer.k_sum, content_rows(codec, layer.packed_k, prefix_pos), block.prefix_rows);
                    }
                }
                add_rows(layer.k_sum, content_rows(codec, source_k, source_pos), take);
                us_mean += ggml_time_us() - lt;
                layer.packed_k.insert(layer.packed_k.end(), moved_k.begin(), moved_k.end());
                layer.packed_v.insert(layer.packed_v.end(), source_v.begin(), source_v.end());
                block.layers.push_back(std::move(layer));
            }
            plan.blocks.push_back(std::move(block));
        }
        require(plan.bytes() <= request.max_plan_bytes, "PIC retained capacity exceeds plan budget");
        const int64_t pt_blocks = ggml_time_us();
        const bool valid = pic_kv_validate(dst, plan, error);
        kvmem_diag("KVMEM_PIC_PREPARE checks=%.1f blocks=%.1f read=%.1f reloc=%.1f mean=%.1f validate=%.1f n_blocks=%zu layers=%zu tokens=%zu\n",
                (pt_checks - pt0) / 1e3, (pt_blocks - pt_checks) / 1e3, us_read / 1e3, us_reloc / 1e3, us_mean / 1e3,
                (ggml_time_us() - pt_blocks) / 1e3, plan.blocks.size(), request.destination_codecs.size(), request.tokens.size());
        require(valid, error.c_str());
        out = std::move(plan);
        error.clear();
        return true;
    } catch (const std::invalid_argument & e) {
        error = e.what();
        return false;
    }
}

bool pic_kv_validate(llama_memory_kvmem & dst, const kv_splice_plan & plan, std::string & error) {
    try {
        require(plan.destination == &dst && plan.approximate && plan.identity.model && plan.identity.epoch &&
                plan.block_tokens == dst.block_tokens() && !plan.blocks.empty(), "invalid PIC plan destination/provenance");
        check_live(dst, plan.destination_begin, plan.destination_end);
        require(dst.attention_view(false).epoch == plan.attention_epoch, "stale PIC attention epoch");
        require(plan.tokens.size() == plan.destination_end - plan.destination_begin &&
                plan.positions.size() == plan.tokens.size(), "invalid PIC plan row metadata");
        for (size_t i = 0; i < plan.tokens.size(); ++i) {
            require(plan.tokens[i] >= 0, "PIC plan contains non-token input");
            text_position(plan.positions[i]);
        }
        std::vector<kv_layer_codec> codecs;
        for (const auto & layer : plan.blocks.front().layers) codecs.push_back(layer.codec);
        check_codecs(dst, codecs);
        require(bool(dst.mtp_follower()) == bool(plan.identity.mtp_model), "PIC plan MTP identity mismatch");
        const auto tail_state = dst.raw().mean_checkpoint(plan.destination_begin);
        const uint32_t first = plan.destination_begin / plan.block_tokens;
        const uint32_t last = (plan.destination_end - 1) / plan.block_tokens;
        require(plan.blocks.size() == uint64_t(last) - first + 1, "PIC plan block coverage mismatch");
        uint32_t slots = 0;
        for (size_t i = 0; i < plan.blocks.size(); ++i) {
            const auto & block = plan.blocks[i];
            const uint32_t bid = first + static_cast<uint32_t>(i), start = bid * plan.block_tokens;
            const uint32_t prefix = i == 0 ? plan.destination_begin % plan.block_tokens : 0;
            require(block.block_id == bid && block.prefix_rows == prefix &&
                    block.valid_rows == std::min(plan.block_tokens, plan.destination_end - start) &&
                    block.layers.size() == codecs.size(), "PIC plan block shape mismatch");
            const int32_t slot = bid < dst.store().block_count() ? dst.store().blocks()[bid].gpu_slot : -1;
            require(slot == block.previous_slot, "PIC destination slot changed after preparation");
            if (slot < 0) ++slots;
            for (size_t il = 0; il < block.layers.size(); ++il) {
                const auto & layer = block.layers[il];
                require(same_codec(layer.codec, codecs[il]), "PIC plan codec changed between blocks");
                const auto & codec = layer.codec;
                require(layer.packed_k.size() == product(block.valid_rows, row_bytes(codec.k.type, k_elements(codec))) &&
                        layer.packed_v.size() == product(block.valid_rows, row_bytes(codec.v_type, codec.v_elements)) &&
                        layer.k_sum.size() == k_elements(codec) &&
                        std::all_of(layer.k_sum.begin(), layer.k_sum.end(), [](float x) { return std::isfinite(x); }),
                        "PIC plan packed rows/mean shape mismatch");
                require(layer.prefix_k_sum.size() == (prefix && !codec.mtp ? k_elements(codec) : 0),
                        "PIC plan prefix mean shape mismatch");
                if (prefix) {
                    const auto & raw = live_raw(dst, codec.mtp);
                    const auto k = read_rows(raw, codec.raw_layer, start, prefix, true);
                    const auto v = read_rows(raw, codec.raw_layer, start, prefix, false);
                    require(std::equal(k.begin(), k.end(), layer.packed_k.begin()) &&
                            std::equal(v.begin(), v.end(), layer.packed_v.begin()), "PIC destination tail changed after preparation");
                    if (!codec.mtp) {
                        const size_t stride = 1 + layer.k_sum.size(), offset = product(codec.raw_layer, stride);
                        require(offset + stride <= tail_state.size() && tail_state[offset] == float(prefix) &&
                                std::equal(layer.prefix_k_sum.begin(), layer.prefix_k_sum.end(), tail_state.begin() + offset + 1),
                                "PIC destination tail mean changed after preparation");
                    }
                }
            }
        }
        require(slots == plan.slots_needed && slots <= dst.free_slot_count(), "PIC slot reservation no longer fits");
        error.clear();
        return true;
    } catch (const std::invalid_argument & e) {
        error = e.what();
        return false;
    }
}

} // namespace kvmem_pic
