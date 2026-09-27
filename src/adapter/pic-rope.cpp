#include "pic-rope.h"
#include "llama-kvmem-quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace kvmem_pic {
namespace {

bool finite(const std::vector<float> & row) {
    return std::all_of(row.begin(), row.end(), [](float v) { return std::isfinite(v); });
}

// With unit scale/amplitude and identical axes, MROPE/IMROPE reduce to NEOX.
// Float angle progression matches ggml-cpu's standard RoPE cache construction.
void rotate_row(float * row, const packed_k_rope_config & cfg, int32_t position, bool inverse) {
    const int pairs = cfg.n_rot / 2;
    const float step = std::pow(cfg.freq_base, -2.0f / float(cfg.n_rot));
    float theta = float(position);
    for (int pair = 0; pair < pairs; ++pair) {
        const float c = std::cos(theta);
        const float s = std::sin(theta) * (inverse ? -1.0f : 1.0f);
        const int x = cfg.mode == GGML_ROPE_TYPE_NORMAL ? 2 * pair : pair;
        const int y = cfg.mode == GGML_ROPE_TYPE_NORMAL ? x + 1 : x + pairs;
        for (int head = 0; head < cfg.heads; ++head) {
            float * h = row + static_cast<size_t>(head) * cfg.head_dim;
            const float a = h[x], b = h[y];
            h[x] = a * c - b * s;
            h[y] = a * s + b * c;
        }
        theta *= step;
    }
}

} // namespace

rope_status relocate_packed_k(const packed_k_rope_config & cfg,
                              const std::vector<rope_position> & source_positions,
                              const std::vector<rope_position> & destination_positions,
                              const void * packed, size_t packed_bytes,
                              std::vector<uint8_t> & output, std::string & reason) {
    const auto reject = [&](rope_status status, const char * message) {
        reason = message;
        return status;
    };
    if (cfg.type != GGML_TYPE_F16 && cfg.type != GGML_TYPE_Q8_0)
        return reject(rope_status::unsupported, "packed K relocation supports only F16 and Q8_0");
    if (cfg.mode != GGML_ROPE_TYPE_NORMAL && cfg.mode != GGML_ROPE_TYPE_NEOX &&
        cfg.mode != GGML_ROPE_TYPE_MROPE && cfg.mode != GGML_ROPE_TYPE_IMROPE)
        return reject(rope_status::unsupported, "unsupported RoPE mode (vision and mode flags are not accepted)");
    if (cfg.head_dim <= 0 || cfg.heads <= 0 || cfg.n_rot <= 0 || cfg.n_rot > cfg.head_dim || cfg.n_rot % 2 != 0)
        return reject(rope_status::invalid_input, "invalid head dimensions or n_rot");
    for (float value : {cfg.freq_base, cfg.freq_scale, cfg.ext_factor, cfg.attn_factor, cfg.beta_fast, cfg.beta_slow}) {
        if (!std::isfinite(value)) return reject(rope_status::invalid_input, "non-finite RoPE parameter");
    }
    if (cfg.freq_base <= 1.0f || cfg.n_ctx_orig <= 0 || cfg.beta_fast <= 0.0f || cfg.beta_slow <= 0.0f)
        return reject(rope_status::invalid_input, "RoPE requires base>1, n_ctx_orig>0 and positive beta parameters");
    if (cfg.freq_scale != 1.0f || cfg.attn_factor != 1.0f || cfg.ext_factor != 0.0f)
        return reject(rope_status::unsupported, "scaled/YaRN RoPE is unsupported: require scale=attn=1 and ext=0");
    if (cfg.n_rot_offset != 0 || !cfg.freq_factors.empty())
        return reject(rope_status::unsupported, "RoPE offsets and frequency-factor tensors are unsupported");
    const bool multi = cfg.mode == GGML_ROPE_TYPE_MROPE || cfg.mode == GGML_ROPE_TYPE_IMROPE;
    int64_t sections = 0;
    for (int section : cfg.sections) {
        if (section < 0) return reject(rope_status::invalid_input, "negative MROPE section");
        sections += section;
    }
    if (multi) {
        if (sections != cfg.n_rot / 2 || (cfg.sections[0] == 0 && cfg.sections[1] == 0 && cfg.sections[2] == 0))
            return reject(rope_status::invalid_input, "MROPE sections must sum to n_rot/2 with a nonempty text/spatial section");
    } else if (sections != 0) {
        return reject(rope_status::unsupported, "sections supplied for non-multi RoPE");
    }
    if (cfg.hadamard_nrot != 0 &&
        (cfg.head_dim < 64 || cfg.head_dim % 64 != 0 ||
         cfg.hadamard_nrot != (cfg.head_dim & -cfg.head_dim)))
        return reject(rope_status::unsupported, "Hadamard width does not match the K cache codec");
    if (source_positions.size() != destination_positions.size())
        return reject(rope_status::invalid_input, "source/destination position counts differ");
    for (const auto * positions : {&source_positions, &destination_positions}) {
        for (const auto & p : *positions) {
            for (int32_t axis : p) {
                if (axis < 0) return reject(rope_status::invalid_input, "negative RoPE position");
                if (axis != p[0]) return reject(rope_status::unsupported, "only text positions with all four axes equal are supported");
            }
        }
    }
    const int64_t row_elements = int64_t(cfg.head_dim) * cfg.heads;
    const auto * traits = ggml_get_type_traits(cfg.type);
    if (!traits || !traits->to_float || !traits->from_float_ref)
        return reject(rope_status::unsupported, "missing GGML dequant/quant traits");
    if (cfg.head_dim % traits->blck_size != 0)
        return reject(rope_status::invalid_input, "head dimension is not aligned to the quantization block");
    // Bound both F32 scratch and GGML's signed row-size arithmetic before allocation.
    const uint64_t max_elements = std::min<uint64_t>(SIZE_MAX, INT64_MAX) / sizeof(float);
    if (row_elements > INT32_MAX || static_cast<uint64_t>(row_elements) > max_elements)
        return reject(rope_status::invalid_input, "packed K row size overflow");
    const size_t row_bytes = ggml_row_size(cfg.type, row_elements);
    if (source_positions.size() > SIZE_MAX / row_bytes || packed_bytes != source_positions.size() * row_bytes)
        return reject(rope_status::invalid_input, "packed K byte count does not match positions and shape");
    if (packed_bytes > 0 && !packed)
        return reject(rope_status::invalid_input, "null packed K input");

    std::vector<uint8_t> result(packed_bytes);
    if (source_positions.empty()) {
        output.swap(result);
        reason.clear();
        return rope_status::ok;
    }
    std::vector<float> row(static_cast<size_t>(row_elements));
    // GGML codecs cast their pointers to block structs; accept unaligned caller bytes.
    std::vector<uint64_t> aligned((row_bytes + sizeof(uint64_t) - 1) / sizeof(uint64_t));
    const auto * bytes = static_cast<const uint8_t *>(packed);
    for (size_t token = 0; token < source_positions.size(); ++token) {
        const size_t offset = token * row_bytes;
        std::memcpy(aligned.data(), bytes + offset, row_bytes);
        traits->to_float(aligned.data(), row.data(), row_elements);
        if (!finite(row)) return reject(rope_status::invalid_input, "packed K dequantized to non-finite values");
        if (source_positions[token] == destination_positions[token]) {
            std::memcpy(result.data() + offset, bytes + offset, row_bytes);
            continue;
        }
        if (cfg.hadamard_nrot) kvmem_hadamard_rows(row.data(), 1, cfg.heads, cfg.head_dim, cfg.hadamard_nrot);
        rotate_row(row.data(), cfg, source_positions[token][0], true);
        rotate_row(row.data(), cfg, destination_positions[token][0], false);
        if (cfg.hadamard_nrot) kvmem_hadamard_rows(row.data(), 1, cfg.heads, cfg.head_dim, cfg.hadamard_nrot);
        if (!finite(row)) return reject(rope_status::invalid_input, "packed K transform overflowed");
        traits->from_float_ref(row.data(), aligned.data(), row_elements);
        // Finite F32 values may overflow F16 (or a Q8_0 block's F16 scale).
        traits->to_float(aligned.data(), row.data(), row_elements);
        if (!finite(row)) return reject(rope_status::invalid_input, "packed K requantization overflowed");
        std::memcpy(result.data() + offset, aligned.data(), row_bytes);
    }
    output.swap(result);
    reason.clear();
    return rope_status::ok;
}

} // namespace kvmem_pic
