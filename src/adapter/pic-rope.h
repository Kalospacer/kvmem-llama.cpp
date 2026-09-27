#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kvmem_pic {

struct packed_k_rope_config {
    ggml_type type = GGML_TYPE_F16;
    int head_dim = 0;
    int heads = 0;
    int n_rot = 0;
    int mode = GGML_ROPE_TYPE_NEOX;
    std::array<int, 4> sections = {};
    int n_ctx_orig = 0;
    float freq_base = 1000000.0f;
    float freq_scale = 1.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
    int n_rot_offset = 0;
    std::vector<float> freq_factors;
    // Explicit source/destination codec metadata, NOT inferred from the environment.
    // 0 means none; otherwise the K width from kvmem_hadamard_nrot_k(head_dim).
    int hadamard_nrot = 0;
};

using rope_position = std::array<int32_t, 4>;
enum class rope_status { ok, unsupported, invalid_input };

// CPU reference for packed token-major K [tokens,heads,head_dim]. Same format,
// Hadamard and RoPE configuration at both positions; V is not transformed.
// Supports F16/Q8_0, NORMAL/NEOX/MROPE/IMROPE, n_rot_offset=0, no freq_factors,
// freq_scale=attn_factor=1, ext_factor=0, nonnegative text positions with all
// four axes equal. Requires base>1, n_ctx_orig>0 and positive finite beta values.
// MROPE sections must sum to n_rot/2. No config is clamped.
// source/destination positions have one four-axis record per token, not planes.
// packed_bytes must equal tokens*ggml_row_size(type, heads*head_dim).
// Dequant -> inverse H -> inverse source RoPE -> destination RoPE -> H -> quant.
// Same-position rows preserve packed bytes. Moved rows incur requantization loss.
// On rejection, output stays unchanged and reason identifies the exact cause.
// Allocation exceptions propagate with output unchanged. Aliasing output is OK.
// Normalized Walsh-Hadamard transform over every nrot-wide chunk of each
// token row [heads*head_dim]; same matrix as kvmem_hadamard_rows (H^2 = I) but
// an O(n log n) butterfly. Rounding differs from the dense reference only at
// F32 epsilon, which the approximate PIC path tolerates. nrot must be 2^k.
void hadamard_rows_fast(float * rows, size_t n_rows, int heads, int head_dim, int nrot);

rope_status relocate_packed_k(const packed_k_rope_config & cfg,
                              const std::vector<rope_position> & source_positions,
                              const std::vector<rope_position> & destination_positions,
                              const void * packed, size_t packed_bytes,
                              std::vector<uint8_t> & output, std::string & reason);

} // namespace kvmem_pic
