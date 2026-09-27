#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

// Layer-local affine GDN state math. This does not implement model-level PIC.
namespace kvmem_pic {

struct transition {
    ggml_tensor * t;
    ggml_tensor * u;
};

// Column-major S[key, value], matching GGML GDN: [d, d, heads, sequences].
struct transition_data {
    int64_t d = 0;
    int64_t heads = 0;
    int64_t sequences = 0;
    std::vector<float> t;
    std::vector<float> u;
};

struct execution_info {
    std::string backend;
    bool cpu_fallback = false;
};

// All inputs must be contiguous F32. k: [d,Hk,N,B], v: [d,Hv,N,B],
// log_gate/beta: [1,Hv,N,B], Hv % Hk == 0. N must be positive.
// Uses scalar log gates (not KDA); caller supplies normalized k and sigmoid beta.
// Without prefix: T = scan(I, V=0), U = scan(0, V=v).
// With prefix: extend its T/U through this chunk, in chronological order.
// Returned tensors belong to ctx. Expand BOTH outputs into the caller's graph.
// Constants are graph operations; no host seed initialization is needed.
// Graph-only calls do not dispatch backends or inspect tensor values. On CUDA,
// use d=16/32/64/128; the synchronous scan below handles other widths on CPU.
transition build_scan(ggml_context * ctx, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * log_gate, ggml_tensor * beta,
                      const transition * prefix = nullptr);

// Returns T*S+U without modifying any input. All tensors have the same F32
// [d,d,Hv,B] layout; matmul explicitly requests F32 precision.
ggml_tensor * build_compose(ggml_context * ctx, const transition & tr, ggml_tensor * state);

// Synchronous, isolated scratch execution using the same graph builders.
// Source tensors must be allocated and their producer must be synchronized.
// Reads captured inputs; never writes source tensors or installs live state.
// Backends are caller-owned. cpu must be a CPU backend. A null preferred backend
// or unsupported graph uses cpu; CUDA GDN widths outside 16/32/64/128 fall back
// before launch. Compute/allocation failures throw, rather than hide GPU errors.
// Inputs/results are checked for finite values. Exceptions publish no result.
transition_data scan(ggml_backend_t preferred, ggml_backend_t cpu,
                     const ggml_tensor * k, const ggml_tensor * v,
                     const ggml_tensor * log_gate, const ggml_tensor * beta,
                     const transition_data * prefix = nullptr,
                     execution_info * info = nullptr);

std::vector<float> compose(ggml_backend_t preferred, ggml_backend_t cpu,
                           const transition_data & tr, const std::vector<float> & state,
                           execution_info * info = nullptr);

} // namespace kvmem_pic
