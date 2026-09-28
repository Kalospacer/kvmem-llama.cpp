#include "llama-kvmem-pic-math.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace kvmem_pic {
namespace {

void require(bool ok, const char * message) {
    if (!ok) throw std::invalid_argument(message);
}

void check_tensor(const ggml_tensor * t) {
    require(t && t->type == GGML_TYPE_F32, "PIC math requires F32 tensors");
    require(ggml_is_contiguous(t), "PIC math requires contiguous tensors; capture a ggml_cont copy");
    for (int i = 0; i < 4; ++i) {
        require(t->ne[i] > 0 && t->ne[i] <= INT32_MAX, "PIC math tensor dimension out of range");
    }
}

void check_inputs(const ggml_tensor * k, const ggml_tensor * v,
                  const ggml_tensor * g, const ggml_tensor * b) {
    for (const auto * t : {k, v, g, b}) check_tensor(t);
    require(k->ne[0] == v->ne[0] && v->ne[1] % k->ne[1] == 0,
            "PIC math requires equal key/value dimensions and Hv divisible by Hk");
    require(g->ne[0] == 1 && b->ne[0] == 1 && g->ne[1] == v->ne[1] && b->ne[1] == v->ne[1],
            "PIC math requires scalar log gates and beta per value head");
    for (int i : {2, 3}) {
        require(k->ne[i] == v->ne[i] && g->ne[i] == v->ne[i] && b->ne[i] == v->ne[i],
                "PIC math token/sequence dimensions differ");
    }
}

void check_state(const ggml_tensor * s, int64_t d, int64_t h, int64_t b) {
    check_tensor(s);
    require(s->ne[0] == d && s->ne[1] == d && s->ne[2] == h && s->ne[3] == b,
            "PIC math state must have shape [d,d,Hv,B]");
}

size_t state_elements(const transition_data & tr) {
    size_t n = 1;
    for (int64_t dim : {tr.d, tr.d, tr.heads, tr.sequences}) {
        require(dim > 0 && dim <= INT32_MAX, "PIC math state dimension out of range");
        const size_t max_elements = std::min<size_t>(SIZE_MAX, INT64_MAX) / sizeof(float);
        require(n <= max_elements / static_cast<size_t>(dim), "PIC math state size overflow");
        n *= static_cast<size_t>(dim);
    }
    require(tr.t.size() == n && tr.u.size() == n, "PIC math T/U payload size mismatch");
    return n;
}

void check_finite(const std::vector<float> & data) {
    require(std::all_of(data.begin(), data.end(), [](float x) { return std::isfinite(x); }),
            "PIC math payload contains non-finite values");
}

ggml_tensor * final_state(ggml_context * ctx, ggml_tensor * result, const ggml_tensor * v) {
    const int64_t d = v->ne[0], h = v->ne[1], n = v->ne[2], b = v->ne[3];
    return ggml_view_4d(ctx, result, d, d, h, b,
                        sizeof(float) * d, sizeof(float) * d * d,
                        sizeof(float) * d * d * h, sizeof(float) * d * h * n * b);
}

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

context_ptr make_context() {
    context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    if (!ctx) throw std::runtime_error("PIC math context allocation failed");
    return ctx;
}

bool supports_graph(ggml_backend_t backend, ggml_cgraph * graph) {
    if (!backend) return false;
    const auto device = ggml_backend_get_device(backend);
    const bool cuda = device && std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") == 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        // The CUDA support query currently accepts widths that its launcher aborts on.
        if (cuda && node->op == GGML_OP_GATED_DELTA_NET) {
            const int64_t d = node->src[2]->ne[0];
            if (d != 16 && d != 32 && d != 64 && d != 128) return false;
        }
        if (!ggml_backend_supports_op(backend, node)) return false;
    }
    return true;
}

ggml_backend_t select_backend(ggml_backend_t preferred, ggml_backend_t cpu, ggml_cgraph * graph) {
    const auto cpu_device = cpu ? ggml_backend_get_device(cpu) : nullptr;
    require(cpu_device && ggml_backend_dev_type(cpu_device) == GGML_BACKEND_DEVICE_TYPE_CPU,
            "PIC math requires a caller-owned CPU fallback backend");
    if (supports_graph(preferred, graph)) return preferred;
    if (!supports_graph(cpu, graph)) throw std::runtime_error("PIC math graph unsupported by CPU fallback");
    return cpu;
}

buffer_ptr allocate(ggml_context * ctx, ggml_backend_t backend) {
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("PIC math scratch allocation failed");
    ggml_backend_buffer_clear(buffer.get(), 0);
    return buffer;
}

std::vector<float> read(const ggml_tensor * t) {
    require(t && t->data, "PIC math input/output tensor is not allocated");
    std::vector<float> data(static_cast<size_t>(ggml_nelements(t)));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    check_finite(data);
    return data;
}

void write(ggml_tensor * t, const std::vector<float> & data) {
    require(data.size() == static_cast<size_t>(ggml_nelements(t)), "PIC math tensor payload size mismatch");
    check_finite(data);
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

void compute(ggml_backend_t backend, ggml_cgraph * graph) {
    const auto status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("PIC math backend compute failed");
}

void report(execution_info * info, ggml_backend_t backend, ggml_backend_t preferred) {
    if (info) *info = {ggml_backend_name(backend), backend != preferred};
}

} // namespace

transition build_scan(ggml_context * ctx, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * log_gate, ggml_tensor * beta, const transition * prefix) {
    require(ctx != nullptr, "PIC math requires a GGML context");
    check_inputs(k, v, log_gate, beta);
    const int64_t d = v->ne[0], h = v->ne[1], b = v->ne[3];
    transition seed{};
    if (prefix) {
        check_state(prefix->t, d, h, b);
        check_state(prefix->u, d, h, b);
        seed = *prefix;
    } else {
        auto * ones = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d), 1.0f);
        seed.t = ggml_repeat_4d(ctx, ggml_diag(ctx, ones), d, d, h, b);
        seed.u = ggml_fill(ctx, seed.t, 0.0f);
    }
    auto * zero_v = ggml_fill(ctx, v, 0.0f);
    // q=k saves a dummy Q allocation. Attention output is deliberately unused.
    auto * t_scan = ggml_gated_delta_net(ctx, k, k, zero_v, log_gate, beta, seed.t, 1);
    auto * u_scan = ggml_gated_delta_net(ctx, k, k, v, log_gate, beta, seed.u, 1);
    return {final_state(ctx, t_scan, v), final_state(ctx, u_scan, v)};
}

ggml_tensor * build_compose(ggml_context * ctx, const transition & tr, ggml_tensor * state) {
    require(ctx != nullptr, "PIC math requires a GGML context");
    check_tensor(state);
    check_state(state, state->ne[0], state->ne[2], state->ne[3]);
    check_state(tr.t, state->ne[0], state->ne[2], state->ne[3]);
    check_state(tr.u, state->ne[0], state->ne[2], state->ne[3]);
    // GGML mul_mat computes A^T B; state storage is column-major S[key,value].
    auto * product = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, tr.t)), state);
    ggml_mul_mat_set_prec(product, GGML_PREC_F32);
    return ggml_add(ctx, product, tr.u);
}

transition_data scan(ggml_backend_t preferred, ggml_backend_t cpu,
                     const ggml_tensor * k, const ggml_tensor * v,
                     const ggml_tensor * log_gate, const ggml_tensor * beta,
                     const transition_data * prefix, execution_info * info) {
    check_inputs(k, v, log_gate, beta);
    if (prefix) {
        state_elements(*prefix);
        require(prefix->d == v->ne[0] && prefix->heads == v->ne[1] && prefix->sequences == v->ne[3],
                "PIC math prefix shape differs from scan state");
        check_finite(prefix->t);
        check_finite(prefix->u);
    }
    auto ctx = make_context();
    const ggml_tensor * sources[] = {k, v, log_gate, beta};
    ggml_tensor * inputs[4];
    for (int i = 0; i < 4; ++i) inputs[i] = ggml_dup_tensor(ctx.get(), sources[i]);
    transition seed{};
    if (prefix) {
        seed.t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, prefix->d, prefix->d, prefix->heads, prefix->sequences);
        seed.u = ggml_dup_tensor(ctx.get(), seed.t);
    }
    auto out = build_scan(ctx.get(), inputs[0], inputs[1], inputs[2], inputs[3], prefix ? &seed : nullptr);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, out.t);
    ggml_build_forward_expand(graph, out.u);
    const auto backend = select_backend(preferred, cpu, graph);
    auto buffer = allocate(ctx.get(), backend);
    for (int i = 0; i < 4; ++i) write(inputs[i], read(sources[i]));
    if (prefix) {
        write(seed.t, prefix->t);
        write(seed.u, prefix->u);
    }
    compute(backend, graph);
    transition_data result{v->ne[0], v->ne[1], v->ne[3], read(out.t), read(out.u)};
    report(info, backend, preferred);
    return result;
}

std::vector<float> compose(ggml_backend_t preferred, ggml_backend_t cpu,
                           const transition_data & tr, const std::vector<float> & state,
                           execution_info * info) {
    require(state.size() == state_elements(tr), "PIC math compose state payload size mismatch");
    check_finite(tr.t);
    check_finite(tr.u);
    check_finite(state);
    auto ctx = make_context();
    auto * s = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, tr.d, tr.d, tr.heads, tr.sequences);
    transition inputs{ggml_dup_tensor(ctx.get(), s), ggml_dup_tensor(ctx.get(), s)};
    auto * out = build_compose(ctx.get(), inputs, s);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, out);
    const auto backend = select_backend(preferred, cpu, graph);
    auto buffer = allocate(ctx.get(), backend);
    write(inputs.t, tr.t);
    write(inputs.u, tr.u);
    write(s, state);
    compute(backend, graph);
    auto result = read(out);
    report(info, backend, preferred);
    return result;
}

} // namespace kvmem_pic
