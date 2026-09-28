#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "../src/adapter/llama-kvmem-pic-math.h"
#include "../src/adapter/pic-rope.h"
#include "../src/adapter/llama-kvmem-quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static std::vector<float> values(size_t n, uint32_t seed, float scale) {
    std::vector<float> result(n);
    for (float & x : result) {
        seed = seed * 1664525u + 1013904223u;
        x = (float(int32_t(seed >> 8)) / 8388608.0f - 1.0f) * scale;
    }
    return result;
}

static void set(ggml_tensor * t, uint32_t seed, float scale) {
    const auto data = values(ggml_nelements(t), seed, scale);
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> result(ggml_nelements(t));
    ggml_backend_tensor_get(t, result.data(), 0, ggml_nbytes(t));
    return result;
}

// Shared oracle from the Vulkan replay regression suite.
static std::vector<double> reference_fold(
        const std::vector<float> & initial, const std::vector<float> & key,
        const std::vector<float> & value, const std::vector<float> & gate,
        const std::vector<float> & beta, int keep) {
    std::vector<double> state(initial.begin(), initial.end());
    for (int t = 0; t < keep; ++t) {
        for (int h = 0; h < 48; ++h) {
            const double decay = std::exp(double(gate[t * 48 + h]));
            const float * k = key.data() + (t * 16 + h % 16) * 128;
            for (int c = 0; c < 128; ++c) {
                double * s = state.data() + (h * 128 + c) * 128;
                double dot = 0;
                for (int r = 0; r < 128; ++r) dot += decay * s[r] * k[r];
                const double delta = (value[(t * 48 + h) * 128 + c] - dot) * beta[t * 48 + h];
                for (int r = 0; r < 128; ++r) s[r] = decay * s[r] + k[r] * delta;
            }
        }
    }
    return state;
}

static std::vector<float> snapshot(ggml_backend_t backend, int tokens) {
    ggml_init_params params{2 * 1024 * 1024, nullptr, true};
    auto * ctx = ggml_init(params);
    require(ctx != nullptr, "context allocation failed");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
    auto * b = ggml_dup_tensor(ctx, g);
    auto * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 128, 48, 1);
    auto * out = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 3);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "buffer allocation failed");
    ggml_backend_buffer_clear(buffer, 0);
    set(q, 123, 0.12f);
    set(k, 234, 0.12f);
    set(v, 345, 0.25f);
    set(g, 456, 0.02f);
    set(b, 567, 0.5f);
    set(s, 678, 0.1f);
    const auto initial = get(s);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "snapshot compute failed");
    const auto result = get(out);
    require(initial == get(s), "snapshot overwrote input state");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static void check_replay(ggml_backend_t backend, int tokens, int rounds, bool fold = true, int capacity = 0) {
    if (capacity == 0) capacity = tokens;
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "replay context allocation failed");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
    auto * k = ggml_dup_tensor(ctx, q);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
    auto * b = ggml_dup_tensor(ctx, g);
    auto * state_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128 * 128 * 48, 1);
    auto * s = ggml_reshape_4d(ctx, state_cache, 128, 128, 48, 1);
    auto * conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * 10240, 1);
    auto * conv_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10240, tokens);
#ifdef GGML_USE_CUDA
    auto * descriptor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(ggml_cuda_gdn_replay_layer));
#else
    require(!fold, "CUDA Fold is not compiled in this test");
#endif
    auto * reference = ggml_gated_delta_net(ctx, q, k, v, g, b, s, tokens);
    auto * recorded = ggml_gated_delta_net(ctx, q, k, v, g, b, s, 0);
    auto * graph = ggml_new_graph(ctx);
    for (int i = 0; i < 5; ++i) {
        ggml_build_forward_expand(graph, ggml_cpy(ctx, inputs[i],
                    ggml_view_1d(ctx, records[i], ggml_nelements(inputs[i]), 0)));
    }
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, recorded);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "replay buffer allocation failed");
    set(q, 12, 0.1f);
    set(k, 23, 0.1f);
    set(v, 34, 0.2f);
    set(s, 45, 0.1f);
    set(conv, 56, 0.1f);
    set(conv_input, 67, 0.1f);
    auto gates = values(ggml_nelements(g), 78, .05f);
    auto betas = values(ggml_nelements(b), 89, .9f);
    for (auto & x : gates) x = -std::fabs(x);
    for (auto & x : betas) x = std::fabs(x);
    ggml_backend_tensor_set(g, gates.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, betas.data(), 0, ggml_nbytes(b));
#ifdef GGML_USE_CUDA
    const ggml_cuda_gdn_replay_layer layer{static_cast<float *>(s->data), static_cast<float *>(conv->data),
        static_cast<float *>(records[0]->data), static_cast<float *>(records[1]->data), static_cast<float *>(records[2]->data),
        static_cast<float *>(records[3]->data), static_cast<float *>(records[4]->data)};
    ggml_backend_tensor_set(descriptor, &layer, 0, sizeof(layer));
#endif
    const auto original_key = get(k);
    const auto original_value = get(v);
    const auto columns = get(conv_input);
#ifdef GGML_USE_CUDA
    if (fold) require(ggml_backend_cuda_gdn_fold(nullptr, 0, 0, 0, nullptr), "zero Fold must not access descriptors");
#endif
    for (int round = 0; round < rounds; ++round) {
        ggml_backend_tensor_set(k, original_key.data(), 0, ggml_nbytes(k));
        const auto initial = get(s);
        const auto history = get(conv);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Record compute failed");
        require(initial == get(s), "Record changed committed state");
        require(history == get(conv), "Record changed committed convolution history");
        for (int i = 0; i < 5; ++i) {
            const auto input = get(inputs[i]);
            const auto record = get(records[i]);
            require(std::memcmp(input.data(), record.data(), input.size() * sizeof(float)) == 0,
                    "Replay capture differs from forward input");
        }
        const auto snapshots = get(reference);
        const auto output = get(recorded);
        require(std::memcmp(snapshots.data(), output.data(), ggml_nbytes(recorded)) == 0, "Record attention differs");
        if (!fold) continue;
        const int first = rounds == 1 ? 0 : round % (tokens + 1);
        const int last = rounds == 1 ? tokens : first;
        for (int keep = first; keep <= last; ++keep) {
            ggml_backend_tensor_set(s, initial.data(), 0, ggml_nbytes(s));
            ggml_backend_tensor_set(conv, history.data(), 0, ggml_nbytes(conv));
            auto key = original_key;
            std::fill(key.begin() + keep * 2048, key.end(), std::numeric_limits<float>::quiet_NaN());
            ggml_backend_tensor_set(k, key.data(), 0, ggml_nbytes(k));
#ifdef GGML_USE_CUDA
            require(ggml_backend_cuda_gdn_fold(static_cast<const ggml_cuda_gdn_replay_layer *>(descriptor->data),
                        1, keep, tokens, nullptr), "Fold launch failed");
#endif
            const auto actual = get(s);
            const float * expected = keep == 0 ? initial.data()
                : snapshots.data() + output.size() + (tokens - keep) * initial.size();
            require(std::memcmp(expected, actual.data(), ggml_nbytes(s)) == 0, "Fold state differs from accepted snapshot");
            if (rounds == 1) {
                const auto oracle = reference_fold(initial, original_key, original_value, gates, betas, keep);
                for (size_t i = 0; i < actual.size(); ++i) {
                    require(std::isfinite(actual[i]) && std::fabs(actual[i] - oracle[i]) < 2e-6 + 2e-5 * std::fabs(oracle[i]),
                            "Fold differs from independent FP64 recurrence");
                }
            }
            const auto actual_conv = get(conv);
            for (int c = 0; c < 10240; ++c) {
                for (int i = 0; i < 3; ++i) {
                    const int source = keep + i;
                    const float expected_conv = source < 3 ? history[c * 3 + source] : columns[(source - 3) * 10240 + c];
                    require(actual_conv[c * 3 + i] == expected_conv, "Fold conv history differs");
                }
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("PASS %s: width=%d capacity=%d rounds=%d\n", fold ? "Record/Fold (all prefixes, rejected NaNs, nonzero state)" : "CPU Record", tokens, capacity, rounds);
}

static void check_multilayer(ggml_backend_t backend) {
    constexpr int count = 4, tokens = 6, keep = 2;
    struct layer_t {
        ggml_tensor * k;
        ggml_tensor * v;
        ggml_tensor * g;
        ggml_tensor * b;
        ggml_tensor * s;
        ggml_tensor * conv;
        ggml_tensor * input;
        ggml_tensor * reference;
    };
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "multilayer context allocation failed");
    auto * graph = ggml_new_graph(ctx);
    std::vector<layer_t> layers;
    for (int il = 0; il < count; ++il) {
        layer_t l{};
        l.k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 16, tokens, 1);
        l.v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, 48, tokens, 1);
        l.g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 48, tokens, 1);
        l.b = ggml_dup_tensor(ctx, l.g);
        l.s = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128 * 128 * 48, 1);
        l.conv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3 * 10240, 1);
        l.input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 10240, tokens);
        l.reference = ggml_gated_delta_net(ctx, l.k, l.k, l.v, l.g, l.b,
                ggml_reshape_4d(ctx, l.s, 128, 128, 48, 1), tokens);
        ggml_build_forward_expand(graph, l.reference);
        layers.push_back(l);
    }
    auto * descriptors = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, count * sizeof(ggml_cuda_gdn_replay_layer));
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "multilayer buffer allocation failed");
    std::vector<ggml_cuda_gdn_replay_layer> gpu_layers;
    std::vector<std::vector<float>> histories;
    uint32_t seed = 100;
    for (const auto & l : layers) {
        for (auto * t : {l.k, l.v, l.s, l.conv, l.input}) set(t, seed++, .1f);
        auto gates = values(48 * tokens, seed++, .05f);
        auto betas = values(48 * tokens, seed++, .9f);
        for (auto & x : gates) x = -std::fabs(x);
        for (auto & x : betas) x = std::fabs(x);
        ggml_backend_tensor_set(l.g, gates.data(), 0, ggml_nbytes(l.g));
        ggml_backend_tensor_set(l.b, betas.data(), 0, ggml_nbytes(l.b));
        gpu_layers.push_back({static_cast<float *>(l.s->data), static_cast<float *>(l.conv->data),
                static_cast<float *>(l.k->data), static_cast<float *>(l.v->data), static_cast<float *>(l.g->data),
                static_cast<float *>(l.b->data), static_cast<float *>(l.input->data)});
        histories.push_back(get(l.conv));
    }
    ggml_backend_tensor_set(descriptors, gpu_layers.data(), 0, ggml_nbytes(descriptors));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "multilayer reference failed");
    require(ggml_backend_cuda_gdn_fold(static_cast<const ggml_cuda_gdn_replay_layer *>(descriptors->data),
                count, keep, tokens, nullptr), "multilayer fold failed");
    for (int il = 0; il < count; ++il) {
        const auto & l = layers[il];
        const auto reference = get(l.reference), actual = get(l.s), conv = get(l.conv), input = get(l.input);
        const float * expected = reference.data() + 128 * 48 * tokens + (tokens - keep) * 128 * 128 * 48;
        require(std::memcmp(expected, actual.data(), ggml_nbytes(l.s)) == 0, "multilayer state differs");
        for (int c = 0; c < 10240; ++c) {
            for (int i = 0; i < 3; ++i) {
                const int source = keep + i;
                const float value = source < 3 ? histories[il][c * 3 + source] : input[(source - 3) * 10240 + c];
                require(conv[c * 3 + i] == value, "multilayer convolution differs");
            }
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    std::printf("PASS native multilayer fold: layers=%d\n", count);
}

static void pic_close(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    require(actual.size() == expected.size(), "PIC comparison size mismatch");
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            error > 3e-5f + 2e-4f * std::fabs(expected[i])) {
            std::fprintf(stderr, "PIC %s mismatch index=%zu actual=%.9g expected=%.9g max_error=%.9g\n",
                         label, i, actual[i], expected[i], max_error);
            throw std::runtime_error("PIC numerical comparison failed");
        }
    }
}

template<class F>
static void pic_invalid(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "PIC accepted invalid input");
}

static void check_pic_case(ggml_backend_t preferred, ggml_backend_t cpu,
                           int d, int hk, int hv, int tokens, int sequences, bool zero_beta = false) {
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}), ggml_free);
    require(ctx != nullptr, "PIC test context allocation failed");
    auto * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, hk, tokens, sequences);
    auto * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, hv, tokens, sequences);
    auto * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, hv, tokens, sequences);
    auto * b = ggml_dup_tensor(ctx.get(), g);
    auto * s = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, d, hv, sequences);
    auto * reference = ggml_gated_delta_net(ctx.get(), k, k, v, g, b, s, 1);
    auto tr = kvmem_pic::build_scan(ctx.get(), k, v, g, b);
    auto * composed = kvmem_pic::build_compose(ctx.get(), tr, s);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, reference);
    ggml_build_forward_expand(graph, tr.t);
    ggml_build_forward_expand(graph, tr.u);
    ggml_build_forward_expand(graph, composed);

    const int split = tokens / 2;
    ggml_tensor * chunks[2][4] = {};
    ggml_tensor * inputs[] = {k, v, g, b};
    if (split > 0) {
        for (int part = 0; part < 2; ++part) {
            const int start = part == 0 ? 0 : split;
            const int count = part == 0 ? split : tokens - split;
            for (int i = 0; i < 4; ++i) {
                auto * src = inputs[i];
                chunks[part][i] = ggml_cont(ctx.get(), ggml_view_4d(ctx.get(), src,
                    src->ne[0], src->ne[1], count, sequences,
                    src->nb[1], src->nb[2], src->nb[3], start * src->nb[2]));
                ggml_build_forward_expand(graph, chunks[part][i]);
            }
        }
    }
    // Width 24 exercises the real CUDA-to-CPU dispatch guard; reference stays on CPU.
    const auto reference_backend = preferred && d != 24 ? preferred : cpu;
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), reference_backend), ggml_backend_buffer_free);
    require(buffer != nullptr, "PIC test allocation failed");
    ggml_backend_buffer_clear(buffer.get(), 0);
    auto key = values(ggml_nelements(k), 912, 0.8f);
    for (size_t row = 0; row < key.size(); row += d) {
        double norm = 0.0;
        for (int i = 0; i < d; ++i) norm += double(key[row + i]) * key[row + i];
        for (int i = 0; i < d; ++i) key[row + i] /= float(std::sqrt(norm));
    }
    ggml_backend_tensor_set(k, key.data(), 0, ggml_nbytes(k));
    set(v, 734, 0.4f);
    set(s, 456, 0.3f);
    auto gates = values(ggml_nelements(g), 323, 0.025f);
    auto betas = values(ggml_nelements(b), 817, 0.9f);
    for (auto & x : gates) x = -std::fabs(x);
    for (auto & x : betas) x = zero_beta ? 0.0f : std::fabs(x);
    ggml_backend_tensor_set(g, gates.data(), 0, ggml_nbytes(g));
    ggml_backend_tensor_set(b, betas.data(), 0, ggml_nbytes(b));
    const auto initial = get(s);
    const auto original_v = get(v);
    require(ggml_backend_graph_compute(reference_backend, graph) == GGML_STATUS_SUCCESS, "PIC reference compute failed");
    ggml_backend_synchronize(reference_backend);
    const auto full = get(reference);
    const size_t offset = static_cast<size_t>(d) * hv * tokens * sequences;
    const std::vector<float> expected(full.begin() + offset, full.end());
    pic_close(get(composed), expected, "graph compose vs full scan");

    kvmem_pic::execution_info info;
    const auto data = kvmem_pic::scan(preferred, cpu, k, v, g, b, nullptr, &info);
    if (d == 24) {
        require(info.backend == ggml_backend_name(cpu), "unsupported GDN width did not use CPU");
        require(info.cpu_fallback == (preferred != cpu), "CPU fallback reporting differs");
    }
    pic_close(data.t, get(tr.t), "executed T");
    pic_close(data.u, get(tr.u), "executed U");
    pic_close(kvmem_pic::compose(preferred, cpu, data, initial), expected, "executed compose vs full scan");

    // Exercise the fallback entry points even on CPU-only hosts.
    const auto fallback = kvmem_pic::scan(nullptr, cpu, k, v, g, b, nullptr, &info);
    require(info.cpu_fallback && info.backend == ggml_backend_name(cpu), "CPU scan fallback was not executed");
    pic_close(fallback.t, data.t, "CPU T vs selected backend");
    pic_close(fallback.u, data.u, "CPU U vs selected backend");
    pic_close(kvmem_pic::compose(nullptr, cpu, fallback, initial, &info), expected, "CPU compose vs full scan");
    require(info.cpu_fallback && info.backend == ggml_backend_name(cpu), "CPU compose fallback was not executed");

    if (split > 0) {
        const auto first = kvmem_pic::scan(preferred, cpu, chunks[0][0], chunks[0][1], chunks[0][2], chunks[0][3]);
        const auto joined = kvmem_pic::scan(preferred, cpu, chunks[1][0], chunks[1][1], chunks[1][2], chunks[1][3], &first);
        pic_close(joined.t, data.t, "chunked T vs full T");
        pic_close(joined.u, data.u, "chunked U vs full U");
        pic_close(kvmem_pic::compose(preferred, cpu, joined, initial), expected, "chunked compose vs full scan");
    }
    if (zero_beta) {
        auto decayed = initial;
        for (int seq = 0; seq < sequences; ++seq) {
            for (int head = 0; head < hv; ++head) {
                double sum = 0.0;
                for (int t = 0; t < tokens; ++t) sum += gates[(seq * tokens + t) * hv + head];
                const size_t base = static_cast<size_t>(seq * hv + head) * d * d;
                for (int i = 0; i < d * d; ++i) decayed[base + i] *= float(std::exp(sum));
            }
        }
        pic_close(data.u, std::vector<float>(initial.size(), 0.0f), "zero beta U");
        pic_close(expected, decayed, "analytic decay");
    }
    if (d == 16 && tokens == 17 && sequences == 1) {
        // Reuse the SAME transition with another nonzero state, not another cache build.
        set(s, 9981, 0.65f);
        const auto other_state = get(s);
        require(ggml_backend_graph_compute(reference_backend, graph) == GGML_STATUS_SUCCESS, "PIC second full scan failed");
        ggml_backend_synchronize(reference_backend);
        const auto other_full = get(reference);
        const std::vector<float> other_expected(other_full.begin() + offset, other_full.end());
        pic_close(kvmem_pic::compose(preferred, cpu, data, other_state), other_expected, "cached transition with another state");
        ggml_backend_tensor_set(s, initial.data(), 0, ggml_nbytes(s));
    }
    require(get(s) == initial && get(k) == key && get(v) == original_v && get(g) == gates && get(b) == betas,
            "PIC math modified captured inputs or initial state");

    auto wrong = data;
    wrong.u.pop_back();
    pic_invalid([&] { kvmem_pic::compose(preferred, cpu, wrong, initial); });
    pic_invalid([&] { kvmem_pic::compose(preferred, cpu, data, {}); });
    wrong = data;
    wrong.t[0] = std::numeric_limits<float>::quiet_NaN();
    pic_invalid([&] { kvmem_pic::compose(preferred, cpu, wrong, initial); });
    if (tokens == 1) {
        auto invalid_key = key;
        invalid_key[0] = std::numeric_limits<float>::quiet_NaN();
        ggml_backend_tensor_set(k, invalid_key.data(), 0, ggml_nbytes(k));
        pic_invalid([&] { kvmem_pic::scan(nullptr, cpu, k, v, g, b); });
        require(get(s) == initial, "failed PIC scan changed initial state");
        ggml_backend_tensor_set(k, key.data(), 0, ggml_nbytes(k));
    }
    std::printf("PASS PIC T/U, compose, full scan, chunking, CPU fallback: d=%d Hk=%d Hv=%d N=%d B=%d zero_beta=%d\n",
                d, hk, hv, tokens, sequences, zero_beta);
}

static void check_pic(ggml_backend_t preferred, ggml_backend_t cpu) {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "PIC validation context allocation failed");
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 2, 3, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 6, 3, 1);
    auto * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 6, 3, 1);
    auto * f16 = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 16, 2, 3, 1);
    auto * bad_heads = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 4, 3, 1);
    auto * bad_gate = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 6, 3, 1);
    auto * bad_batch = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 6, 3, 2);
    auto * bad_state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 16, 15, 6, 1);
    pic_invalid([&] { kvmem_pic::build_scan(ctx, nullptr, v, g, g); });
    pic_invalid([&] { kvmem_pic::build_scan(ctx, f16, v, g, g); });
    pic_invalid([&] { kvmem_pic::build_scan(ctx, bad_heads, v, g, g); });
    pic_invalid([&] { kvmem_pic::build_scan(ctx, k, v, bad_gate, g); });
    pic_invalid([&] { kvmem_pic::build_scan(ctx, k, v, g, bad_batch); });
    pic_invalid([&] { kvmem_pic::build_scan(ctx, ggml_transpose(ctx, k), v, g, g); });
    pic_invalid([&] { kvmem_pic::build_compose(ctx, {bad_state, bad_state}, bad_state); });
    ggml_free(ctx);
    for (int n : {1, 17, 129, 512}) check_pic_case(preferred, cpu, 16, 2, 6, n, 1);
    check_pic_case(preferred, cpu, 32, 2, 6, 17, 2);
    check_pic_case(preferred, cpu, 128, 16, 48, 17, 1);
    check_pic_case(preferred, cpu, 24, 2, 6, 17, 1);
    check_pic_case(preferred, cpu, 16, 2, 6, 17, 2, true);
}

static std::vector<float> rope_reference(ggml_backend_t cpu, const kvmem_pic::packed_k_rope_config & cfg,
                                       const std::vector<float> & input,
                                       const std::vector<kvmem_pic::rope_position> & positions, bool inverse) {
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    require(ctx != nullptr, "RoPE reference context allocation failed");
    const int64_t n = static_cast<int64_t>(positions.size());
    const bool multi = cfg.mode == GGML_ROPE_TYPE_MROPE || cfg.mode == GGML_ROPE_TYPE_IMROPE;
    auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cfg.head_dim, cfg.heads, n);
    auto * p = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n * (multi ? 4 : 1));
    auto sections = cfg.sections;
    ggml_tensor * out;
    if (multi) {
        out = inverse ? ggml_rope_multi_back(ctx.get(), x, p, nullptr, cfg.n_rot, sections.data(), cfg.mode,
            cfg.n_ctx_orig, cfg.freq_base, cfg.freq_scale, cfg.ext_factor, cfg.attn_factor, cfg.beta_fast, cfg.beta_slow)
            : ggml_rope_multi(ctx.get(), x, p, nullptr, cfg.n_rot, sections.data(), cfg.mode,
            cfg.n_ctx_orig, cfg.freq_base, cfg.freq_scale, cfg.ext_factor, cfg.attn_factor, cfg.beta_fast, cfg.beta_slow);
    } else {
        out = inverse ? ggml_rope_ext_back(ctx.get(), x, p, nullptr, cfg.n_rot, cfg.mode,
            cfg.n_ctx_orig, cfg.freq_base, cfg.freq_scale, cfg.ext_factor, cfg.attn_factor, cfg.beta_fast, cfg.beta_slow)
            : ggml_rope_ext(ctx.get(), x, p, nullptr, cfg.n_rot, cfg.mode,
            cfg.n_ctx_orig, cfg.freq_base, cfg.freq_scale, cfg.ext_factor, cfg.attn_factor, cfg.beta_fast, cfg.beta_slow);
    }
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu), ggml_backend_buffer_free);
    require(buffer != nullptr, "RoPE reference buffer allocation failed");
    require(input.size() == static_cast<size_t>(ggml_nelements(x)), "RoPE reference shape mismatch");
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    std::vector<int32_t> planes(static_cast<size_t>(ggml_nelements(p)));
    for (int axis = 0; axis < (multi ? 4 : 1); ++axis) {
        for (int64_t token = 0; token < n; ++token) planes[axis * n + token] = positions[token][axis];
    }
    ggml_backend_tensor_set(p, planes.data(), 0, ggml_nbytes(p));
    require(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS, "RoPE reference compute failed");
    ggml_backend_synchronize(cpu);
    return get(out);
}

static void check_pic_rope_case(ggml_backend_t cpu, ggml_type type, int mode, int d, int n_rot, int hadamard) {
    using namespace kvmem_pic;
    packed_k_rope_config cfg;
    cfg.type = type;
    cfg.head_dim = d;
    cfg.heads = 2;
    cfg.n_rot = n_rot;
    cfg.mode = mode;
    cfg.n_ctx_orig = 131072;
    cfg.hadamard_nrot = hadamard;
    if (mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE)
        cfg.sections = {n_rot / 4, n_rot / 8, n_rot / 8, 0};
    const std::vector<rope_position> source = {{0,0,0,0}, {7,7,7,7}, {4096,4096,4096,4096}, {65537,65537,65537,65537}};
    const std::vector<rope_position> destination = {{11,11,11,11}, {7,7,7,7}, {3,3,3,3}, {2,2,2,2}};
    const size_t row_elements = static_cast<size_t>(d) * cfg.heads;
    const size_t elements = source.size() * row_elements;
    const auto original = values(elements, 3345, 0.8f);
    auto old_rotated = rope_reference(cpu, cfg, original, source, false);
    if (hadamard) kvmem_hadamard_rows(old_rotated.data(), source.size(), cfg.heads, d, hadamard);
    const size_t row_bytes = ggml_row_size(type, row_elements);
    const auto * traits = ggml_get_type_traits(type);
    std::vector<uint8_t> packed(source.size() * row_bytes);
    traits->from_float_ref(old_rotated.data(), packed.data(), elements);
    const auto saved = packed;
    std::vector<float> decoded(elements);
    traits->to_float(packed.data(), decoded.data(), elements);
    if (hadamard) kvmem_hadamard_rows(decoded.data(), source.size(), cfg.heads, d, hadamard);
    const auto canonical = rope_reference(cpu, cfg, decoded, source, true);
    auto expected = rope_reference(cpu, cfg, canonical, destination, false);
    if (hadamard) kvmem_hadamard_rows(expected.data(), source.size(), cfg.heads, d, hadamard);

    std::vector<uint8_t> actual;
    std::string reason;
    const auto status = relocate_packed_k(cfg, source, destination, packed.data(), packed.size(), actual, reason);
    require(status == rope_status::ok, reason.c_str());
    require(actual.size() == packed.size() && reason.empty(), "RoPE output size/reason differs");
    std::vector<float> actual_f32(elements);
    traits->to_float(actual.data(), actual_f32.data(), elements);
    float max_error = 0.0f;
    for (size_t i = 0; i < elements; ++i) {
        // The unchanged token is intentionally copied, not re-quantized.
        if (i / row_elements == 1) continue;
        float tolerance = 1e-5f + 0.0006f * std::fabs(expected[i]);
        if (type == GGML_TYPE_Q8_0) {
            const size_t block = i / 32 * 32;
            float peak = 0.0f;
            for (size_t j = block; j < block + 32; ++j) peak = std::max(peak, std::fabs(expected[j]));
            tolerance = 1e-5f + 0.65f * peak / 127.0f;
        }
        const float error = std::fabs(actual_f32[i] - expected[i]);
        max_error = std::max(max_error, error);
        require(std::isfinite(actual_f32[i]) && error <= tolerance, "packed RoPE differs from GGML beyond codec rounding");
    }
    require(std::memcmp(actual.data() + row_bytes, saved.data() + row_bytes, row_bytes) == 0,
            "unchanged-position row lost packed identity");
    require(packed == saved, "RoPE changed source bytes");
    std::vector<uint8_t> identity;
    require(relocate_packed_k(cfg, source, source, packed.data(), packed.size(), identity, reason) == rope_status::ok && identity == packed,
            "same-position RoPE is not byte-identical");
    auto aliased = packed;
    require(relocate_packed_k(cfg, source, destination, aliased.data(), aliased.size(), aliased, reason) == rope_status::ok && aliased == actual,
            "aliased RoPE output differs");
    std::vector<uint8_t> unaligned(packed.size() + 1);
    std::memcpy(unaligned.data() + 1, packed.data(), packed.size());
    std::vector<uint8_t> from_unaligned;
    require(relocate_packed_k(cfg, source, destination, unaligned.data() + 1, packed.size(), from_unaligned, reason) == rope_status::ok && from_unaligned == actual,
            "unaligned RoPE input differs");
    if (type == GGML_TYPE_F16 && hadamard == 0 && n_rot < d) {
        for (size_t head = 0; head < source.size() * cfg.heads; ++head) {
            const size_t offset = (head * d + n_rot) * sizeof(ggml_fp16_t);
            require(std::memcmp(actual.data() + offset, packed.data() + offset, (d - n_rot) * sizeof(ggml_fp16_t)) == 0,
                    "RoPE changed unrotated channels");
        }
    }
    std::printf("PASS PIC packed RoPE vs GGML: type=%s mode=%d d=%d n_rot=%d H=%d max_error=%.9g\n",
                ggml_type_name(type), mode, d, n_rot, hadamard, max_error);
}

static void check_pic_fast_hadamard() {
    for (int nrot : {64, 128, 256}) {
        const int heads = 4, head_dim = 256, rows = 3;
        std::vector<float> dense(size_t(rows) * heads * head_dim);
        for (size_t i = 0; i < dense.size(); ++i) dense[i] = std::sin(0.37f * float(i) + 0.11f) * 3.0f;
        auto fast = dense;
        kvmem_hadamard_rows(dense.data(), rows, heads, head_dim, nrot);
        kvmem_pic::hadamard_rows_fast(fast.data(), rows, heads, head_dim, nrot);
        float max_error = 0.0f;
        for (size_t i = 0; i < dense.size(); ++i) max_error = std::max(max_error, std::fabs(dense[i] - fast[i]));
        require(max_error < 1e-4f, "fast PIC Hadamard differs from dense reference");
        std::printf("PASS PIC fast Hadamard vs dense: nrot=%d max_error=%.9g\n", nrot, max_error);
    }
}

static void check_pic_rope(ggml_backend_t cpu) {
    using namespace kvmem_pic;
    check_pic_fast_hadamard();
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        for (int mode : {GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX, GGML_ROPE_TYPE_MROPE, GGML_ROPE_TYPE_IMROPE})
            check_pic_rope_case(cpu, type, mode, 128, 64, type == GGML_TYPE_Q8_0 ? 128 : 0);
    }
    check_pic_rope_case(cpu, GGML_TYPE_Q8_0, GGML_ROPE_TYPE_IMROPE, 256, 64, 256);
    check_pic_rope_case(cpu, GGML_TYPE_Q8_0, GGML_ROPE_TYPE_NEOX, 128, 128, 0);
    packed_k_rope_config cfg;
    cfg.head_dim = 128; cfg.heads = 1; cfg.n_rot = 64; cfg.n_ctx_orig = 8192;
    std::vector<rope_position> source = {{0,0,0,0}, {1,1,1,1}};
    std::vector<rope_position> destination = {{2,2,2,2}, {3,3,3,3}};
    std::vector<uint8_t> packed(2 * ggml_row_size(cfg.type, cfg.head_dim), 0);
    const std::vector<uint8_t> sentinel = {11, 22, 33};
    auto reject = [&](const packed_k_rope_config & bad, const std::vector<rope_position> & src,
                      const std::vector<rope_position> & dst, const void * bytes, size_t size, rope_status expected) {
        auto output = sentinel;
        std::string reason;
        require(relocate_packed_k(bad, src, dst, bytes, size, output, reason) == expected && !reason.empty() && output == sentinel,
                "RoPE rejection failed or published partial output");
    };
    auto bad = cfg; bad.type = GGML_TYPE_Q4_0;
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::unsupported);
    bad = cfg; bad.freq_scale = std::nextafter(1.0f, 2.0f);
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::unsupported);
    bad = cfg; bad.attn_factor = 1.01f;
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::unsupported);
    bad = cfg; bad.ext_factor = 1.0f;
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::unsupported);
    bad = cfg; bad.mode = GGML_ROPE_TYPE_VISION;
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::unsupported);
    bad = cfg; bad.mode = GGML_ROPE_TYPE_MROPE; bad.sections = {1,1,1,0};
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::invalid_input);
    bad = cfg; bad.n_rot_offset = 2;
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::unsupported);
    bad = cfg; bad.freq_factors = {1.0f};
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::unsupported);
    bad = cfg; bad.beta_fast = std::numeric_limits<float>::quiet_NaN();
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::invalid_input);
    bad = cfg; bad.hadamard_nrot = 64;
    reject(bad, source, destination, packed.data(), packed.size(), rope_status::unsupported);
    auto spatial = destination; spatial[1][2] = 9;
    reject(cfg, source, spatial, packed.data(), packed.size(), rope_status::unsupported);
    auto negative = source; negative[0].fill(-1);
    reject(cfg, negative, destination, packed.data(), packed.size(), rope_status::invalid_input);
    reject(cfg, source, destination, packed.data(), packed.size() - 1, rope_status::invalid_input);
    reject(cfg, source, destination, nullptr, packed.size(), rope_status::invalid_input);
    const ggml_fp16_t nan = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    std::memcpy(packed.data() + ggml_row_size(cfg.type, cfg.head_dim), &nan, sizeof(nan));
    reject(cfg, source, destination, packed.data(), packed.size(), rope_status::invalid_input);
    std::fill(packed.begin(), packed.end(), 0);
    const ggml_fp16_t hi = ggml_fp32_to_fp16(65504.0f), lo = ggml_fp32_to_fp16(-65504.0f);
    std::memcpy(packed.data(), &hi, sizeof(hi));
    std::memcpy(packed.data() + (cfg.n_rot / 2) * sizeof(lo), &lo, sizeof(lo));
    reject(cfg, source, destination, packed.data(), packed.size(), rope_status::invalid_input);
    std::printf("PASS PIC packed RoPE rejection and atomic output\n");
}

int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        const bool pic_only = argc == 2 && (std::string(argv[1]) == "--pic" || std::string(argv[1]) == "--pic-cpu");
        const bool cpu = argc == 2 && (std::string(argv[1]) == "--cpu" || std::string(argv[1]) == "--pic-cpu");
        if (argc == 1 || cpu || pic_only) {
            auto * cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            require(cpu_device != nullptr, "PIC tests require a CPU backend");
            using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
            backend_ptr cpu_backend(ggml_backend_dev_init(cpu_device, nullptr), ggml_backend_free);
            require(cpu_backend != nullptr, "CPU backend initialization failed");
            auto * gpu_device = cpu ? nullptr : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            backend_ptr gpu_backend(gpu_device ? ggml_backend_dev_init(gpu_device, nullptr) : nullptr, ggml_backend_free);
            if (gpu_device) require(gpu_backend != nullptr, "GPU backend initialization failed");
            auto preferred = gpu_backend ? gpu_backend.get() : cpu_backend.get();
            check_pic(preferred, cpu_backend.get());
            check_pic_rope(cpu_backend.get());
            if (!pic_only) {
                bool fold = false;
#ifdef GGML_USE_CUDA
                fold = gpu_device && std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(gpu_device)), "CUDA") == 0;
#endif
                for (int width : {1, 2, 3, 4, 5, 6}) check_replay(preferred, width, 1, fold);
                if (fold) check_replay(preferred, 6, 1000);
            }
            return 0;
        }
        auto * device = ggml_backend_dev_by_type(cpu ? GGML_BACKEND_DEVICE_TYPE_CPU : GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) return 77;
        auto backend = ggml_backend_dev_init(device, nullptr);
        require(backend != nullptr, "GPU backend initialization failed");
        const bool write = argc == 3 && std::string(argv[1]) == "--write";
        const bool check = argc == 3 && std::string(argv[1]) == "--check";
        require(write || check, "usage: gdn-replay-test [--cpu|--pic|--pic-cpu|--write FILE|--check FILE]");
        std::fstream file(argv[2], std::ios::binary | (write ? std::ios::out | std::ios::trunc : std::ios::in));
        require(bool(file), "cannot open snapshot fixture");
        for (int tokens : {1, 2, 3, 17, 512}) {
            const auto result = snapshot(backend, tokens);
            const size_t bytes = result.size() * sizeof(float);
            if (write) {
                file.write(reinterpret_cast<const char *>(result.data()), bytes);
            } else {
                std::vector<float> expected(result.size());
                file.read(reinterpret_cast<char *>(expected.data()), bytes);
                require(bool(file), "truncated snapshot fixture");
                require(std::memcmp(expected.data(), result.data(), bytes) == 0, "FP32 snapshot differs from original kernel");
            }
            std::printf("PASS original snapshots: tokens=%d bytes=%zu\n", tokens, bytes);
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
