#pragma once

#include "llama.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_cparams;
struct llm_graph_result;
class llama_kvmem_pic_graph_input;

// Host F32, contiguous in GGML order (ne[0] is the fastest dimension).
struct llama_kvmem_pic_tensor {
    std::array<int64_t, 4> ne = {};
    std::vector<float> data;
};

struct llama_kvmem_pic_layer_capture {
    int32_t layer = -1;
    // [head_dim, heads, tokens_per_sequence, sequence_sets]. K is L2 normalized.
    // K heads may differ from V heads with fused GDN; preserve the actual input.
    llama_kvmem_pic_tensor k_conv;
    llama_kvmem_pic_tensor v_conv;
    // [1, value_heads, tokens_per_sequence, sequence_sets]. Gate is log decay,
    // not exp(gate); beta has already passed through sigmoid.
    llama_kvmem_pic_tensor gate;
    llama_kvmem_pic_tensor beta;
    // [history + tokens_per_sequence, channels, sequence_sets, 1], before SiLU.
    // The history prefix and raw QKV suffix permit boundaries inside a ubatch.
    llama_kvmem_pic_tensor conv_input;
    // [history, channels, sequence_sets, 1], before/after this ubatch.
    llama_kvmem_pic_tensor conv_begin;
    llama_kvmem_pic_tensor conv_end;
};

struct llama_kvmem_pic_ubatch_capture {
    uint64_t ordinal = 0;
    uint32_t n_tokens = 0;
    uint32_t n_seq_tokens = 0;
    uint32_t n_seqs = 0;
    uint32_t n_pos = 0;
    // All position planes, as supplied by llama_ubatch (plane-major for M-RoPE).
    std::vector<llama_pos> positions;
    std::vector<llama_pos> logical_positions;
    // Token row = sequence_set * n_seq_tokens + token_in_sequence.
    // One set of sequence IDs per token, including shared-prefix membership.
    std::vector<std::vector<llama_seq_id>> seq_ids;
    std::vector<llama_kvmem_pic_layer_capture> layers;
    // [n_embd, n_tokens, 1, 1], final RMS norm, before output-row selection.
    llama_kvmem_pic_tensor target_hidden;
};

// One collector per trunk context. Install before context creation, and keep it
// alive until that context is freed. Call control/read methods only between
// synchronized decodes on the same thread. No process-global capture state.
class LLAMA_API llama_kvmem_pic_capture {
public:
    llama_kvmem_pic_capture();
    ~llama_kvmem_pic_capture();
    llama_kvmem_pic_capture(const llama_kvmem_pic_capture &) = delete;
    llama_kvmem_pic_capture & operator=(const llama_kvmem_pic_capture &) = delete;

    // Preserve and chain the existing eval callback. Do not replace this wrapper
    // after installation. Install a different collector for a different context.
    void install(llama_context_params & params);
    void start(); // Clear the previous session and enable capture.
    // Disable capture. False on missing callbacks, incomplete ubatch, or error.
    // Callers must also check llama_decode's result before using captured data.
    bool stop();
    bool enabled() const;
    const std::string & error() const;
    // Move completed ubatches to a segment builder; pending data stays private.
    std::vector<llama_kvmem_pic_ubatch_capture> take_ubatches();
    uint64_t captured_ubatches() const;

    static bool eval_callback(ggml_tensor * tensor, bool ask, void * user_data);

private:
    struct impl;
    std::unique_ptr<impl> state;
    friend class llama_kvmem_pic_graph_input;
};

// Qwen3.5 graph integration. These helpers do nothing unless the wrapper is
// installed and enabled, except attach() also guards reuse of disabled graphs.
enum class llama_kvmem_pic_field { k_conv, v_conv, gate, beta, conv_input, target_hidden };

bool llama_kvmem_pic_capture_enabled(const llama_cparams & params);
void llama_kvmem_pic_capture_tensor(ggml_context * ctx, ggml_cgraph * graph,
        const llama_cparams & params, ggml_tensor * tensor, llama_kvmem_pic_field field, int layer);
void llama_kvmem_pic_capture_attach(llm_graph_result * result, const llama_cparams & params,
        const std::vector<int32_t> & recurrent_layers);
