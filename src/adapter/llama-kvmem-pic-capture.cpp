#include "llama-kvmem-pic-capture.h"

#include "llama-cparams.h"
#include "llama-graph.h"
#include "ggml-backend.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

using field = llama_kvmem_pic_field;

struct binding {
    size_t layer;
    field which;
};

using bindings = std::unordered_map<ggml_tensor *, binding>;

llama_kvmem_pic_capture * collector(const llama_cparams & params) {
    if (params.cb_eval != llama_kvmem_pic_capture::eval_callback) {
        return nullptr;
    }
    return static_cast<llama_kvmem_pic_capture *>(params.cb_eval_user_data);
}

std::string tensor_name(field which, int layer) {
    const char * name = nullptr;
    switch (which) {
        case field::k_conv:        name = "k"; break;
        case field::v_conv:        name = "v"; break;
        case field::gate:          name = "gate"; break;
        case field::beta:          name = "beta"; break;
        case field::conv_input:    name = "conv"; break;
        case field::target_hidden: name = "hidden"; break;
    }
    return std::string("kvmem_pic_") + name + "-" + std::to_string(layer);
}

llama_kvmem_pic_tensor read_tensor(ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) ||
            !tensor->buffer || !tensor->data) {
        throw std::runtime_error("PIC capture requires an allocated contiguous F32 tensor");
    }
    llama_kvmem_pic_tensor out;
    std::copy_n(tensor->ne, 4, out.ne.begin());
    out.data.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, out.data.data(), 0, out.data.size() * sizeof(float));
    return out;
}

void extract_conv_boundaries(llama_kvmem_pic_layer_capture & layer, uint32_t n_seq_tokens) {
    const auto & input = layer.conv_input;
    const int64_t history = input.ne[0] - n_seq_tokens;
    if (history < 0) {
        throw std::runtime_error("PIC convolution history has a negative length");
    }
    layer.conv_begin.ne = layer.conv_end.ne = { history, input.ne[1], input.ne[2], 1 };
    const size_t rows = size_t(input.ne[1]) * size_t(input.ne[2]);
    layer.conv_begin.data.resize(rows * size_t(history));
    layer.conv_end.data.resize(rows * size_t(history));
    for (size_t row = 0; history > 0 && row < rows; ++row) {
        const float * src = input.data.data() + row * size_t(input.ne[0]);
        std::copy_n(src, history, layer.conv_begin.data.data() + row * size_t(history));
        std::copy_n(src + n_seq_tokens, history, layer.conv_end.data.data() + row * size_t(history));
    }
}

void validate_layer(const llama_kvmem_pic_layer_capture & layer) {
    const auto & k = layer.k_conv.ne;
    const auto & v = layer.v_conv.ne;
    const auto & g = layer.gate.ne;
    const auto & b = layer.beta.ne;
    if (k[0] <= 0 || k[1] <= 0 || v[0] <= 0 || v[1] <= 0 || v[1] % k[1] != 0 ||
            g != b || g[0] != 1 || g[1] != v[1]) {
        throw std::runtime_error("PIC GDN input dimensions do not match");
    }
}

} // namespace

struct llama_kvmem_pic_capture::impl {
    bool active = false;
    std::string failure;
    uint64_t completed = 0;
    ggml_backend_sched_eval_callback next = nullptr;
    void * next_data = nullptr;
    std::unordered_set<ggml_tensor *> next_requested;
    bindings expected;
    std::unordered_set<ggml_tensor *> received;
    std::unique_ptr<llama_kvmem_pic_ubatch_capture> pending;
    std::vector<llama_kvmem_pic_ubatch_capture> ready;

    void fail(const std::string & message) {
        if (failure.empty()) {
            failure = message;
        }
        pending.reset();
        expected.clear();
        received.clear();
    }

    void begin(const llama_ubatch & batch, const bindings & nodes, const std::vector<int32_t> & layers) {
        if (!active || !failure.empty()) {
            return;
        }
        if (pending) {
            fail("PIC previous ubatch is incomplete; check eval callback and decode status");
            return;
        }
        if (!batch.equal_seqs() || !batch.n_tokens || !batch.n_seq_tokens || !batch.n_seqs ||
                uint64_t(batch.n_seq_tokens) * batch.n_seqs != batch.n_tokens ||
                !batch.n_pos || !batch.pos || !batch.n_seq_id || !batch.seq_id) {
            fail("PIC requires equal-length sequence sets and token position/sequence metadata");
            return;
        }
        expected = nodes;
        received.clear();
        pending.reset(new llama_kvmem_pic_ubatch_capture);
        auto & out = *pending;
        out.ordinal = completed;
        out.n_tokens = batch.n_tokens;
        out.n_seq_tokens = batch.n_seq_tokens;
        out.n_seqs = batch.n_seqs;
        out.n_pos = batch.n_pos;
        out.positions.assign(batch.pos, batch.pos + size_t(batch.n_tokens) * batch.n_pos);
        const auto * logical = batch.logical_pos ? batch.logical_pos : batch.pos;
        out.logical_positions.assign(logical, logical + batch.n_tokens);
        out.seq_ids.resize(batch.n_tokens);
        for (uint32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.n_seq_id[i] <= 0 || !batch.seq_id[i]) {
                throw std::runtime_error("PIC token has no sequence membership");
            }
            out.seq_ids[i].assign(batch.seq_id[i], batch.seq_id[i] + batch.n_seq_id[i]);
        }
        for (int32_t layer : layers) {
            llama_kvmem_pic_layer_capture entry;
            entry.layer = layer;
            out.layers.push_back(std::move(entry));
        }
    }

    void capture(ggml_tensor * tensor, const binding & node) {
        if (!received.insert(tensor).second) {
            throw std::runtime_error("PIC tensor evaluated twice in one ubatch");
        }
        auto & out = *pending;
        auto data = read_tensor(tensor);
        if (node.which == field::target_hidden) {
            if (data.ne[0] <= 0 || data.ne[1] != out.n_tokens || data.ne[2] != 1 || data.ne[3] != 1) {
                throw std::runtime_error("PIC target hidden does not contain every ubatch token");
            }
            out.target_hidden = std::move(data);
        } else {
            auto & layer = out.layers.at(node.layer);
            if (node.which == field::conv_input) {
                if (data.ne[1] <= 0 || data.ne[2] != out.n_seqs || data.ne[3] != 1) {
                    throw std::runtime_error("PIC convolution sequence dimensions do not match");
                }
                layer.conv_input = std::move(data);
                extract_conv_boundaries(layer, out.n_seq_tokens);
            } else {
                if (data.ne[2] != out.n_seq_tokens || data.ne[3] != out.n_seqs) {
                    throw std::runtime_error("PIC GDN token/sequence dimensions do not match");
                }
                switch (node.which) {
                    case field::k_conv: layer.k_conv = std::move(data); break;
                    case field::v_conv: layer.v_conv = std::move(data); break;
                    case field::gate:   layer.gate = std::move(data); break;
                    case field::beta:   layer.beta = std::move(data); break;
                    default: throw std::runtime_error("PIC unexpected capture field");
                }
            }
        }
        if (received.size() == expected.size()) {
            for (const auto & layer : out.layers) {
                validate_layer(layer);
            }
            ready.push_back(std::move(out));
            ++completed;
            pending.reset();
            expected.clear();
            received.clear();
        }
    }
};

class llama_kvmem_pic_graph_input : public llm_graph_input_i {
public:
    llama_kvmem_pic_graph_input(llm_graph_result * result, const llama_cparams & params,
            const std::vector<int32_t> & layers) : owner(collector(params)),
            active(owner && owner->enabled()), layers(layers) {
        if (!active) {
            return;
        }
        if (layers.empty()) {
            missing = "PIC graph has no recurrent layers";
        }
        for (size_t i = 0; i < layers.size(); ++i) {
            for (field which : { field::k_conv, field::v_conv, field::gate, field::beta, field::conv_input }) {
                add(result->get_gf(), which, layers[i], i);
            }
        }
        add(result->get_gf(), field::target_hidden, -1, 0);
    }

    void set_input(const llama_ubatch * batch) override {
        if (!active || !owner->enabled()) {
            return;
        }
        try {
            if (!missing.empty()) {
                owner->state->fail(missing);
            } else {
                owner->state->begin(*batch, nodes, layers);
            }
        } catch (const std::exception & ex) {
            owner->state->fail(ex.what());
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return collector(params.cparams) == owner && active == (owner && owner->enabled());
    }

private:
    void add(ggml_cgraph * graph, field which, int layer, size_t index) {
        const auto name = tensor_name(which, layer);
        auto * tensor = ggml_graph_get_tensor(graph, name.c_str());
        if (!tensor) {
            missing = "PIC graph is missing " + name;
        } else {
            nodes.emplace(tensor, binding{ index, which });
        }
    }

    llama_kvmem_pic_capture * owner;
    bool active;
    std::vector<int32_t> layers;
    bindings nodes;
    std::string missing;
};

llama_kvmem_pic_capture::llama_kvmem_pic_capture() : state(new impl) {}
llama_kvmem_pic_capture::~llama_kvmem_pic_capture() = default;

void llama_kvmem_pic_capture::install(llama_context_params & params) {
    if (params.cb_eval == eval_callback) {
        if (params.cb_eval_user_data != this) {
            throw std::invalid_argument("PIC callback already belongs to another collector");
        }
        return;
    }
    state->next = params.cb_eval;
    state->next_data = params.cb_eval_user_data;
    params.cb_eval = eval_callback;
    params.cb_eval_user_data = this;
}

void llama_kvmem_pic_capture::start() {
    state->ready.clear();
    state->pending.reset();
    state->expected.clear();
    state->received.clear();
    state->next_requested.clear();
    state->failure.clear();
    state->completed = 0;
    state->active = true;
}

bool llama_kvmem_pic_capture::stop() {
    state->active = false;
    if (state->pending) {
        state->fail("PIC final ubatch is incomplete; check eval callback and decode status");
    } else if (state->completed == 0 && state->failure.empty()) {
        state->fail("PIC session captured no ubatches");
    }
    return state->failure.empty();
}

bool llama_kvmem_pic_capture::enabled() const { return state->active; }
const std::string & llama_kvmem_pic_capture::error() const { return state->failure; }
uint64_t llama_kvmem_pic_capture::captured_ubatches() const { return state->completed; }

std::vector<llama_kvmem_pic_ubatch_capture> llama_kvmem_pic_capture::take_ubatches() {
    std::vector<llama_kvmem_pic_ubatch_capture> result;
    result.swap(state->ready);
    return result;
}

bool llama_kvmem_pic_capture::eval_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * self = static_cast<llama_kvmem_pic_capture *>(user_data);
    if (!self) {
        return !ask;
    }
    auto & s = *self->state;
    try {
        if (ask) {
            const bool own = s.active && s.failure.empty() && s.pending && s.expected.count(tensor);
            const bool chained = s.next && s.next(tensor, true, s.next_data);
            if (chained) {
                s.next_requested.insert(tensor);
            }
            return own || chained;
        }
        if (s.next_requested.erase(tensor) && s.next && !s.next(tensor, false, s.next_data)) {
            s.fail("PIC chained eval callback interrupted computation");
            return false;
        }
        const auto found = s.expected.find(tensor);
        if (s.active && s.failure.empty() && s.pending && found != s.expected.end()) {
            // GGML synchronizes the producing backend before the ask=false call.
            const auto node = found->second;
            s.capture(tensor, node);
        }
    } catch (const std::exception & ex) {
        s.fail(ex.what());
        return !ask;
    }
    return true;
}

bool llama_kvmem_pic_capture_enabled(const llama_cparams & params) {
    auto * capture = collector(params);
    return capture && capture->enabled();
}

void llama_kvmem_pic_capture_tensor(ggml_context * ctx, ggml_cgraph * graph,
        const llama_cparams & params, ggml_tensor * tensor, field which, int layer) {
    if (!llama_kvmem_pic_capture_enabled(params)) {
        return;
    }
    // A real copy node also makes strided views visible to the eval callback.
    auto * copy = tensor->type == GGML_TYPE_F32 ? ggml_cont(ctx, tensor) : ggml_cast(ctx, tensor, GGML_TYPE_F32);
    ggml_set_name(copy, tensor_name(which, layer).c_str());
    ggml_set_output(copy);
    ggml_build_forward_expand(graph, copy);
}

void llama_kvmem_pic_capture_attach(llm_graph_result * result, const llama_cparams & params,
        const std::vector<int32_t> & recurrent_layers) {
    if (!collector(params)) {
        return;
    }
    result->add_input(std::unique_ptr<llm_graph_input_i>(
            new llama_kvmem_pic_graph_input(result, params, recurrent_layers)));
}
