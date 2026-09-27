#include "llama-kvmem-pic-state.h"

#include "llama-memory-recurrent.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kvmem_pic {
namespace {

constexpr int64_t conv_width = 3 * 10240;
constexpr int64_t recurrent_width = 128 * 128 * 48;

struct layer_io {
    int32_t layer;
    ggml_tensor * r;
    ggml_tensor * s;
};

struct memory_layout {
    llama_pos end = 0;
    size_t source_row = 0;
    std::vector<layer_io> layers;
};

void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}

void validate_tensor(const ggml_tensor * t, int64_t width, int64_t rows) {
    require(t && t->type == GGML_TYPE_F32, "PIC state requires F32 recurrent/conv tensors");
    require(t->ne[0] == width && t->ne[1] == rows && t->ne[2] == 1 && t->ne[3] == 1 &&
            ggml_is_contiguous(t) && t->nb[0] == sizeof(float) && t->nb[1] == (size_t) width * sizeof(float),
            "PIC state tensor shape/stride mismatch");
    require(t->buffer && t->data && !t->view_src, "PIC state requires allocated, non-view tensors");
    require((uint64_t) rows <= std::numeric_limits<size_t>::max() / ((size_t) width * sizeof(float)),
            "PIC state tensor byte count overflow");
    require(ggml_nbytes(t) == (size_t) rows * (size_t) width * sizeof(float), "PIC state tensor storage mismatch");
}

memory_layout describe(const llama_memory_recurrent & mem) {
    require(mem.supports_pic_state(), "PIC state supports only Qwen 48-head scalar GDN");
    require(!mem.replay_recording && !mem.replay_poisoned && mem.replay_cells.empty(),
            "PIC state rejects unsettled or poisoned replay");
    require(mem.size > 0 && mem.size <= (uint32_t) std::numeric_limits<int32_t>::max() &&
            mem.cells.size() == mem.size && mem.rs_idx.size() == 1 && mem.rollback_floor.size() == 1,
            "PIC state requires single-sequence memory");
    require(mem.used <= 1 && mem.n <= mem.size && mem.head < mem.size &&
            mem.rs_idx[0] <= mem.n_rs_seq, "PIC state has invalid row metadata");
    require(mem.rs_z >= -1 && (mem.rs_z < 0 || (uint32_t) mem.rs_z < mem.size), "PIC state has invalid zero row");
    require(mem.r_l.size() == mem.s_l.size() && mem.p_l.size() == mem.r_l.size(), "PIC state layer table mismatch");
    require(std::all_of(mem.p_l.begin(), mem.p_l.end(), [](const ggml_tensor * p) { return !p; }),
            "PIC state does not support PLE history");

    memory_layout result;
    int32_t cell_id = -1;
    for (size_t i = 0; i < mem.cells.size(); ++i) {
        const auto & cell = mem.cells[i];
        if (cell.is_empty()) {
            require(cell.pos == -1, "PIC state has a positioned empty cell");
            continue;
        }
        require(cell_id == -1 && cell.seq_id.size() == 1 && cell.has_seq_id(0), "PIC state contains other sequences");
        require(cell.pos >= 0 && cell.pos < std::numeric_limits<llama_pos>::max(), "PIC state position overflow");
        cell_id = (int32_t) i;
    }
    if (cell_id < 0) {
        require(mem.used == 0 && mem.cells[0].tail == -1 && mem.rs_idx[0] == 0 && mem.rollback_floor[0] == -1,
                "PIC state empty-sequence metadata mismatch");
    } else {
        const auto & cell = mem.cells[cell_id];
        require(mem.used == 1 && mem.cells[0].tail == cell_id && mem.rollback_floor[0] >= -1 &&
                cell.pos >= mem.rollback_floor[0], "PIC state tail/floor mismatch");
        require(cell.src >= -1 && (cell.src < 0 || (uint32_t) cell.src < mem.size), "PIC state source row out of range");
        result.end = cell.pos + 1;
        result.source_row = (size_t) mem.rs_idx[0] * mem.size + (cell.src >= 0 ? cell.src : cell_id);
    }

    const int64_t rows = (int64_t) mem.size * (1 + (int64_t) mem.n_rs_seq);
    require(rows > 0 && (uint64_t) rows <= std::numeric_limits<size_t>::max() && result.source_row < (uint64_t) rows,
            "PIC state snapshot plane out of range");
    require(mem.r_l.size() <= (size_t) std::numeric_limits<int32_t>::max(), "PIC state layer count overflow");
    for (size_t i = 0; i < mem.r_l.size(); ++i) {
        auto * r = mem.r_l[i];
        auto * s = mem.s_l[i];
        require((r == nullptr) == (s == nullptr), "PIC state missing recurrent/conv pair");
        if (!r) continue;
        validate_tensor(r, conv_width, rows);
        validate_tensor(s, recurrent_width, rows);
        result.layers.push_back({(int32_t) i, r, s});
    }
    require(!result.layers.empty(), "PIC state has no GDN layers");
    return result;
}

void validate_values(const std::vector<float> & values) {
    require(std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }),
            "PIC state contains non-finite values");
}

} // namespace

bool pic_state_capture(llama_memory_recurrent & mem, recurrent_state & out, std::string & error) {
    try {
        error.clear();
        const auto layout = describe(mem);
        recurrent_state prepared;
        prepared.end = layout.end;
        prepared.layers.resize(layout.layers.size());
        for (size_t i = 0; i < layout.layers.size(); ++i) {
            auto & layer = prepared.layers[i];
            layer.layer = layout.layers[i].layer;
            layer.conv.resize(conv_width, 0.0f);
            layer.recurrent.resize(recurrent_width, 0.0f);
        }
        if (layout.end > 0) {
            for (size_t i = 0; i < layout.layers.size(); ++i) {
                const auto & io = layout.layers[i];
                auto & layer = prepared.layers[i];
                ggml_backend_tensor_get(io.r, layer.conv.data(), layout.source_row * io.r->nb[1], layer.conv.size() * sizeof(float));
                ggml_backend_tensor_get(io.s, layer.recurrent.data(), layout.source_row * io.s->nb[1], layer.recurrent.size() * sizeof(float));
                validate_values(layer.conv);
                validate_values(layer.recurrent);
            }
        }
        out = std::move(prepared);
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}

bool pic_state_install(llama_memory_recurrent & mem, const recurrent_state & state, llama_pos end, std::string & error) {
    bool writing = false;
    try {
        error.clear();
        const auto layout = describe(mem);
        require(end > 0 && state.end == end, "PIC state install requires matching positive end position");
        require(state.layers.size() == layout.layers.size(), "PIC state install layer count mismatch");
        for (size_t i = 0; i < layout.layers.size(); ++i) {
            const auto & layer = state.layers[i];
            require(layer.layer == layout.layers[i].layer && layer.conv.size() == conv_width &&
                    layer.recurrent.size() == recurrent_width, "PIC state install layer shape/order mismatch");
            validate_values(layer.conv);
            validate_values(layer.recurrent);
        }
        auto prepared = state.layers;
        std::vector<llama_memory_recurrent::mem_cell> cells(mem.size);
        cells[0].pos = end - 1;
        cells[0].src = cells[0].src0 = cells[0].tail = 0;
        cells[0].seq_id.insert(0);

        writing = true;
        mem.replay_poisoned = true;
        for (size_t i = 0; i < layout.layers.size(); ++i) {
            const auto & io = layout.layers[i];
            const auto & layer = prepared[i];
            ggml_backend_tensor_set(io.r, layer.conv.data(), 0, layer.conv.size() * sizeof(float));
            ggml_backend_tensor_set(io.s, layer.recurrent.data(), 0, layer.recurrent.size() * sizeof(float));
        }
        // No allocations after tensor writes; old planes remain inaccessible.
        mem.cells.swap(cells);
        mem.head = 0;
        mem.n = mem.used = 1;
        mem.rs_z = -1;
        mem.rs_idx[0] = 0;
        mem.rollback_floor[0] = end - 1;
        mem.replay_cells.clear();
        mem.replay_start = -1;
        mem.replay_width = mem.replay_head = mem.replay_used = mem.replay_n = 0;
        mem.replay_rs_z = -1;
        mem.replay_recording = false;
        mem.replay_poisoned = false;
        return true;
    } catch (const std::exception & e) {
        error = writing ? std::string("PIC state install may be partial; restore checkpoint: ") + e.what() : e.what();
        return false;
    }
}

} // namespace kvmem_pic
