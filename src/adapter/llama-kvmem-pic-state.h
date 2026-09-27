#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

class llama_memory_recurrent;

namespace kvmem_pic {

struct layer_state {
    int32_t layer = -1;
    // GGML order: recurrent [128,128,48], conv [3,10240].
    std::vector<float> recurrent;
    std::vector<float> conv;
};

struct recurrent_state {
    llama_pos end = 0; // Exclusive logical position; empty capture has end == 0.
    std::vector<layer_state> layers; // Ascending model layer IDs; excludes FA layers.
};

// Caller must synchronize the context and exclude concurrent decode/state changes.
// Supports a single seq0, scalar Qwen GDN with 48 value heads, allocated F32 R/S,
// and no PLE history. Capture follows the pending rollback plane and source row.
// Failure leaves out unchanged. Empty seq0 captures the implicit zero state.
LLAMA_API bool pic_state_capture(llama_memory_recurrent & mem, recurrent_state & out, std::string & error);

// Caller supplies composed state with state.end == end > 0, and validates model,
// segment and attention provenance. Prepares ALL host rows and cell metadata
// before writing plane 0; commits seq0 at end-1 and invalidates old rollback planes.
// A false result requires the caller to restore its clean recurrent/KV checkpoint.
// Backend tensor_set is void: no recoverable GPU error check is implied. If an
// exception follows the first write, memory stays poisoned until checkpoint restore.
// This changes recurrent/conv state only, not FA KV, MTP carry or model logits.
LLAMA_API bool pic_state_install(llama_memory_recurrent & mem, const recurrent_state & state,
                                llama_pos end, std::string & error);

} // namespace kvmem_pic
