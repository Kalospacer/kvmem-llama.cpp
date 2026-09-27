#pragma once

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <set>
#include <numeric>
#include <map>

static void pool_maintain(ServerState & st);

static const char * multimodal_query_invalid(const ServerState & st, const MultimodalQuery & q,
                                            const kvmem_prompt & prompt, int rows) {
    if (q.generation != st.mm_cache_generation || q.model != st.model) return "query_generation";
    if (q.begin < 0 || q.end <= q.begin || q.end > rows ||
            (size_t) q.end > prompt.tokens.size() || q.prefix.size() != (size_t) q.end) return "query_rows";
    if (!std::equal(q.prefix.begin(), q.prefix.end(), prompt.tokens.begin())) return "query_prefix";
    if (q.media != prompt.media_identity()) return "query_media";
    const auto expected = (uint32_t) (q.end - q.begin);
    if (q.state.count.empty() || *std::max_element(q.state.count.begin(), q.state.count.end()) != expected ||
            std::any_of(q.state.count.begin(), q.state.count.end(),
                [&](uint32_t n) { return n != 0 && n != expected; })) return "query_counts";
    // Layer shapes, required-layer counts and finite values are checked by set_query.
    return nullptr;
}

static void multimodal_invalidate_queries(ServerState & st) {
    ++st.mm_cache_generation;
    st.mm_query.reset();
    st.mm_pending_query.reset();
    for (auto & e : st.pool) e->query.reset();
    llama_kvmem_reset_query();
    llama_kvmem_freeze_query(false);
}

static void multimodal_validate_capacity(const ServerState & st, const kvmem_prompt & prompt, int query, int end) {
    if (!st.kparams.enabled || !st.kparams.budget || !prompt.has_media()) return;
    const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
    const uint32_t budget = st.kparams.budget / block;
    std::vector<std::pair<uint32_t, uint32_t>> groups;
    for (const auto & range : prompt.media_ranges()) {
        const uint32_t lo = (range.first ? range.first - 1 : 0) / block;
        const uint32_t hi = (std::min<uint32_t>(end, range.second + 1) + block - 1) / block;
        if (!groups.empty() && lo < groups.back().second) groups.back().second = hi;
        else groups.emplace_back(lo, hi);
    }
    const uint32_t sink = std::max(1u, st.kparams.sink_tokens / block);
    for (const auto & group : groups) {
        if (group.second - group.first + std::min(group.first, sink) > budget)
            throw std::invalid_argument("image group exceeds KV budget; reduce --image-max-tokens or increase --kvmem-budget");
    }
    std::set<uint32_t> required;
    for (uint32_t i = 0; i < sink; ++i) required.insert(i);
    for (uint32_t i = groups.back().first; i < groups.back().second; ++i) required.insert(i);
    for (uint32_t i = std::max(0, query) / block; i < ((uint32_t) end + block - 1) / block; ++i) required.insert(i);
    if (required.size() > budget)
        throw std::invalid_argument("latest image and text query exceed KV budget; reduce the image size or query span");
}

// Included after the single-slot server state and stream helpers.
static MultimodalCheckpoint multimodal_checkpoint(ServerState & st, int row) {
    MultimodalCheckpoint result;
    result.row = row;
    if (st.mm_live_checkpoint && st.mm_live_row == row) {
        result.data = st.mm_live_checkpoint;
        ++st.mm_perf.shared;
        return result;
    }
    kvmem_scoped_ms timer(st.mm_perf.save_ms);
    ++st.mm_perf.saves;
    auto data = std::make_shared<MultimodalCheckpointData>();
    llama_synchronize(st.ctx);
    {
        kvmem_scoped_ms mean_timer(st.mm_perf.mean_ms);
        llama_kvmem_get_tail_mean(row, data->tail_mean);
    }
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t size = llama_state_seq_get_size_ext(st.ctx, 0, flags);
    data->recurrent.resize(size);
    if (llama_state_seq_get_data_ext(st.ctx, data->recurrent.data(), size, 0, flags) != size) {
        throw std::runtime_error("multimodal recurrent checkpoint failed");
    }
    {
        kvmem_scoped_ms carry_timer(st.mm_perf.carry_ms);
        if (st.spec.ok && !common_speculative_get_state(st.spec.spec, 0, data->draft_carry)) {
            throw std::runtime_error("MTP carry checkpoint failed");
        }
    }
    data->accounting = st.mm_checkpoint_accounting;
    data->accounting->live_bytes += data->bytes();
    data->accounting->peak_bytes = std::max(data->accounting->peak_bytes, data->accounting->live_bytes);
    result.data = std::move(data);
    st.mm_live_checkpoint = result.data;
    return result;
}

static void multimodal_remember(ServerState & st, MultimodalCheckpoint checkpoint) {
    auto & entries = st.mm_checkpoints;
    // Inserting drops everything at or past the new row, so rows stay ascending.
    entries.erase(std::remove_if(entries.begin(), entries.end(), [&](const auto & entry) {
        return entry.row >= checkpoint.row;
    }), entries.end());
    entries.push_back(std::move(checkpoint));
    const size_t cap = (size_t) std::max(4, st.mm_ckpt_max);
    const size_t keep_newest = 4;
    while (entries.size() > cap) {
        const auto media_count = std::count_if(entries.begin(), entries.end(), [](const auto & e) { return e.media_boundary; });
        const bool evict_media = media_count > 2;
        // Thin the densest interior region. The oldest entry (system/tool prefix)
        // and the newest few (recent turns, eval_end, commit) are what the next
        // turns restore from.
        size_t victim = entries.size();
        int best_gap = INT_MAX;
        for (size_t i = 1; i + keep_newest < entries.size(); ++i) {
            if (media_count > 0 && entries[i].media_boundary != evict_media) continue;
            const int gap = entries[i + 1].row - entries[i - 1].row;
            if (gap < best_gap) {
                best_gap = gap;
                victim = i;
            }
        }
        if (victim == entries.size()) {
            auto it = std::find_if(entries.begin(), entries.end(), [&](const auto & e) {
                return e.media_boundary == evict_media;
            });
            victim = it == entries.end() ? 0 : (size_t) (it - entries.begin());
        }
        entries.erase(entries.begin() + (std::ptrdiff_t) victim);
    }
}

// First-pass rows in (begin, end) that get a recurrent checkpoint: the first
// message after the system block, message starts within the window before the
// query, and a fixed interval. Message starts win over nearby interval rows.
static std::vector<int> multimodal_plan_checkpoints(const ServerState & st, const kvmem_prompt & prompt, int begin, int end) {
    std::vector<int> rows;
    const int gap = std::max(1, st.mm_ckpt_min_gap);
    if (end - begin <= 2 * gap) return rows;
    std::vector<int> candidates;
    if (st.mm_msg_token != LLAMA_TOKEN_NULL) {
        const int window_begin = st.mm_ckpt_window > 0 ? end - st.mm_ckpt_window : end;
        int seen = 0;
        for (int row = 0; row < end; ++row) {
            if (prompt.tokens[row] != st.mm_msg_token) continue;
            ++seen;
            if (row > begin && (seen == 2 || row >= window_begin)) candidates.push_back(row);
        }
    }
    const size_t n_messages = candidates.size();
    if (st.mm_ckpt_interval > 0) {
        for (int row = (begin / st.mm_ckpt_interval + 1) * st.mm_ckpt_interval; row < end; row += st.mm_ckpt_interval) {
            candidates.push_back(row);
        }
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
        const int row = candidates[i];
        if (row - begin < gap || end - row < gap) continue;
        if (prompt.tokens[row] == LLAMA_TOKEN_NULL) continue; // never split a media chunk
        const bool crowded = std::any_of(rows.begin(), rows.end(), [&](int kept) { return std::abs(kept - row) < gap; });
        if (!crowded) rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end());
    kvmem_diag("KVMEM_TRACE prefix_checkpoint_plan span=[%d,%d) message_candidates=%zu planned=%zu\n",
            begin, end, n_messages, rows.size());
    return rows;
}

static void multimodal_restore(ServerState & st, const MultimodalCheckpoint & checkpoint, bool truncate) {
    kvmem_scoped_ms timer(st.mm_perf.restore_ms);
    const bool live = st.mm_live_checkpoint == checkpoint.data && st.mm_live_row == checkpoint.row;
    if (!checkpoint.data) throw std::runtime_error("missing multimodal checkpoint data");
    if (live) ++st.mm_perf.restore_skips;
    else ++st.mm_perf.restores;
    llama_synchronize(st.ctx);
    if (st.spec.ctx_dft) llama_synchronize(st.spec.ctx_dft);
    llama_kvmem_decode_mean_flush();
    llama_kvmem_decode_mean_discard();
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const auto & data = checkpoint.data->recurrent;
    if (!live && llama_state_seq_set_data_ext(st.ctx, data.data(), data.size(), 0, flags) != data.size()) {
        throw std::runtime_error("multimodal recurrent restore failed");
    }
    if (!llama_kvmem_remove_logical(st.ctx, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted target rows");
    }
    if (st.spec.ctx_dft && !llama_kvmem_remove_logical(st.spec.ctx_dft, checkpoint.row, -1)) {
        throw std::runtime_error("cannot remove uncommitted MTP rows");
    }
    if (st.spec.ok && !live) {
        kvmem_scoped_ms carry_timer(st.mm_perf.carry_ms);
        common_speculative_set_state(st.spec.spec, 0, checkpoint.data->draft_carry);
    }
    if (truncate) {
        llama_kvmem_truncate_cached(checkpoint.row);
        llama_kvmem_set_tail_mean(checkpoint.row, checkpoint.data->tail_mean);
    }
    st.mm_live_row = checkpoint.row;
    st.mm_live_checkpoint = checkpoint.data;
}

static void multimodal_finish_request(ServerState & st) {
    if (st.mm_committed || !st.mm_rollback) return;
    st.mm_query.reset();
    st.mm_pending_query.reset();
    try {
        llama_kvmem_set_replay(false);
        MultimodalCheckpoint target = *st.mm_rollback;
        std::shared_ptr<kvmem_prompt> target_prompt = st.mm_rollback_prompt;
        if (st.mm_keep_aborted && st.active_prompt) {
            // Entries past the rollback row were all taken by this request on the
            // active prompt. Keep the newest so a retried prompt resumes there
            // instead of prefilling from the request start again.
            for (const auto & checkpoint : st.mm_checkpoints) {
                if (checkpoint.data && checkpoint.row > target.row && checkpoint.row <= st.mm_live_row) target = checkpoint;
            }
            if (target.row > st.mm_rollback->row) target_prompt = st.active_prompt->prefix(target.row);
        }
        multimodal_restore(st, target, true);
        llama_kvmem_begin_cached_turn();
        st.mm_query.reset();
        st.mm_pending_query.reset();
        st.cached_prompt = target_prompt;
        st.cached_tokens = st.cached_prompt ? st.cached_prompt->tokens : std::vector<llama_token>{};
        st.cached_tokens.resize(std::min(st.cached_tokens.size(), (size_t) st.mm_live_row));
        multimodal_remember(st, target);
        kvmem_diag("KVMEM_TRACE multimodal_rollback context=%p row=%d request_start=%d kept_progress=%d\n",
                (void *) st.ctx, st.mm_live_row, st.mm_rollback->row, st.mm_live_row - st.mm_rollback->row);
        kvmem_diag("KVMEM_CHECKPOINT_ROLLBACK live_bytes=%zu peak_bytes=%zu\n",
                st.mm_checkpoint_accounting->live_bytes, st.mm_checkpoint_accounting->peak_bytes);
        st.mm_committed = true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        LOG_ERR("srv    KVMEM_TRACE multimodal_rollback_failed error=%s\n", e.what());
        memory_clear_all(st);
        st.mm_committed = true;
    }
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
    pool_maintain(st);
}

static int multimodal_decode_span(ServerState & st, int begin, int end, bool replay, StreamIo * io,
                                  const std::vector<int> * checkpoint_rows = nullptr) {
    const auto & prompt = *st.active_prompt;
    auto dispatch = [&](llama_batch batch) -> int {
        if (!stream_heartbeat(io)) return KVMEM_DECODE_ABORT;
        kvmem_diag("KVMEM_TRACE multimodal_decode context=%p rows=[%d,%d) model_pos=%d image=%d replay=%d\n",
                (void *) st.ctx, batch.logical_pos[0], batch.logical_pos[batch.n_tokens - 1] + 1,
                batch.pos[0], batch.token == nullptr, replay);
        const auto start = std::chrono::steady_clock::now();
        const bool diagnostic = llama_kvmem_get_transfer_stats().enabled;
        st.mm_live_checkpoint.reset();
        int rc = llama_decode(st.ctx, batch);
        if (diagnostic) {
            llama_synchronize(st.ctx);
            st.mm_perf.target_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        const auto draft_start = std::chrono::steady_clock::now();
        if (rc == 0 && st.spec.ok && !common_speculative_process(st.spec.spec, batch)) rc = -1;
        if (diagnostic && st.spec.ok) {
            llama_synchronize(st.spec.ctx_dft);
            st.mm_perf.draft_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - draft_start).count();
        }
        llama_synchronize(st.ctx);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        (replay ? st.mm_perf.replay_ms : st.mm_perf.first_ms) += elapsed;
        kvmem_diag("KVMEM_TRACE multimodal_compute rows=%d elapsed_ms=%.3f image=%d replay=%d\n",
                batch.n_tokens, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(),
                batch.token == nullptr, replay);
        if (rc == 0) {
            st.mm_live_row = batch.logical_pos[batch.n_tokens - 1] + 1;
            if (replay) st.mm_replayed += batch.n_tokens;
            else {
                const int tail = std::clamp(st.mm_lcp - batch.logical_pos[0], 0, batch.n_tokens);
                st.mm_tail_replayed += tail;
                if (batch.token) st.mm_new_text += batch.n_tokens - tail;
                else st.mm_new_image += batch.n_tokens - tail;
                // Count actual first-pass work, including reconstruction after the
                // reused checkpoint; exclude the later retrieval query replay.
                st.log.prefilled(batch.n_tokens, (int) prompt.tokens.size() - batch.logical_pos[0]);
            }
        }
        return rc;
    };
    int row = begin;
    // Planned rows split the batches; a checkpoint is taken once a row is reached.
    size_t next_checkpoint = 0;
    auto next_checkpoint_row = [&]() {
        while (checkpoint_rows && next_checkpoint < checkpoint_rows->size() && (*checkpoint_rows)[next_checkpoint] < row) {
            ++next_checkpoint;
        }
        return checkpoint_rows && next_checkpoint < checkpoint_rows->size() ? (*checkpoint_rows)[next_checkpoint] : INT_MAX;
    };
    while (row < end) {
        if (!stream_heartbeat(io)) return KVMEM_DECODE_ABORT;
        if (!replay && next_checkpoint_row() == row && row > begin) {
            multimodal_remember(st, multimodal_checkpoint(st, row));
            ++next_checkpoint;
        }
        if (prompt.tokens[row] == LLAMA_TOKEN_NULL) {
            const int next = (int) prompt.media_end(row);
            if (next > end) throw std::runtime_error("prefill boundary splits an image");
            if (!replay) {
                auto checkpoint = multimodal_checkpoint(st, row);
                checkpoint.media_boundary = true;
                multimodal_remember(st, std::move(checkpoint));
            }
            const int rc = st.vision->decode(st.ctx, prompt, row, st.n_batch, dispatch);
            if (rc != 0) return rc;
            row = next;
            continue;
        }
        int limit = std::min(end, row + st.n_batch);
        const int planned = next_checkpoint_row();
        if (planned > row) limit = std::min(limit, planned);
        int next = row;
        while (next < limit && prompt.tokens[next] != LLAMA_TOKEN_NULL) ++next;
        std::vector<llama_pos> pos(next - row), logical(next - row);
        const auto pos0 = prompt.model_pos(row);
        for (int i = row; i < next; ++i) {
            pos[i - row] = pos0 + i - row;
            logical[i - row] = i;
        }
        llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(prompt.tokens.data()) + row, next - row);
        batch.pos = pos.data();
        batch.logical_pos = logical.data();
        std::vector<int32_t> n_seq(next - row, 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> seq_ids(next - row, &seq);
        std::vector<int8_t> outputs(next - row, 0);
        outputs.back() = !st.spec.ok;
        batch.n_seq_id = n_seq.data();
        batch.seq_id = seq_ids.data();
        batch.logits = outputs.data();
        const int rc = dispatch(batch);
        if (rc != 0) return rc;
        row = next;
    }
    return 0;
}

// ---- Conversation pool ------------------------------------------------------
// The slot holds one live conversation. Others are parked as host-side stashes
// with their recurrent checkpoints; a request that continues a parked
// conversation swaps it back (take) or copies its shared prefix (fork).

static bool pool_active(const ServerState & st) {
    return st.pool_enabled && st.kparams.enabled && st.pool_max > 0 && st.mm_msg_token != LLAMA_TOKEN_NULL;
}

// Largest checkpoint row (> 0) this conversation can resume the prompt from.
static int pool_resume_row(const kvmem_prompt & prompt, int cap, const kvmem_prompt * cached, int rows,
                           const std::vector<MultimodalCheckpoint> & checkpoints, int * lcp_out) {
    if (lcp_out) *lcp_out = 0;
    if (!cached || cached->has_media()) return -1;
    const int lcp = (int) prompt.common_prefix(*cached);
    if (lcp_out) *lcp_out = lcp;
    const int keep = std::min({lcp, rows, cap});
    int best = -1;
    for (const auto & checkpoint : checkpoints) {
        if (checkpoint.data && checkpoint.row > 0 && checkpoint.row <= keep && checkpoint.row > best) best = checkpoint.row;
    }
    return best;
}

// Would dropping rows (lcp, rows] lose something a later request can resume?
// A checkpoint at a message start means a later turn of that branch lands
// there; the rewritten tail of the current turn only holds query/end rows.
static bool pool_branch_worth(const ServerState & st, const kvmem_prompt & cached, int rows,
                              const std::vector<MultimodalCheckpoint> & checkpoints, int lcp, bool unrelated) {
    if (rows < st.pool_min_rows || rows - std::max(lcp, 0) < st.pool_min_gain) return false;
    for (const auto & checkpoint : checkpoints) {
        if (!checkpoint.data || checkpoint.row <= lcp || checkpoint.row > rows) continue;
        if (unrelated && checkpoint.row >= st.pool_min_rows) return true;
        if (checkpoint.row < (int) cached.tokens.size() && cached.tokens[checkpoint.row] == st.mm_msg_token) return true;
    }
    return false;
}

enum class PoolAllocationKind { raw, checkpoint, query, prompt_estimate, metadata };

struct PoolBytes {
    size_t logical = 0; // Sum over parked owners; shared payload counted per owner.
    size_t pool = 0, live = 0, combined = 0, shared_live = 0;
};

struct PoolEntryBytes {
    size_t logical = 0, reclaimable = 0, pool_only = 0;
};

struct PoolMemory {
    struct Allocation {
        size_t bytes = 0;
        PoolAllocationKind kind = PoolAllocationKind::raw;
        std::vector<uint64_t> owners; // 0 = live; positive IDs = parked entries.
    };
    std::map<const void *, Allocation> allocations;

    void add(const void * identity, size_t bytes, uint64_t owner, PoolAllocationKind kind) {
        if (!bytes) return;
        if (!identity) throw std::runtime_error("null pool allocation identity");
        auto result = allocations.try_emplace(identity);
        auto & a = result.first->second;
        if (result.second) {
            a.bytes = bytes;
            a.kind = kind;
        } else if (a.bytes != bytes || a.kind != kind) {
            throw std::runtime_error("pool allocation changed during snapshot");
        }
        if (std::find(a.owners.begin(), a.owners.end(), owner) == a.owners.end()) a.owners.push_back(owner);
    }

    PoolBytes totals(const PoolAllocationKind * kind = nullptr) const {
        PoolBytes result;
        for (const auto & item : allocations) {
            const auto & a = item.second;
            if (kind && a.kind != *kind) continue;
            const bool live = std::find(a.owners.begin(), a.owners.end(), uint64_t(0)) != a.owners.end();
            const size_t parked = a.owners.size() - (live ? 1 : 0);
            result.logical += a.bytes * parked;
            result.pool += parked ? a.bytes : 0;
            result.live += live ? a.bytes : 0;
            result.combined += a.bytes;
            result.shared_live += live && parked ? a.bytes : 0;
        }
        return result;
    }

    PoolEntryBytes entry(uint64_t id) const {
        PoolEntryBytes result;
        for (const auto & item : allocations) {
            const auto & a = item.second;
            if (std::find(a.owners.begin(), a.owners.end(), id) == a.owners.end()) continue;
            result.logical += a.bytes;
            if (a.owners.size() == 1) result.reclaimable += a.bytes;
            if (std::find(a.owners.begin(), a.owners.end(), uint64_t(0)) == a.owners.end()) result.pool_only += a.bytes;
        }
        return result;
    }

    void erase(uint64_t id) {
        for (auto it = allocations.begin(); it != allocations.end();) {
            auto & owners = it->second.owners;
            owners.erase(std::remove(owners.begin(), owners.end(), id), owners.end());
            if (owners.empty()) it = allocations.erase(it);
            else ++it;
        }
    }
};

// Called under the inference slot lock, between operations. Drain live writers
// FIRST, then retain every owner without decoding/take/put until collection ends.
// The snapshot owns no payload. Eviction removes owner IDs without dereferencing
// its now possibly freed allocation keys; no allocation hook is re-run per victim.
static PoolMemory pool_memory(ServerState & st) {
    PoolMemory memory;
    std::vector<std::pair<const void *, size_t>> raw;
    if (st.kparams.enabled && st.ctx) {
        llama_synchronize(st.ctx);
        if (st.spec.ctx_dft) llama_synchronize(st.spec.ctx_dft);
        if (!llama_kvmem_live_allocations(raw)) throw std::runtime_error("live allocation snapshot failed");
        for (const auto & a : raw) memory.add(a.first, a.second, 0, PoolAllocationKind::raw);
    }
    auto checkpoint = [&](const std::shared_ptr<const MultimodalCheckpointData> & data, uint64_t owner) {
        if (data) memory.add(data.get(), sizeof(*data) + data->bytes(), owner, PoolAllocationKind::checkpoint);
    };
    auto query = [&](const std::shared_ptr<const MultimodalQuery> & q, uint64_t owner) {
        if (q) memory.add(q.get(), q->bytes(), owner, PoolAllocationKind::query);
    };
    auto prompt = [&](const std::shared_ptr<kvmem_prompt> & p, uint64_t owner) {
        // Native token capacity is private. Media/embedding storage is excluded.
        if (p) memory.add(p.get(), sizeof(*p) + 2*p->tokens.capacity()*sizeof(llama_token),
                          owner, PoolAllocationKind::prompt_estimate);
    };
    for (const auto & e : st.pool) {
        raw.clear();
        if (!llama_kvmem_stash_allocations(e->stash, raw)) throw std::runtime_error("stash allocation snapshot failed");
        for (const auto & a : raw) memory.add(a.first, a.second, e->id, PoolAllocationKind::raw);
        memory.add(e.get(), sizeof(*e) + e->checkpoints.capacity()*sizeof(MultimodalCheckpoint),
                   e->id, PoolAllocationKind::metadata);
        prompt(e->prompt, e->id);
        for (const auto & c : e->checkpoints) checkpoint(c.data, e->id);
        query(e->query, e->id);
    }
    for (const auto & c : st.mm_checkpoints) checkpoint(c.data, 0);
    checkpoint(st.mm_live_checkpoint, 0);
    if (st.mm_rollback) checkpoint(st.mm_rollback->data, 0);
    query(st.mm_query, 0);
    query(st.mm_pending_query, 0);
    prompt(st.cached_prompt, 0);
    prompt(st.mm_rollback_prompt, 0);
    memory.add(st.mm_checkpoints.data(), st.mm_checkpoints.capacity()*sizeof(MultimodalCheckpoint), 0, PoolAllocationKind::metadata);
    memory.add(st.cached_tokens.data(), st.cached_tokens.capacity()*sizeof(llama_token), 0, PoolAllocationKind::metadata);
    return memory;
}

static void pool_publish(ServerState & st, const PoolMemory & memory) {
    const auto now = std::chrono::steady_clock::now();
    const auto total = memory.totals();
    const size_t budget = (size_t) (st.pool_gb * 1073741824.0);
    json entries = json::array();
    for (const auto & e : st.pool) {
        json rows = json::array();
        for (const auto & c : e->checkpoints) rows.push_back(c.row);
        const auto bytes = memory.entry(e->id);
        entries.push_back({{"id", e->id}, {"rows", e->rows}, {"bytes", bytes.logical}, {"hits", e->hits},
                           {"legacy_stash_payload_bytes", e->bytes}, {"query", e->query != nullptr},
                           {"query_end", e->query ? e->query->end : -1},
                           {"exclusive_bytes", bytes.reclaimable}, {"reclaimable_bytes", bytes.reclaimable},
                           {"shared_bytes", bytes.logical - bytes.reclaimable},
                           {"age_s", std::chrono::duration<double>(now - e->created).count()},
                           {"idle_s", std::chrono::duration<double>(now - e->last_used).count()},
                           {"checkpoint_rows", rows}});
    }
    json j = {{"enabled", pool_active(st)}, {"entries", entries}, {"accounting_valid", true},
              {"total_bytes", total.logical}, {"unique_bytes", total.pool},
              {"pool_logical_bytes", total.logical}, {"pool_unique_bytes", total.pool},
              {"live_unique_bytes", total.live}, {"pool_live_unique_bytes", total.combined},
              {"pool_live_shared_bytes", total.shared_live},
              {"pool_exclusive_bytes", total.pool - total.shared_live},
              {"budget_scope", "pool_and_live"}, {"budget_bytes", budget}, {"budget_used_bytes", total.combined},
              {"over_budget_bytes", total.combined > budget ? total.combined - budget : 0},
              {"live_over_budget_bytes", total.live > budget ? total.live - budget : 0},
              {"bytes_accounting", "allocation capacities deduplicated by owner; total_bytes=pool logical; unique_bytes=pool unique"},
              {"accounting_excludes", "allocator/control blocks, runtime tiers/scratch, fixed adapter state, GPU, media embeddings, request temporaries"},
              {"prompt_accounting", "estimated native/public token capacity"},
              {"query_enabled", st.pool_query_enabled}, {"restore", st.pool_restore_prefix ? "prefix" : "full"},
              {"copy", st.pool_copy_cow ? "cow" : "deep"}, {"max_entries", st.pool_max}, {"live_rows", st.mm_live_row}};
    llama_kvmem_resident_stats resident;
    const bool resident_available = llama_kvmem_get_resident_stats(resident);
    j["gpu_reuse"] = st.pool_gpu_reuse ? "on" : "off";
    j["resident_stats"] = {{"available", resident_available},
                           {"hit_blocks", resident.hit_blocks}, {"miss_blocks", resident.miss_blocks},
                           {"skipped_target_bytes", resident.skipped_target_bytes},
                           {"skipped_mtp_bytes", resident.skipped_mtp_bytes}};
    const std::pair<const char *, PoolAllocationKind> kinds[] = {
        {"raw", PoolAllocationKind::raw}, {"checkpoint", PoolAllocationKind::checkpoint},
        {"query", PoolAllocationKind::query}, {"prompt_estimated", PoolAllocationKind::prompt_estimate},
        {"metadata", PoolAllocationKind::metadata},
    };
    for (const auto & kind : kinds) {
        const auto n = memory.totals(&kind.second);
        const std::string name = kind.first;
        j[name + "_logical_bytes"] = n.logical;
        j[name + "_pool_unique_bytes"] = n.pool;
        j[name + "_live_unique_bytes"] = n.live;
        j[name + "_pool_live_unique_bytes"] = n.combined;
    }
    std::lock_guard<std::mutex> lk(st.pool_status_mu);
    st.pool_status = j.dump();
}

static void pool_evict(ServerState & st, PoolMemory & memory) {
    const auto now = std::chrono::steady_clock::now();
    const size_t budget = (size_t) (st.pool_gb * 1073741824.0);
    auto drop = [&](size_t i, const char * reason) {
        const auto & e = *st.pool[i];
        const auto bytes = memory.entry(e.id);
        LOG_INF("srv    kvmem pool: evict id=%llu rows=%d logical_bytes=%zu reclaimable_bytes=%zu hits=%u reason=%s\n",
                (unsigned long long) e.id, e.rows, bytes.logical, bytes.reclaimable, e.hits, reason);
        memory.erase(e.id);
        st.pool.erase(st.pool.begin() + (std::ptrdiff_t) i);
    };
    if (st.pool_ttl_min > 0) {
        for (size_t i = st.pool.size(); i-- > 0;) {
            if (now - st.pool[i]->last_used > std::chrono::minutes(st.pool_ttl_min)) drop(i, "ttl");
        }
    }
    while (!st.pool.empty()) {
        const bool count_limit = (int) st.pool.size() > st.pool_max;
        if (!count_limit && memory.totals().combined <= budget) break;
        size_t victim = st.pool.size(), shared_victim = st.pool.size();
        for (size_t i = 0; i < st.pool.size(); ++i) {
            const auto bytes = memory.entry(st.pool[i]->id);
            if (count_limit || bytes.reclaimable) {
                if (victim == st.pool.size() || st.pool[i]->last_used < st.pool[victim]->last_used) victim = i;
            } else if (bytes.pool_only &&
                    (shared_victim == st.pool.size() || st.pool[i]->last_used < st.pool[shared_victim]->last_used)) shared_victim = i;
        }
        // A pool-only shared allocation may require dropping multiple owners.
        // Payload held by live is never credited as released or used as a reason
        // to repeatedly evict zero-gain entries. Count/TTL eviction is independent.
        if (victim == st.pool.size()) victim = shared_victim;
        if (victim == st.pool.size()) break;
        drop(victim, count_limit ? "lru" : "unique_bytes");
    }
    const auto total = memory.totals();
    if (total.combined > budget) {
        LOG_WRN("srv    kvmem pool: budget unsatisfied unique_bytes=%zu live_unique_bytes=%zu budget_bytes=%zu; live retained\n",
                total.combined, total.live, budget);
    }
}

static void pool_maintain(ServerState & st) {
    try {
        auto memory = pool_memory(st);
        const size_t budget = (size_t) (st.pool_gb * 1073741824.0);
        if (st.pool_gpu_reuse && memory.totals().combined > budget) {
            // Tags are optional live owners. Rebuild after releasing them;
            // the previous snapshot would incorrectly pin their payloads.
            llama_kvmem_drop_resident_tags();
            memory = pool_memory(st);
        }
        if (pool_active(st)) pool_evict(st, memory);
        pool_publish(st, memory);
    } catch (const std::exception & e) {
        // Never report a partial/failed snapshot as zero usage or evict from it.
        LOG_ERR("srv    kvmem pool: accounting failed: %s\n", e.what());
        std::lock_guard<std::mutex> lk(st.pool_status_mu);
        st.pool_status = "{\"accounting_valid\":false,\"error\":\"allocation snapshot unavailable\"}";
    }
}

struct PoolPreserveScope {
    bool armed;
    explicit PoolPreserveScope(bool enabled) : armed(enabled && llama_kvmem_pool_preserve_begin()) {}
    PoolPreserveScope(const PoolPreserveScope &) = delete;
    PoolPreserveScope & operator=(const PoolPreserveScope &) = delete;
    ~PoolPreserveScope() { finish(false); }
    void finish(bool success) {
        if (!armed) return;
        llama_kvmem_pool_preserve_end(success);
        armed = false;
    }
};

// Move the live conversation into the pool. The live memory is left empty
// (llama_memory_clear'ed by memory_clear_all). Returns the new entry or null.
static PoolEntry * pool_stash_live(ServerState & st, const char * reason) {
    const auto started = std::chrono::steady_clock::now();
    if (!st.mm_committed || !st.cached_prompt || st.cached_prompt->has_media() || st.mm_live_row <= 0) return nullptr;
    // Allocate the owner before take so allocation failure cannot leak a detached stash.
    auto e = std::make_unique<PoolEntry>();
    uint32_t rows = 0;
    llama_synchronize(st.ctx);
    if (st.spec.ctx_dft) llama_synchronize(st.spec.ctx_dft);
    PoolPreserveScope preserve(st.pool_gpu_reuse);
    e->stash = llama_kvmem_stash_take((uint32_t) st.mm_live_row, &rows);
    if (!e->stash) {
        LOG_WRN("srv    kvmem pool: stash failed live_rows=%d reason=%s\n", st.mm_live_row, reason);
        preserve.finish(false);
        memory_clear_all(st);
        return nullptr;
    }
    if (!rows || rows > (uint32_t) st.mm_live_row || rows > st.cached_prompt->tokens.size()) {
        preserve.finish(false);
        memory_clear_all(st);
        return nullptr;
    }
    e->id = st.pool_next_id++;
    e->rows = (int) rows;
    e->prompt = std::make_shared<kvmem_prompt>(std::vector<llama_token>(
        st.cached_prompt->tokens.begin(), st.cached_prompt->tokens.begin() + rows));
    e->generation = st.mm_cache_generation;
    if (st.pool_query_enabled && st.mm_query &&
            !multimodal_query_invalid(st, *st.mm_query, *e->prompt, e->rows)) e->query = st.mm_query;
    std::vector<MultimodalCheckpoint> kept;
    for (const auto & c : st.mm_checkpoints) {
        if (c.data && c.row > 0 && c.row <= e->rows) kept.push_back(c);
    }
    // A saved Q is useful only if a retained recurrent state can resume after
    // its complete source prefix. Prefer Q.end, otherwise the earliest later row.
    int query_anchor = -1;
    if (e->query) {
        for (const auto & c : kept) {
            if (c.row >= e->query->end && (query_anchor < 0 || c.row < query_anchor)) query_anchor = c.row;
        }
        if (query_anchor < 0) e->query.reset();
    }
    // A later turn of this conversation resumes at a message start; the rows
    // inside the rewritten tail (query, eval_end, commit) are rarely reused.
    // Drop those first, oldest first; protect the oldest entry and the Q anchor.
    auto msg_start = [&](const MultimodalCheckpoint & c) {
        return c.row < (int) st.cached_prompt->tokens.size() && st.cached_prompt->tokens[c.row] == st.mm_msg_token;
    };
    while ((int) kept.size() > st.pool_ckpts) {
        auto victim = std::find_if(kept.begin() + 1, kept.end(),
            [&](const auto & c) { return c.row != query_anchor && !msg_start(c); });
        if (victim == kept.end()) victim = std::find_if(kept.begin() + 1, kept.end(),
            [&](const auto & c) { return c.row != query_anchor; });
        if (victim == kept.end()) break; // At most two protected entries; CLI requires cap >= 2.
        kept.erase(victim);
    }
    e->checkpoints = std::move(kept);
    e->bytes = llama_kvmem_stash_bytes(e->stash);
    e->created = e->last_used = std::chrono::steady_clock::now();
    memory_clear_all(st);
    LOG_INF("srv    kvmem pool: stash id=%llu rows=%d ckpts=%zu query=%d query_anchor=%d legacy_stash_bytes=%.2fGB ms=%.1f reason=%s entries=%zu\n",
            (unsigned long long) e->id, e->rows, e->checkpoints.size(), (int) (e->query != nullptr), query_anchor, e->bytes / 1073741824.0,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
            reason, st.pool.size() + 1);
    st.pool.push_back(std::move(e));
    preserve.finish(true);
    return st.pool.back().get();
}

// Make rows [0, rows) of a parked conversation live. take moves the entry out
// of the pool; otherwise its prefix is copied and the entry stays parked.
static bool pool_restore(ServerState & st, size_t index, int rows, bool take, int lcp) {
    const auto started = std::chrono::steady_clock::now();
    if (index >= st.pool.size()) return false;
    PoolEntry & e = *st.pool[index];
    if (!e.stash || !e.prompt || e.prompt->has_media() || rows <= 0 || rows > e.rows || rows > lcp ||
            (size_t) rows > e.prompt->tokens.size() || !st.active_prompt ||
            st.active_prompt->common_prefix(*e.prompt) < (size_t) rows ||
            std::none_of(e.checkpoints.begin(), e.checkpoints.end(),
                [&](const auto & c) { return c.data && c.row == rows; })) {
        LOG_WRN("srv    kvmem pool: invalid resume id=%llu rows=%d stored=%d lcp=%d\n",
                (unsigned long long) e.id, rows, e.rows, lcp);
        return false;
    }
    const uint64_t id = e.id;
    const int stored_rows = e.rows;
    // Prepare every allocating metadata copy before consuming the stash.
    auto prompt = std::make_shared<kvmem_prompt>(std::vector<llama_token>(
        e.prompt->tokens.begin(), e.prompt->tokens.begin() + rows));
    auto tokens = prompt->tokens;
    std::vector<MultimodalCheckpoint> checkpoints;
    for (const auto & c : e.checkpoints) {
        if (c.data && c.row >= 0 && c.row <= rows) checkpoints.push_back(c);
    }
    std::shared_ptr<const MultimodalQuery> query;
    if (st.pool_query_enabled && e.generation == st.mm_cache_generation && e.query &&
            !multimodal_query_invalid(st, *e.query, *prompt, rows)) query = e.query;
    bool ok;
    if (take) {
        auto * stash = e.stash;
        e.stash = nullptr;
        // Both put exports consume stash, including failure. Remove the route
        // before entering either export so a failed take cannot be retried/freed.
        st.pool.erase(st.pool.begin() + (std::ptrdiff_t) index);
        ok = st.pool_restore_prefix ? llama_kvmem_stash_put_prefix(stash, (uint32_t) rows)
                                    : llama_kvmem_stash_put(stash);
    } else {
        ok = llama_kvmem_stash_fork(e.stash, (uint32_t) rows);
        e.last_used = std::chrono::steady_clock::now();
        e.hits++;
    }
    if (!ok) {
        LOG_WRN("srv    kvmem pool: %s failed id=%llu rows=%d\n", take ? "take" : "fork", (unsigned long long) id, rows);
        memory_clear_all(st);
        return false;
    }
    // The full mode is the comparison path: restage all, then truncate. The
    // prefix export and fork already truncate on the host before restaging.
    if (take && !st.pool_restore_prefix) llama_kvmem_truncate_cached((uint32_t) rows);
    const auto restored_rows = llama_kvmem_store_n_tokens();
    if (restored_rows != (uint32_t) rows) {
        LOG_WRN("srv    kvmem pool: restore row mismatch id=%llu wanted=%d actual=%u\n",
                (unsigned long long) id, rows, (unsigned) restored_rows);
        memory_clear_all(st);
        return false;
    }
    st.cached_prompt = std::move(prompt);
    st.cached_tokens = std::move(tokens);
    st.mm_checkpoints = std::move(checkpoints);
    st.mm_live_row = rows;
    st.mm_live_checkpoint.reset();
    st.mm_query = std::move(query);
    st.mm_pending_query.reset();
    LOG_INF("srv    kvmem pool: %s id=%llu rows=%d resume=%d lcp=%d stored_rows=%d query=%d restore=%s ms=%.1f entries=%zu\n",
            take ? "take" : "fork", (unsigned long long) id, rows, rows, lcp, stored_rows,
            (int) (st.mm_query != nullptr), st.pool_restore_prefix ? "prefix" : "full",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
            st.pool.size());
    return true;
}

// Called once per request before the live checkpoint is chosen.
static void multimodal_pool_route(ServerState & st, const kvmem_prompt & prompt, int eval_end) {
    if (st.pool_reset_requested) {
        multimodal_invalidate_queries(st);
        llama_kvmem_drop_resident_tags();
        if (!st.pool.empty()) LOG_INF("srv    kvmem pool: reset entries=%zu\n", st.pool.size());
        st.pool.clear();
        st.pool_reset_requested = false;
    }
    if (!pool_active(st) || prompt.has_media()) {
        pool_maintain(st);
        return;
    }
    try {
        const int cap = eval_end - (st.spec.ok ? 0 : 1);
        int live_lcp = 0;
        const int live_row = pool_resume_row(prompt, cap, st.cached_prompt.get(), st.mm_live_row, st.mm_checkpoints, &live_lcp);
        size_t best = st.pool.size();
        int best_row = -1, best_lcp = 0;
        for (size_t i = 0; i < st.pool.size(); ++i) {
            int lcp = 0;
            const int row = pool_resume_row(prompt, cap, st.pool[i]->prompt.get(), st.pool[i]->rows, st.pool[i]->checkpoints, &lcp);
            if (row > best_row) {
                best = i;
                best_row = row;
                best_lcp = lcp;
            }
        }
        const bool live_valuable = st.cached_prompt && !st.cached_prompt->has_media() &&
            pool_branch_worth(st, *st.cached_prompt, st.mm_live_row, st.mm_checkpoints, live_lcp, live_row < 0);
        if (best < st.pool.size() && best_row >= std::max(live_row, 0) + st.pool_min_gain) {
            const PoolEntry * target = st.pool[best].get();
            if (live_valuable && pool_stash_live(st, "switch")) {
                best = (size_t) (std::find_if(st.pool.begin(), st.pool.end(), [&](const auto & e) { return e.get() == target; }) - st.pool.begin());
            } else {
                memory_clear_all(st);
            }
            const PoolEntry & e = *st.pool[best];
            const bool fork = pool_branch_worth(st, *e.prompt, e.rows, e.checkpoints, best_lcp, false);
            pool_restore(st, best, best_row, !fork, best_lcp);
        } else if (live_valuable) {
            // The request abandons the live branch: park it, keep only the shared prefix live.
            PoolEntry * parked = pool_stash_live(st, live_row < 0 ? "miss" : "branch");
            if (parked && live_row > 0) {
                const size_t index = (size_t) (std::find_if(st.pool.begin(), st.pool.end(), [&](const auto & e) { return e.get() == parked; }) - st.pool.begin());
                pool_restore(st, index, live_row, false, live_lcp);
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("srv    kvmem pool: route failed: %s\n", e.what());
        memory_clear_all(st);
    }
    pool_maintain(st);
}

static bool run_prefill_multimodal(ServerState & st, StreamIo * io, int * n_cache_hit) {
    const auto started = std::chrono::steady_clock::now();
    st.mm_perf = {};
    const auto copies_before = llama_kvmem_get_transfer_stats();
    st.mm_checkpoint_accounting->peak_bytes = st.mm_checkpoint_accounting->live_bytes;
    st.mm_pending_query.reset();
    try {
        st.mm_error.clear();
        st.mm_error_status = 500;
        // cache_reset clears the live conversation and bypasses the pool for this
        // request, so it always measures a cold prefill.
        const bool cold = st.mm_reset_requested;
        if (st.mm_reset_requested) {
            kvmem_diag("KVMEM_TRACE multimodal_reset context=%p reason=explicit_cache_reset\n", (void *) st.ctx);
            multimodal_invalidate_queries(st);
            memory_clear_all(st);
            st.mm_reset_requested = false;
        }
        st.mm_new_text = st.mm_new_image = st.mm_replayed = st.mm_tail_replayed = 0;
        if (st.vision) st.vision->reset_stats();
        const auto & prompt = *st.active_prompt;
        const int eval_end = (int) prompt.tokens.size() - (st.spec.ok ? 1 : 0);
        if (!cold) multimodal_pool_route(st, prompt, eval_end);
        else if (st.pool_reset_requested) {
            llama_kvmem_drop_resident_tags();
            st.pool.clear();
            st.pool_reset_requested = false;
            pool_maintain(st);
        }
        const int lcp = st.cached_prompt ? (int) prompt.common_prefix(*st.cached_prompt) : 0;
        st.mm_lcp = lcp;
        // Sequence checkpoints do not restore logits. Ordinary decoding must evaluate
        // at least one token; MTP evaluates the pending prompt token in spec_generate.
        const int keep = std::min({lcp, st.mm_live_row, eval_end - (st.spec.ok ? 0 : 1)});
        MultimodalCheckpoint base;
        bool found = false;
        {
            kvmem_scoped_ms timer(st.mm_perf.select_checkpoint_ms);
            for (const auto & checkpoint : st.mm_checkpoints) {
                if (checkpoint.row <= keep && (!found || checkpoint.row > base.row)) {
                    base = checkpoint;
                    found = true;
                }
            }
        }
        if (!found) {
            // Shared template tokens do not identify a conversation. Like llama-server,
            // treat a missing recurrent checkpoint as a cache miss and evaluate the supplied prompt.
            kvmem_diag("KVMEM_TRACE multimodal_reset context=%p reason=%s lcp=%d keep=%d cached_rows=%zu live_rows=%d oldest_checkpoint=%d checkpoint_count=%zu\n",
                    (void *) st.ctx, st.cached_prompt ? "no_recurrent_checkpoint" : "new_conversation",
                    lcp, keep, st.cached_tokens.size(), st.mm_live_row,
                    st.mm_checkpoints.empty() ? -1 : st.mm_checkpoints.front().row, st.mm_checkpoints.size());
            memory_clear_all(st);
            st.mm_lcp = 0; // No old rows survived the reset; count all evaluated rows as new.
            base = multimodal_checkpoint(st, 0);
        } else {
            // Checkpoints past the common prefix were computed on the abandoned branch.
            const size_t before = st.mm_checkpoints.size();
            st.mm_checkpoints.erase(std::remove_if(st.mm_checkpoints.begin(), st.mm_checkpoints.end(),
                    [&](const auto & checkpoint) { return checkpoint.row > lcp; }), st.mm_checkpoints.end());
            kvmem_diag("KVMEM_TRACE prefix_checkpoint_reuse base=%d lcp=%d keep=%d catchup=%d dropped=%zu kept=%zu\n",
                    base.row, lcp, keep, keep - base.row, before - st.mm_checkpoints.size(), st.mm_checkpoints.size());
        }
        st.mm_rollback = std::make_shared<MultimodalCheckpoint>(base);
        st.mm_rollback_prompt = st.cached_prompt ? st.cached_prompt->prefix(base.row) : nullptr;
        st.mm_committed = false;
        multimodal_restore(st, base, true);
        llama_kvmem_begin_cached_turn();
        std::vector<uint32_t> starts, ends;
        const auto ranges = prompt.media_ranges();
        for (const auto & range : ranges) {
            starts.push_back(range.first > 0 ? range.first - 1 : 0);
            ends.push_back(std::min<uint32_t>(prompt.tokens.size(), range.second + 1));
        }
        llama_kvmem_set_media_ranges(starts.data(), ends.data(), starts.size());
        const int user_begin = std::clamp(st.kparams.query_begin, 0, eval_end);
        const int user_end = std::clamp(st.kparams.query_end, user_begin, eval_end);
        const auto media = prompt.media_identity();
        const auto cached_query = st.mm_query;
        const char * query_invalid = cached_query ? multimodal_query_invalid(st, *cached_query, prompt, base.row) : nullptr;
        if (cached_query && !query_invalid &&
                (cached_query->begin != user_begin || cached_query->end != user_end ||
                 cached_query->force != st.kparams.force_pos || cached_query->user != st.turn_last_user))
            query_invalid = "query_identity";
        const bool same_query = st.query_policy_user && st.turn_query_exact && cached_query &&
            !query_invalid && base.row >= user_end;
        const bool capture_user = st.query_policy_user && st.turn_query_exact &&
            user_begin >= base.row && user_end > user_begin;
        int query = std::max(base.row, std::min(st.kparams.query_begin, eval_end));
        if (!ranges.empty()) query = std::max(query, (int) ranges.back().second);
        query = std::min(query, eval_end);
        const bool retrieve = st.kparams.enabled && st.kparams.method == 1 && query < eval_end;
        llama_kvmem_set_request_span(query, eval_end, st.kparams.force_pos);
        llama_kvmem_turn_spans spans;
        spans.query = {{query, eval_end}};
        spans.mandatory = {{query, eval_end}};
        spans.replay_begin = query;
        if (st.query_policy_user) {
            spans.query = capture_user || same_query
                ? std::vector<llama_kvmem_row_range>{{user_begin, user_end}}
                : std::vector<llama_kvmem_row_range>{{std::max(query, eval_end - st.query_max_tokens), eval_end}};
        }
        llama_kvmem_set_turn_spans(spans);
        std::string path = "legacy", reason = "legacy_requested";
        std::string reuse_fallback = query_invalid ? query_invalid : "none";
        if (cached_query && !same_query) {
            st.mm_query.reset();
            llama_kvmem_reset_query();
            llama_kvmem_freeze_query(false);
        }
        bool reused_query = false;
        const bool imported_query = same_query && st.kparams.enabled && st.kparams.method == 1 &&
            llama_kvmem_set_query(cached_query->state);
        if (imported_query) {
            kvmem_scoped_ms timer(st.mm_perf.decision_ms);
            if (llama_kvmem_can_append(eval_end, st.turn_generation_rows, false, reason)) {
                path = "keep_selected";
                reused_query = true;
            } else {
                auto select_spans = spans;
                // Q predates these tool observations. Do not let stale query
                // scores discard newly learned facts during a fast reselect.
                // If the observed tail no longer fits the selection budget,
                // use a fresh suffix probe instead. The old Q source itself
                // is not a replay dependency and need not consume mandatory slots.
                select_spans.mandatory = {{user_end, base.row}};
                const uint32_t block = st.kparams.block_tokens ? st.kparams.block_tokens : 32;
                if (base.row % block) select_spans.mandatory.push_back({base.row - 1, base.row});
                llama_kvmem_set_turn_spans(select_spans);
                const auto selection = llama_kvmem_preview_retrieval();
                if (llama_kvmem_selection_fits(selection, eval_end, st.turn_generation_rows)) {
                    {
                        kvmem_scoped_ms retrieval_timer(st.mm_perf.retrieval_ms);
                        llama_kvmem_apply_selection(selection);
                    }
                    llama_kvmem_begin_cached_turn_keep_query();
                    llama_kvmem_freeze_query(true);
                    reused_query = llama_kvmem_can_append(eval_end, st.turn_generation_rows, false, reason);
                    if (reused_query) path = "cached_q_reselect";
                } else reason = "observed_tail_or_append_exceeds_capacity";
            }
            if (reused_query) {
                st.mm_pending_query = cached_query;
                llama_kvmem_keep_selected();
            }
        }
        if (same_query && !reused_query) {
            reuse_fallback = imported_query ? reason : "invalid_query_state";
            st.mm_query.reset();
            llama_kvmem_reset_query();
            llama_kvmem_freeze_query(false);
            spans.query = {{std::max(query, eval_end - st.query_max_tokens), eval_end}};
            llama_kvmem_set_turn_spans(spans);
        }
        bool all_resident = false;
        {
            kvmem_scoped_ms timer(st.mm_perf.decision_ms);
            all_resident = !reused_query && retrieve && st.query_replay_auto &&
                llama_kvmem_can_append(eval_end, st.turn_generation_rows, true, reason);
        }
        if (all_resident) llama_kvmem_keep_selected();
        if (st.query_policy_user && retrieve && !reused_query && !all_resident) {
            multimodal_validate_capacity(st, prompt, query, eval_end);
        }
        if (n_cache_hit) *n_cache_hit = base.row;
        const auto checkpoint_rows = multimodal_plan_checkpoints(st, prompt, base.row, query);
        if (multimodal_decode_span(st, base.row, query, false, io, &checkpoint_rows) != 0) throw std::runtime_error("multimodal prefill failed or cancelled");
        auto query_checkpoint = multimodal_checkpoint(st, query);
        multimodal_remember(st, query_checkpoint);
        const auto probe_view = llama_kvmem_get_attention_view();
        if (multimodal_decode_span(st, query, eval_end, false, io) != 0) throw std::runtime_error("multimodal query prefill failed or cancelled");
        if (reused_query) {
            if (!llama_kvmem_commit_resident(false)) throw std::runtime_error("incomplete KV after query continuation");
        } else if (retrieve) {
            bool replay = true;
            {
                kvmem_scoped_ms timer(st.mm_perf.retrieval_ms);
                if (all_resident && llama_kvmem_commit_resident()) {
                    path = "all_resident";
                    replay = false;
                } else {
                    const auto selection = llama_kvmem_preview_retrieval();
                    if (st.query_replay_auto && llama_kvmem_commit_unchanged(probe_view, selection)) {
                        path = "unchanged_selection";
                        reason = "same_attention_view";
                        replay = false;
                    } else {
                        path = "query_replay";
                        if (st.query_replay_auto) reason = "selection_or_attention_view_changed";
                        llama_kvmem_apply_selection(selection);
                    }
                }
            }
            if (replay && !prompt.has_media() && !llama_kvmem_query_replay_fits(query, eval_end)) {
                // Selection may trim the mandatory suffix when a long tool history
                // exceeds the retrieval budget. Keep the completed first-pass
                // recurrent state, logits and MTP carry; restoring the query
                // checkpoint would require replaying rows with no resident slot.
                path = "query_replay_skipped";
                reason = "replay_exceeds_budget";
                replay = false;
                kvmem_diag("KVMEM_TRACE replay_skipped reason=over_budget query=[%d,%d) replay_rows=%d budget_tokens=%u\n",
                        query, eval_end, eval_end - query, st.kparams.budget);
            }
            if (replay) {
                multimodal_restore(st, query_checkpoint, false);
                llama_kvmem_set_replay(true);
                const int rc = multimodal_decode_span(st, query, eval_end, true, io);
                llama_kvmem_set_replay(false);
                if (rc != 0) throw std::runtime_error("multimodal query replay failed or cancelled");
            }
        }
        if (!reused_query && capture_user && st.kparams.enabled && st.kparams.method == 1) {
            auto saved = std::make_shared<MultimodalQuery>();
            if (llama_kvmem_get_query(saved->state) &&
                    !saved->state.count.empty() &&
                    *std::max_element(saved->state.count.begin(), saved->state.count.end()) == (uint32_t) (user_end - user_begin)) {
                saved->begin = user_begin;
                saved->end = user_end;
                saved->force = st.kparams.force_pos;
                saved->generation = st.mm_cache_generation;
                saved->model = st.model;
                saved->user = st.turn_last_user;
                saved->media = media;
                saved->prefix.assign(prompt.tokens.begin(), prompt.tokens.begin() + user_end);
                if (!multimodal_query_invalid(st, *saved, prompt, st.mm_live_row))
                    st.mm_pending_query = std::move(saved);
            }
        }
        const auto & counts = st.mm_pending_query ? st.mm_pending_query->state.count : std::vector<uint32_t>{};
        const uint32_t q_rows = counts.empty() ? 0 : *std::max_element(counts.begin(), counts.end());
        const char * source = !st.query_policy_user ? "legacy_suffix" : reused_query ? "cached_user" :
            capture_user ? "user" : "bootstrap_suffix";
        if (!retrieve && !reused_query) reason = "no_query_suffix";
        kvmem_diag("KVMEM_PREFILL_DECISION path=%s reason=%s reuse_fallback=%s append=[%d,%d) query=[%d,%d) feature=[%d,%d) query_source=%s query_reused=%d q_rows=%u replay_rows=%u decision_ms=%.3f\n",
                path.c_str(), reason.c_str(), reuse_fallback.c_str(), base.row, eval_end, query, eval_end,
                spans.query.front().begin, spans.query.back().end, source, reused_query, q_rows, st.mm_replayed, st.mm_perf.decision_ms);
        llama_kvmem_pin_working_set();
        multimodal_remember(st, multimodal_checkpoint(st, eval_end));
        std::vector<uint8_t> carry;
        llama_pos synced = 0;
        if (st.spec.ok) {
            common_speculative_get_state(st.spec.spec, 0, carry);
            if (carry.size() < sizeof(synced)) throw std::runtime_error("MTP carry missing");
            std::memcpy(&synced, carry.data(), sizeof(synced));
            if (synced != eval_end) throw std::runtime_error("MTP has unsynchronized visual rows");
        }
        kvmem_diag("KVMEM_TRACE multimodal_prefill context=%p prefix_hit_rows=%d lcp=%d new_text_rows=%u new_image_rows=%u replayed_rows=%u vision_encode_calls=%u encoder_ms=%.2f logical_cursor=%d model_cursor=%d mtp_synced_rows=%d cached_tail_rows=%u replay_reason=%s embedding_cache_bytes=%zu checkpoint_bytes=%zu\n",
                (void *) st.ctx, base.row, lcp, st.mm_new_text, st.mm_new_image, st.mm_replayed,
                st.vision ? st.vision->encode_calls : 0, st.vision ? st.vision->encode_ms : 0.0,
                eval_end, prompt.model_pos(eval_end), synced, st.mm_tail_replayed, st.mm_replayed ? reason.c_str() : "none",
                st.vision ? st.vision->cache_bytes() : 0,
                std::accumulate(st.mm_checkpoints.begin(), st.mm_checkpoints.end(), size_t(0),
                    [](size_t n, const auto & c) { return n + c.data->bytes(); }));
        std::set<const MultimodalCheckpointData *> unique;
        size_t unique_bytes = 0;
        size_t ref_bytes = 0;
        auto count = [&](const MultimodalCheckpoint & cp) {
            if (cp.data && unique.insert(cp.data.get()).second) {
                unique_bytes += cp.data->bytes();
                ref_bytes += cp.data->bytes() * cp.data.use_count();
            }
        };
        for (const auto & cp : st.mm_checkpoints) count(cp);
        if (st.mm_rollback) count(*st.mm_rollback);
        count(base);
        count(query_checkpoint);
        const auto & p = st.mm_perf;
        kvmem_diag("KVMEM_PREFILL_PERF total_ms=%.3f first_ms=%.3f replay_ms=%.3f retrieval_ms=%.3f checkpoint_select_ms=%.3f checkpoint_save_ms=%.3f checkpoint_restore_ms=%.3f mean_nested_ms=%.3f carry_nested_ms=%.3f saves=%u restores=%u shared=%u restore_skips=%u checkpoint_unique_bytes=%zu\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count(),
                p.first_ms, p.replay_ms, p.retrieval_ms, p.select_checkpoint_ms, p.save_ms, p.restore_ms,
                p.mean_ms, p.carry_ms, p.saves, p.restores, p.shared, p.restore_skips, unique_bytes);
        kvmem_diag("KVMEM_CHECKPOINT_MEMORY ref_bytes=%zu unique_bytes=%zu live_bytes=%zu peak_bytes=%zu\n",
                ref_bytes, unique_bytes, st.mm_checkpoint_accounting->live_bytes, st.mm_checkpoint_accounting->peak_bytes);
        if (copies_before.enabled) {
            const auto copies = llama_kvmem_get_transfer_stats();
            fprintf(stderr, "KVMEM_PREFILL_DIAGNOSTIC target_ms=%.3f draft_ms=%.3f extra_sync=1 adapter_h2d_bytes=%llu adapter_d2h_bytes=%llu adapter_d2d_bytes=%llu h2d_calls=%llu d2h_calls=%llu d2d_calls=%llu\n",
                    p.target_ms, p.draft_ms,
                    (unsigned long long) (copies.bytes[0] - copies_before.bytes[0]),
                    (unsigned long long) (copies.bytes[1] - copies_before.bytes[1]),
                    (unsigned long long) (copies.bytes[2] - copies_before.bytes[2]),
                    (unsigned long long) (copies.calls[0] - copies_before.calls[0]),
                    (unsigned long long) (copies.calls[1] - copies_before.calls[1]),
                    (unsigned long long) (copies.calls[2] - copies_before.calls[2]));
        }
        return true;
    } catch (const std::exception & e) {
        st.mm_error = e.what();
        if (dynamic_cast<const std::invalid_argument *>(&e)) st.mm_error_status = 400;
        LOG_ERR("srv    KVMEM_TRACE multimodal_error error=%s\n", e.what());
        multimodal_finish_request(st);
        return false;
    }
}

static void multimodal_commit(ServerState & st, const std::vector<llama_token> & gen) {
    st.cached_prompt = st.active_prompt->with_generated(gen);
    st.mm_live_row = st.kparams.enabled ? (int) llama_kvmem_store_n_tokens()
        : st.mm_live_row;
    multimodal_remember(st, multimodal_checkpoint(st, st.mm_live_row));
    st.mm_query = st.mm_pending_query &&
        !multimodal_query_invalid(st, *st.mm_pending_query, *st.cached_prompt, st.mm_live_row)
        ? st.mm_pending_query : nullptr;
    st.mm_pending_query.reset();
    st.mm_committed = true;
    st.mm_rollback.reset();
    st.mm_rollback_prompt.reset();
    kvmem_diag("KVMEM_CHECKPOINT_COMMIT live_bytes=%zu peak_bytes=%zu\n",
            st.mm_checkpoint_accounting->live_bytes, st.mm_checkpoint_accounting->peak_bytes);
    pool_maintain(st);
}

static int multimodal_decode_generated(ServerState & st, llama_token id, int row) {
    st.mm_live_checkpoint.reset();
    llama_batch batch = llama_batch_get_one(&id, 1);
    llama_pos logical = row;
    llama_pos pos = st.active_prompt->model_pos(row);
    batch.logical_pos = &logical;
    batch.pos = &pos;
    const int rc = llama_decode(st.ctx, batch);
    if (rc == 0) st.mm_live_row = row + 1;
    return rc;
}
