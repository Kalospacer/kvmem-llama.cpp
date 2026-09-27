#pragma once

// Server-side transparent PIC orchestration. Off by default, enabled with
// --kvmem-pic on. The client request, tokenization and positions are unchanged.

// Included after multimodal_checkpoint/restore/decode_span, before pool accounting.
#include "llama-kvmem-pic-capture.h"
#include "llama-kvmem-pic-store.h"
#include "llama-kvmem-pic-state.h"
#include "llama-kvmem-pic-kv.h"
#include "llama-memory-kvmem-hybrid.h"
#include "llama-memory-kvmem-mtp.h"
#include "llama-kvmem-quant.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml-alloc.h"

#include <cmath>
#include <limits>

static size_t pic_exact_cache_bytes(ServerState & st);

using PicStash = std::unique_ptr<llama_kvmem_stash, decltype(&llama_kvmem_stash_free)>;

struct KvmemPicRaw {
    PicStash stash {nullptr, llama_kvmem_stash_free};
    kvmem_pic::kv_segment_identity identity;
    std::vector<kvmem_pic::kv_layer_codec> codecs;
    std::vector<uint8_t> body_carry;

    static bool allocations(const void * owner, kvmem_pic::allocation_list & out) {
        const auto & raw = *static_cast<const KvmemPicRaw *>(owner);
        if (!llama_kvmem_stash_allocations(raw.stash.get(), out)) return false;
        out.emplace_back(&raw, sizeof(raw));
        if (raw.codecs.capacity()) out.emplace_back(raw.codecs.data(), raw.codecs.capacity()*sizeof(raw.codecs[0]));
        for (const auto & codec : raw.codecs) {
            if (codec.k.freq_factors.capacity()) out.emplace_back(codec.k.freq_factors.data(), codec.k.freq_factors.capacity()*sizeof(float));
        }
        if (raw.body_carry.capacity()) out.emplace_back(raw.body_carry.data(), raw.body_carry.capacity());
        return true;
    }
};

struct KvmemPicServer {
    static constexpr int seam = 8;
    llama_kvmem_pic_capture capture;
    kvmem_pic::store segments;
    kvmem_pic::segment_key key;
    kvmem_pic::kv_segment_identity identity;
    std::vector<kvmem_pic::kv_layer_codec> codecs;
    std::vector<kvmem_pic::segment_match> matches;
    ggml_backend_t cpu = nullptr, preferred = nullptr;
    uint64_t hits = 0, builds = 0, failures = 0, skipped_rows = 0;
    double build_ms = 0, splice_ms = 0;
    size_t capture_host_peak_bytes = 0, math_input_peak_bytes = 0, transaction_backup_peak_bytes = 0;
    bool ready = false, building = false, feature_fallback = false, restore_failed = false;
    int query = -1, end = -1, force = -1;
    llama_kvmem_turn_spans spans;
    std::string unsupported;

    ~KvmemPicServer() {
        if (preferred) ggml_backend_free(preferred);
        if (cpu) ggml_backend_free(cpu);
    }
};


static void pic_require(bool ok, const char * error) {
    if (!ok) throw std::runtime_error(error);
}

static void pic_check(bool ok, const std::string & error) {
    if (!ok) throw std::runtime_error(error.empty() ? "PIC operation failed" : error);
}

static void pic_sync(ServerState & st) {
    llama_synchronize(st.ctx);
    if (st.spec.ctx_dft) llama_synchronize(st.spec.ctx_dft);
}

static llama_memory_kvmem_hybrid & pic_hybrid(ServerState & st) {
    auto * hybrid = dynamic_cast<llama_memory_kvmem_hybrid *>(llama_get_memory(st.ctx));
    pic_require(hybrid && hybrid->get_mem_recr()->supports_pic_state(), "PIC requires single-sequence Qwen48 GDN memory");
    return *hybrid;
}

// Configuration is captured once, from the actual context/cache, before any entry
// is constructed. A hit uses the stored source codecs and this same destination key.
static void pic_initialize(ServerState & st, llama_context_params & params) {
    if (!st.pic_enabled) return;
    st.pic = std::make_shared<KvmemPicServer>();
    st.pic->capture.install(params);
}

static bool pic_ready(ServerState & st) {
    if (!st.pic_enabled || !st.pic || !st.kparams.enabled || !st.query_policy_user) return false;
    auto & pic = *st.pic;
    if (pic.ready) return true;
    if (!pic.unsupported.empty()) return false;
    try {
        auto & hybrid = pic_hybrid(st);
        auto * target = hybrid.attn_kvmem();
        pic_require(target && !target->raw().nvme_enabled() && !target->v_trans(), "PIC requires non-transposed RAM KV");
        pic_require(st.model->arch == LLM_ARCH_QWEN35 || st.model->arch == LLM_ARCH_QWEN35MOE,
                    "PIC graph capture is implemented for Qwen3.5 only");
        pic_require(!st.spec.ok || (target->mtp_follower() && st.spec.ctx_dft), "PIC missing original MTP KV follower");
        auto add_codecs = [&](llama_context * ctx, llama_kv_cache * cache, bool mtp) {
            const auto & model = *llama_get_model(ctx);
            const auto & hp = model.hparams;
            const auto & cp = ctx->get_cparams();
            auto layers = cache->get_layer_ids();
            std::sort(layers.begin(), layers.end());
            pic_require(!mtp || layers.size() == 1, "PIC requires one MTP layer");
            for (const auto id : layers) {
                kvmem_pic::kv_layer_codec codec;
                codec.mtp = mtp;
                codec.raw_layer = mtp ? 0 : id;
                codec.graph_layer = id;
                auto & k = codec.k;
                k.type = cache->type_k();
                k.head_dim = hp.n_embd_head_k(id);
                k.heads = hp.n_head_kv(id);
                k.n_rot = hp.n_rot(id);
                k.mode = hp.rope_type;
                k.sections = hp.rope_sections;
                k.n_ctx_orig = cp.n_ctx_orig_yarn;
                k.freq_base = cp.rope_freq_base;
                k.freq_scale = cp.rope_freq_scale;
                k.ext_factor = cp.yarn_ext_factor;
                k.attn_factor = cp.yarn_attn_factor;
                k.beta_fast = cp.yarn_beta_fast;
                k.beta_slow = cp.yarn_beta_slow;
                pic_require(!model.get_rope_factors(cp, id), "PIC does not support per-frequency factors");
                k.hadamard_nrot = kvmem_attn_rot_on(k.type, k.head_dim) ? kvmem_hadamard_nrot_k(k.head_dim) : 0;
                codec.v_type = cache->type_v();
                codec.v_elements = hp.n_embd_v_gqa(id);
                codec.v_hadamard_nrot = kvmem_attn_rot_on(codec.v_type, hp.n_embd_head_v(id))
                    ? kvmem_hadamard_nrot_v(hp.n_embd_head_v(id)) : 0;
                std::vector<uint8_t> probe;
                std::string reason;
                pic_check(kvmem_pic::relocate_packed_k(k, {}, {}, nullptr, 0, probe, reason) == kvmem_pic::rope_status::ok, reason);
                pic.codecs.push_back(std::move(codec));
            }
        };
        add_codecs(st.ctx, target->get_kv(), false);
        if (st.spec.ok) add_codecs(st.spec.ctx_dft, target->mtp_follower()->get_kv(), true);
        pic.cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        pic_require(pic.cpu != nullptr, "PIC CPU math backend unavailable");
        pic.preferred = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        pic.identity = {st.model, st.spec.ok ? llama_get_model(st.spec.ctx_dft) : nullptr, st.mm_cache_generation, true};
        pic.key.model = st.model_name + ":" + std::to_string(reinterpret_cast<uintptr_t>(st.model));
        pic.key.configuration = "qwen48-f32-independent-body-seam8-v1:" + std::to_string(st.cache_type_k) + ":" +
            std::to_string(st.cache_type_v) + ":" + std::to_string(st.spec_cache_type) + ":" + std::to_string(st.kparams.block_tokens);
        pic.key.model_generation = st.mm_cache_generation;
        pic.key.configuration_generation = st.mm_cache_generation;
        pic.key.origin = kvmem_pic::provenance::approximate;
        pic.key.approximation = "independent-zero-origin:GDN-affine+FA-relocation:body-only:seam8:v1";
        pic.ready = true;
        return true;
    } catch (const std::exception & e) {
        pic.unsupported = e.what();
        LOG_WRN("srv    PIC unavailable reason=%s\n", e.what());
        return false;
    }
}

// Captures every server field modified by memory_clear_all. Active request,
// rollback checkpoint and pool remain owned by the caller throughout scratch work.
struct PicServerMetadata {
    std::vector<llama_token> tokens;
    std::shared_ptr<kvmem_prompt> prompt;
    std::vector<MultimodalCheckpoint> checkpoints;
    int row, n_batch;
    std::shared_ptr<const MultimodalCheckpointData> live_checkpoint;
    std::shared_ptr<const MultimodalQuery> query, pending_query;
    bool approximate;
    std::vector<uint8_t> gdn, gdn_query, carry, query_carry;
    int gdn_pos, gdn_query_pos, query_begin, query_end, n_gen;
    std::string user;

    explicit PicServerMetadata(const ServerState & st) : tokens(st.cached_tokens), prompt(st.cached_prompt),
        checkpoints(st.mm_checkpoints), row(st.mm_live_row), n_batch(st.n_batch), live_checkpoint(st.mm_live_checkpoint),
        query(st.mm_query), pending_query(st.mm_pending_query), approximate(st.mm_approximate),
        gdn(st.gdn_ckpt), gdn_query(st.gdn_ckpt_query), carry(st.gdn_carry), query_carry(st.gdn_query_carry),
        gdn_pos(st.gdn_ckpt_pos), gdn_query_pos(st.gdn_ckpt_query_pos), query_begin(st.last_query_begin),
        query_end(st.last_query_end), n_gen(st.last_n_gen), user(st.last_user_text) {}

    void restore(ServerState & st) const {
        st.cached_tokens = tokens;
        st.cached_prompt = prompt;
        st.mm_checkpoints = checkpoints;
        st.mm_live_row = row;
        st.n_batch = n_batch;
        st.mm_live_checkpoint = live_checkpoint;
        st.mm_query = query;
        st.mm_pending_query = pending_query;
        st.mm_approximate = approximate;
        st.gdn_ckpt = gdn;
        st.gdn_ckpt_query = gdn_query;
        st.gdn_carry = carry;
        st.gdn_query_carry = query_carry;
        st.gdn_ckpt_pos = gdn_pos;
        st.gdn_ckpt_query_pos = gdn_query_pos;
        st.last_query_begin = query_begin;
        st.last_query_end = query_end;
        st.last_n_gen = n_gen;
        st.last_user_text = user;
    }
};

struct PicLiveTransaction {
    ServerState & st;
    PicServerMetadata metadata;
    MultimodalCheckpoint checkpoint;
    PicStash stash {nullptr, llama_kvmem_stash_free};
    llama_kvmem_query_state accumulator;
    bool have_query = false, detached = false, finished = false;

    explicit PicLiveTransaction(ServerState & state) : st(state), metadata(state) {
        pic_sync(st);
        checkpoint = multimodal_checkpoint(st, st.mm_live_row);
        have_query = llama_kvmem_get_query(accumulator);
    }

    void begin(bool keep_live) {
        uint32_t rows = 0;
        if (metadata.row > 0) {
            stash.reset(llama_kvmem_stash_take(metadata.row, &rows));
            detached = true;
            pic_require(stash && rows == (uint32_t) metadata.row, "PIC could not preserve complete live stash");
            st.pic->transaction_backup_peak_bytes = std::max(st.pic->transaction_backup_peak_bytes,
                llama_kvmem_stash_bytes(stash.get()) + checkpoint.data->bytes());
        } else detached = true;
        memory_clear_all(st);
        std::string error;
        pic_check(pic_hybrid(st).attn_kvmem()->pic_bind_epoch(st.pic->identity.epoch, error), error);
        if (keep_live) {
            pic_require(metadata.row == 0 || llama_kvmem_stash_fork(stash.get(), metadata.row), "PIC live transaction fork failed");
            restore_state();
        }
    }

    void restore_state() {
        st.mm_live_checkpoint.reset();
        multimodal_restore(st, checkpoint, true);
        llama_kvmem_begin_cached_turn();
        if (have_query) pic_require(llama_kvmem_set_query(accumulator), "PIC could not restore query accumulator");
        llama_kvmem_freeze_query(false); // These transactions run before bootstrap query collection.
        if (st.pic->end >= 0) {
            llama_kvmem_set_request_span(st.pic->query, st.pic->end, st.pic->force);
            llama_kvmem_set_turn_spans(st.pic->spans);
        }
        metadata.restore(st);
    }

    void restore() {
        if (finished) return;
        try {
            pic_sync(st);
            memory_clear_all(st);
            if (metadata.row > 0) {
                pic_require(stash != nullptr, "PIC original live stash unavailable; cold rollback required");
                auto * owned = stash.release();
                pic_require(llama_kvmem_stash_put_prefix(owned, metadata.row), "PIC live restore failed; cold rollback required");
            }
            pic_require(llama_kvmem_store_n_tokens() == (uint32_t) metadata.row, "PIC restored row mismatch");
            restore_state();
            finished = true;
        } catch (...) {
            st.pic->restore_failed = true;
            st.n_batch = metadata.n_batch;
            finished = true;
            throw;
        }
    }

    void commit() { finished = true; }

    ~PicLiveTransaction() {
        if (!detached || finished) return;
        try { restore(); }
        catch (const std::exception & e) {
            LOG_ERR("srv    PIC transaction recovery failed: %s\n", e.what());
            try { memory_clear_all(st); } catch (...) {}
        }
    }
};

struct PicHostInputs {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * k = nullptr, * v = nullptr, * gate = nullptr, * beta = nullptr;
    explicit PicHostInputs(ggml_backend_t cpu, const llama_kvmem_pic_layer_capture & input) {
        ggml_init_params params {8*ggml_tensor_overhead(), nullptr, true};
        ctx = ggml_init(params);
        pic_require(ctx != nullptr, "PIC math input context allocation failed");
        try {
            auto tensor = [&](const llama_kvmem_pic_tensor & src) {
                auto * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, src.ne[0], src.ne[1], src.ne[2], src.ne[3]);
                pic_require((size_t) ggml_nelements(t) == src.data.size(), "PIC math input shape mismatch");
                return t;
            };
            k = tensor(input.k_conv); v = tensor(input.v_conv);
            gate = tensor(input.gate); beta = tensor(input.beta);
            buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
            pic_require(buffer != nullptr, "PIC math input buffer allocation failed");
            ggml_backend_tensor_set(k, input.k_conv.data.data(), 0, ggml_nbytes(k));
            ggml_backend_tensor_set(v, input.v_conv.data.data(), 0, ggml_nbytes(v));
            ggml_backend_tensor_set(gate, input.gate.data.data(), 0, ggml_nbytes(gate));
            ggml_backend_tensor_set(beta, input.beta.data.data(), 0, ggml_nbytes(beta));
        } catch (...) {
            if (buffer) ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            throw;
        }
    }
    ~PicHostInputs() { if (buffer) ggml_backend_buffer_free(buffer); if (ctx) ggml_free(ctx); }
};

static void pic_accumulate_capture(ServerState & st, kvmem_pic::segment_data & data,
                                    std::vector<llama_kvmem_pic_ubatch_capture> captures,
                                    size_t & next_row, int body_end, std::vector<float> & last_hidden) {
    auto & pic = *st.pic;
    size_t capture_bytes = 0;
    for (const auto & batch : captures) {
        capture_bytes += batch.target_hidden.data.capacity()*sizeof(float);
        for (const auto & layer : batch.layers) {
            for (const auto * tensor : {&layer.k_conv, &layer.v_conv, &layer.gate, &layer.beta,
                                       &layer.conv_input, &layer.conv_begin, &layer.conv_end})
                capture_bytes += tensor->data.capacity()*sizeof(float);
        }
    }
    pic.capture_host_peak_bytes = std::max(pic.capture_host_peak_bytes, capture_bytes);
    for (auto & batch : captures) {
        pic_require(batch.n_seqs == 1 && batch.n_seq_tokens == batch.n_tokens && batch.n_tokens > 0,
                    "PIC scratch capture must be contiguous seq0");
        pic_require(batch.logical_positions.size() == batch.n_tokens && batch.seq_ids.size() == batch.n_tokens,
                    "PIC capture is missing logical rows/sequence IDs");
        for (uint32_t i = 0; i < batch.n_tokens; ++i) {
            pic_require(batch.logical_positions[i] == (llama_pos) (next_row + i) &&
                batch.seq_ids[i] == std::vector<llama_seq_id>{0}, "PIC scratch capture row discontinuity");
        }
        const auto & hidden = batch.target_hidden;
        const auto width = (int64_t) llama_model_n_embd(st.model);
        pic_require(hidden.ne == std::array<int64_t, 4>{width, batch.n_tokens, 1, 1} &&
                    hidden.data.size() == (size_t) width * batch.n_tokens, "PIC hidden capture shape mismatch");
        last_hidden.assign(hidden.data.end() - width, hidden.data.end());
        batch.target_hidden = {};
        if (next_row >= KvmemPicServer::seam && next_row < (size_t) body_end) {
            pic_require(next_row + batch.n_tokens <= (size_t) body_end, "PIC body capture crosses a seam");
            for (auto & layer : batch.layers) {
                auto found = std::find_if(data.layers.begin(), data.layers.end(), [&](const auto & x) { return x.layer == layer.layer; });
                if (found == data.layers.end()) {
                    data.layers.push_back({});
                    found = std::prev(data.layers.end());
                    found->layer = layer.layer;
                }
                PicHostInputs inputs(pic.cpu, layer);
                pic.math_input_peak_bytes = std::max(pic.math_input_peak_bytes, ggml_backend_buffer_get_size(inputs.buffer));
                const auto * prefix = found->transition.t.empty() ? nullptr : &found->transition;
                found->transition = kvmem_pic::scan(pic.preferred, pic.cpu, inputs.k, inputs.v, inputs.gate, inputs.beta, prefix);
                found->conv_end = {layer.conv_end.ne, std::move(layer.conv_end.data)};
                layer = {};
            }
        }
        next_row += batch.n_tokens;
        batch.layers.clear();
    }
}

static void pic_scratch_decode(ServerState & st, const std::vector<llama_token> & tokens, int begin, int end,
                               kvmem_pic::segment_data & data, size_t & captured, int body_end, std::vector<float> & last_hidden, StreamIo * io) {
    for (int row = begin; row < end;) {
        pic_require(stream_heartbeat(io), "PIC independent build cancelled");
        const int count = std::min(st.n_batch, end - row);
        std::vector<llama_pos> positions(count), logical(count);
        std::vector<int32_t> n_seq(count, 1);
        llama_seq_id seq = 0;
        std::vector<llama_seq_id *> seqs(count, &seq);
        std::vector<int8_t> outputs(count, 0);
        for (int i = 0; i < count; ++i) positions[i] = logical[i] = row + i;
        outputs.back() = !st.spec.ok;
        auto batch = llama_batch_get_one(const_cast<llama_token *>(tokens.data()) + row, count);
        batch.pos = positions.data(); batch.logical_pos = logical.data();
        batch.n_seq_id = n_seq.data(); batch.seq_id = seqs.data(); batch.logits = outputs.data();
        pic_require(llama_decode(st.ctx, batch) == 0, "PIC independent target decode failed");
        if (st.spec.ok) pic_require(common_speculative_process(st.spec.spec, batch), "PIC independent original MTP decode failed");
        pic_sync(st);
        pic_accumulate_capture(st, data, st.pic->capture.take_ubatches(), captured, body_end, last_hidden);
        row += count;
    }
}

static size_t pic_candidate_bytes(const kvmem_pic::segment_data & data) {
    kvmem_pic::allocation_list allocations;
    pic_require(data.raw.allocations(data.raw.owner.get(), allocations), "PIC candidate allocation snapshot failed");
    std::map<const void *, size_t> unique;
    for (const auto & a : allocations) unique[a.first] = a.second;
    size_t result = 0;
    for (const auto & a : unique) result += a.second;
    result += data.target_hidden.data.capacity()*sizeof(float) + data.layers.capacity()*sizeof(kvmem_pic::segment_layer);
    for (const auto & layer : data.layers) result += (layer.transition.t.capacity() + layer.transition.u.capacity() +
                                                    layer.conv_end.data.capacity())*sizeof(float);
    return result;
}

static bool pic_build_segment(ServerState & st, const kvmem_prompt & prompt, kvmem_pic::token_range source, StreamIo * io) {
    auto & pic = *st.pic;
    const auto started = std::chrono::steady_clock::now();
    PicLiveTransaction transaction(st);
    auto & recurrent = *pic_hybrid(st).get_mem_recr();
    std::vector<int32_t> layers;
    for (size_t i = 0; i < recurrent.s_l.size(); ++i) if (recurrent.s_l[i]) layers.push_back((int32_t) i);
    auto build = pic.segments.begin(pic.key, prompt.tokens, source, layers, false);
    auto & data = build.data();
    std::vector<llama_token> tokens(prompt.tokens.begin() + source.begin, prompt.tokens.begin() + source.end);
    const int body_end = (int) tokens.size() - KvmemPicServer::seam;
    auto raw = std::make_shared<KvmemPicRaw>();
    raw->identity = pic.identity;
    raw->codecs = pic.codecs;
    size_t captured = 0;
    std::vector<float> last_hidden;
    try {
        pic.building = true;
        transaction.begin(false);
        st.n_batch = std::max(1, std::min(st.n_batch, 128));
        llama_kvmem_begin_cached_turn();
        llama_kvmem_set_request_span((int) tokens.size(), (int) tokens.size(), -1);
        llama_kvmem_set_turn_spans({{}, {{0, (int32_t) tokens.size()}}, 0});
        pic.capture.start();
        pic_scratch_decode(st, tokens, 0, KvmemPicServer::seam, data, captured, body_end, last_hidden, io);
        pic_scratch_decode(st, tokens, KvmemPicServer::seam, body_end, data, captured, body_end, last_hidden, io);
        if (st.spec.ok) {
            pic_require(common_speculative_get_state(st.spec.spec, 0, raw->body_carry), "PIC missing body MTP carry");
            const size_t row_bytes = (size_t) llama_model_n_embd(st.model)*sizeof(float);
            pic_require(raw->body_carry.size() == sizeof(llama_pos) + row_bytes, "PIC MTP carry shape mismatch");
            llama_pos synced = -1;
            std::memcpy(&synced, raw->body_carry.data(), sizeof(synced));
            pic_require(synced == body_end && last_hidden.size()*sizeof(float) == row_bytes,
                        "PIC MTP body boundary not synchronized");
            pic_require(std::memcmp(raw->body_carry.data() + sizeof(llama_pos),
                        last_hidden.data(), row_bytes) == 0,
                        "PIC captured hidden differs from original MTP carry");
        }
        pic_scratch_decode(st, tokens, body_end, (int) tokens.size(), data, captured, body_end, last_hidden, io);
        pic_require(pic.capture.stop() && captured == tokens.size(), "PIC independent capture incomplete");
        // Only body_carry retains a target hidden row; full captured hidden is discarded.
        uint32_t rows = 0;
        raw->stash.reset(llama_kvmem_stash_take((uint32_t) tokens.size(), &rows));
        pic_require(raw->stash && rows == tokens.size(), "PIC independent raw/MTP stash incomplete");
        data.raw = {raw, KvmemPicRaw::allocations};
        data.raw_rows = rows;
        transaction.restore();
        pic.building = false;
        // Original live is restored BEFORE publication; experimental entries do
        // not evict exact cache to obtain admission under the common budget.
        const size_t budget = (size_t) (st.pool_gb*1073741824.0);
        const size_t exact = pic_exact_cache_bytes(st);
        const size_t candidate = pic_candidate_bytes(data) + 65536 + 2*tokens.capacity()*sizeof(llama_token);
        if (candidate > kvmem_pic::maximum_store_bytes || exact > budget || candidate > budget - exact) return false;
        if (pic.segments.bytes() > budget - exact - candidate) pic.segments.clear();
        if (pic.segments.bytes() > budget - exact - candidate) return false;
        raw.reset();
        auto published = pic.segments.publish(std::move(build));
        if (!published) return false;
        if (pic.segments.bytes() > budget - exact) {
            published.entry.reset();
            pic.segments.clear();
            return false;
        }
        ++pic.builds;
        pic.build_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        LOG_INF("srv    PIC build id=%llu tokens=%zu body=%d mtp=%d bytes=%llu\n",
                (unsigned long long) published.entry->id, tokens.size(), body_end - KvmemPicServer::seam,
                (int) st.spec.ok, (unsigned long long) pic.segments.bytes());
        return true;
    } catch (const std::exception & e) {
        if (pic.capture.enabled()) pic.capture.stop();
        transaction.restore(); // Recovery failure propagates; never decode on scratch state.
        pic.building = false;
        ++pic.failures;
        LOG_WRN("srv    PIC build fallback reason=%s\n", e.what());
        return false;
    }
}

static void pic_reset(ServerState & st) {
    if (!st.pic) return;
    auto & pic = *st.pic;
    pic.matches.clear();
    pic.segments.clear();
    pic.feature_fallback = false;
    pic.key.model_generation = pic.key.configuration_generation = st.mm_cache_generation;
    pic.identity.epoch = st.mm_cache_generation;
}

static bool pic_plan_request(ServerState & st, const kvmem_prompt & prompt, int base, int eval_end, StreamIo * io) {
    if (st.pic) { st.pic->matches.clear(); st.pic->feature_fallback = false; }
    if (prompt.has_media() || st.mm_msg_token == LLAMA_TOKEN_NULL || !pic_ready(st)) return false;
    auto & pic = *st.pic;
    // Reserve a fully computed suffix for retrieval features/logits and pending MTP.
    const int cutoff = std::max(base, eval_end - st.query_max_tokens);
    const int length = (int) kvmem_pic::minimum_segment_tokens;
    std::vector<int> messages;
    for (int i = 0; i < eval_end; ++i) if (prompt.tokens[i] == st.mm_msg_token) messages.push_back(i);
    if (messages.size() < 2 || cutoff - std::max(base, messages[1]) < length) return false;
    const std::vector<kvmem_pic::token_range> allowed = {{(size_t) std::max(base, messages[1]), (size_t) cutoff}};
    pic.matches = pic.segments.match(pic.key, prompt.tokens, allowed, false);
    if (pic.matches.empty()) {
        // Stable message-local chunks, taken from the already tokenized request.
        messages.push_back(eval_end);
        bool attempted = false;
        for (size_t i = 1; i + 1 < messages.size() && !attempted; ++i) {
            for (int begin = messages[i]; begin + length <= std::min(messages[i + 1], cutoff); begin += length) {
                if (begin < base) continue;
                attempted = true;
                try { pic_build_segment(st, prompt, {(size_t) begin, (size_t) begin + length}, io); }
                catch (const std::exception & e) {
                    if (pic.restore_failed) throw;
                    LOG_WRN("srv    PIC build declined reason=%s\n", e.what());
                }
                break;
            }
        }
        pic.matches = pic.segments.match(pic.key, prompt.tokens, allowed, false);
    }
    // No exact-user Q is published on this request. Its feature span may be
    // replaced by a body whose FA query activations were never evaluated here.
    pic.feature_fallback = !pic.matches.empty();
    return pic.feature_fallback;
}

static void pic_request_spans(ServerState & st, int query, int end, int force, const llama_kvmem_turn_spans & spans) {
    if (!st.pic) return;
    st.pic->query = query;
    st.pic->end = end;
    st.pic->force = force;
    st.pic->spans = spans;
}

static bool pic_splice_body(ServerState & st, const kvmem_pic::segment_match & match) {
    auto & pic = *st.pic;
    const auto & entry = *match.entry;
    const auto & raw = *static_cast<const KvmemPicRaw *>(entry.data.raw.owner.get());
    const int begin = (int) match.range.begin + KvmemPicServer::seam;
    const int end = (int) match.range.end - KvmemPicServer::seam;
    const auto started = std::chrono::steady_clock::now();
    pic_require(st.mm_live_row == begin && end > begin, "PIC body cursor mismatch");
    PicLiveTransaction transaction(st);
    double phase_begin = 0, phase_capture = 0, phase_compose = 0, phase_prepare = 0, phase_validate = 0,
           phase_commit = 0, phase_install = 0;
    try {
        transaction.begin(true);
        const auto t_after_begin = std::chrono::steady_clock::now();
        auto & hybrid = pic_hybrid(st);
        auto & adapter = *hybrid.attn_kvmem();
        pic_sync(st);
        kvmem_pic::allocation_list drained;
        pic_require(llama_kvmem_live_allocations(drained), "PIC splice harvest flush failed");
        kvmem_pic::recurrent_state state;
        std::string error;
        pic_check(kvmem_pic::pic_state_capture(*hybrid.get_mem_recr(), state, error), error);
        const auto t_after_capture = std::chrono::steady_clock::now();
        pic_require(state.end == begin && state.layers.size() == entry.data.layers.size(), "PIC GDN source boundary mismatch");
        for (size_t i = 0; i < state.layers.size(); ++i) {
            auto & layer = state.layers[i];
            const auto & saved = entry.data.layers[i];
            pic_require(layer.layer == saved.layer, "PIC GDN layer identity mismatch");
            layer.recurrent = kvmem_pic::compose(pic.preferred, pic.cpu, saved.transition, layer.recurrent);
            layer.conv = saved.conv_end.data;
        }
        const auto t_after_compose = std::chrono::steady_clock::now();
        state.end = end;
        std::vector<uint8_t> carry;
        if (st.spec.ok) {
            const size_t bytes = sizeof(llama_pos) + (size_t) llama_model_n_embd(st.model)*sizeof(float);
            pic_require(raw.body_carry.size() == bytes && raw.identity.mtp_model == llama_get_model(st.spec.ctx_dft),
                        "PIC missing original MTP body KV/carry");
            carry = raw.body_carry;
            const llama_pos synced = end;
            std::memcpy(carry.data(), &synced, sizeof(synced));
        }
        kvmem_pic::kv_splice_request request;
        request.source_identity = raw.identity;
        request.destination_identity = pic.identity;
        request.destination_identity.independently_evaluated = false;
        request.source_codecs = raw.codecs;
        request.destination_codecs = pic.codecs;
        request.source_begin = KvmemPicServer::seam;
        request.destination_begin = begin;
        request.tokens.assign(entry.tokens.begin() + KvmemPicServer::seam, entry.tokens.end() - KvmemPicServer::seam);
        for (int row = begin; row < end; ++row) request.positions.push_back({row, row, row, row});
        request.max_plan_bytes = kvmem_pic::maximum_store_bytes;
        kvmem_pic::kv_splice_plan plan;
        pic_check(kvmem_pic::pic_kv_prepare(*raw.stash, adapter, request, plan, error), error);
        const auto t_after_prepare = std::chrono::steady_clock::now();
        pic_check(kvmem_pic::pic_kv_validate(adapter, plan, error), error);
        const auto t_after_validate = std::chrono::steady_clock::now();
        pic_check(adapter.pic_kv_commit(plan, error), error);
        const auto t_after_commit = std::chrono::steady_clock::now();
        pic_check(kvmem_pic::pic_state_install(*hybrid.get_mem_recr(), state, end, error), error);
        const auto t_after_install = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        phase_begin = ms(started, t_after_begin);
        phase_capture = ms(t_after_begin, t_after_capture);
        phase_compose = ms(t_after_capture, t_after_compose);
        phase_prepare = ms(t_after_compose, t_after_prepare);
        phase_validate = ms(t_after_prepare, t_after_validate);
        phase_commit = ms(t_after_validate, t_after_commit);
        phase_install = ms(t_after_commit, t_after_install);
        if (st.spec.ok) common_speculative_set_state(st.spec.spec, 0, carry);
        pic_sync(st);
        pic_require(llama_kvmem_store_n_tokens() == (uint32_t) end && llama_kvmem_recr_pos_max() == end - 1,
                    "PIC committed cursors disagree");
        if (st.spec.ok) {
            std::vector<uint8_t> actual;
            pic_require(common_speculative_get_state(st.spec.spec, 0, actual) && actual == carry,
                        "PIC MTP carry readback mismatch");
        }
        st.mm_live_row = end;
        st.mm_live_checkpoint.reset();
        st.mm_query.reset();
        st.mm_pending_query.reset();
        st.mm_approximate = true;
        llama_kvmem_reset_query();
        llama_kvmem_freeze_query(false);
        llama_kvmem_set_request_span(pic.query, pic.end, pic.force);
        llama_kvmem_set_turn_spans(pic.spans);
        multimodal_remember(st, multimodal_checkpoint(st, end));
        transaction.commit();
        ++pic.hits;
        pic.skipped_rows += end - begin;
        pic.splice_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        LOG_INF("srv    PIC hit id=%llu body=[%d,%d) mtp=%d approximate=1 query_source=bootstrap_suffix\n",
                (unsigned long long) entry.id, begin, end, (int) st.spec.ok);
        LOG_INF("srv    PIC splice phases begin=%.1f capture=%.1f compose=%.1f prepare=%.1f validate=%.1f commit=%.1f install=%.1f total=%.1f layers=%zu\n",
                phase_begin, phase_capture, phase_compose, phase_prepare, phase_validate, phase_commit,
                phase_install,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
                state.layers.size());
        return true;
    } catch (const std::exception & e) {
        transaction.restore();
        ++pic.failures;
        LOG_WRN("srv    PIC splice fallback body=[%d,%d) reason=%s\n", begin, end, e.what());
        return false;
    }
}

static int pic_decode_span(ServerState & st, int begin, int end, StreamIo * io, const std::vector<int> * checkpoints) {
    if (!st.pic || st.pic->matches.empty()) return multimodal_decode_span(st, begin, end, false, io, checkpoints);
    int row = begin;
    for (const auto & match : st.pic->matches) {
        if ((int) match.range.begin < row || (int) match.range.end > end) continue;
        const int body_begin = (int) match.range.begin + KvmemPicServer::seam;
        const int body_end = (int) match.range.end - KvmemPicServer::seam;
        const int rc = multimodal_decode_span(st, row, body_begin, false, io, checkpoints);
        if (rc != 0) return rc;
        if (!stream_heartbeat(io)) return KVMEM_DECODE_ABORT;
        row = pic_splice_body(st, match) ? body_end : body_begin;
        const int tail = multimodal_decode_span(st, row, (int) match.range.end, false, io, checkpoints);
        if (tail != 0) return tail;
        row = (int) match.range.end;
    }
    const int rc = multimodal_decode_span(st, row, end, false, io, checkpoints);
    st.pic->matches.clear();
    return rc;
}
