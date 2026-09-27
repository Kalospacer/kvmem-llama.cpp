#pragma once

#include <cstring>

// Included after the transfer helpers in mtp-kv-test.cpp. This runs a real
// Qwen hybrid graph and reads allocated backend KV, not simulated tag state.
struct harvest_boundary_probe {
    llama_memory_kvmem * mem = nullptr;
    uint32_t completed = 128;
    std::vector<int64_t> widths;
    std::string failure;

    static bool eval(ggml_tensor * tensor, bool ask, void * user) {
        auto & probe = *static_cast<harvest_boundary_probe *>(user);
        const bool selected = probe.mem && std::strcmp(ggml_get_name(tensor), "beta_sigmoid-0") == 0;
        if (ask) return selected;
        if (!selected) return true;
        try {
            // Previous ubatch harvest has returned. The current ubatch has not
            // reached any target attention layer yet. Force queued D2H to commit.
            probe.mem->harvest_flush();
            kvmem_transfer_test_access::flush(*probe.mem);
            for (const auto & block : probe.mem->store().blocks()) {
                if (block.gpu_slot < 0 || block.orig_pos_end() <= probe.completed) continue;
                const uint32_t first_unfinished = probe.completed > block.orig_pos_start
                        ? probe.completed - block.orig_pos_start : 0;
                for (int32_t il : probe.mem->get_kv()->get_layer_ids()) {
                    if (probe.mem->raw().has_k_gpu(block.block_id, il, first_unfinished + 1) ||
                            probe.mem->raw().has_v_gpu(block.block_id, il, first_unfinished + 1)) {
                        probe.failure = "packed raw includes unfinished rows after " + std::to_string(probe.completed) + " tokens";
                    }
                }
            }
            probe.widths.push_back(tensor->ne[2]);
            probe.completed += uint32_t(tensor->ne[2]);
        } catch (const std::exception & ex) {
            probe.failure = ex.what();
        }
        return true;
    }
};

static void check_harvest_boundary(llama_model * model) {
    llama_kvmem_params kp{};
    kp.enabled = true;
    kp.method = 1;
    kp.block_tokens = 128;
    kp.budget = 1024;
    kp.gen_reserve = 256;
    kp.query_begin = kp.query_end = kp.force_pos = -1;
    llama_kvmem_set_params(&kp);
    llama_kvmem_set_pool_gpu_reuse(true);
    harvest_boundary_probe probe;
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 5;
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = harvest_boundary_probe::eval;
    cp.cb_eval_user_data = &probe;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    require(bool(ctx), "harvest boundary context init failed");
    auto * hybrid = dynamic_cast<llama_memory_kvmem_hybrid *>(llama_get_memory(ctx.get()));
    require(hybrid != nullptr, "harvest boundary test requires a Qwen hybrid model");
    auto * mem = hybrid->attn_kvmem();
    const auto seed = common_tokenize(llama_model_get_vocab(model), " Preserve every computed token exactly.", false, true);
    require(!seed.empty(), "empty boundary token fixture");
    std::vector<llama_token> input(385);
    for (size_t i = 0; i < input.size(); ++i) input[i] = seed[i % seed.size()];
    auto decode = [&](uint32_t begin, uint32_t count) {
        std::vector<llama_pos> positions(count);
        for (uint32_t i = 0; i < count; ++i) positions[i] = begin + i;
        auto batch = llama_batch_get_one(input.data() + begin, count);
        batch.pos = batch.logical_pos = positions.data();
        require(llama_decode(ctx.get(), batch) == 0, "harvest boundary decode failed");
        llama_synchronize(ctx.get());
    };

    decode(0, 128);
    const auto flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    std::vector<uint8_t> recurrent(llama_state_seq_get_size_ext(ctx.get(), 0, flags));
    require(llama_state_seq_get_data_ext(ctx.get(), recurrent.data(), recurrent.size(), 0, flags) == recurrent.size(),
            "boundary recurrent checkpoint failed");
    require(llama_kvmem_pool_preserve_begin(), "GPU reuse unavailable for boundary regression");
    uint32_t rows = 0;
    std::unique_ptr<llama_kvmem_stash, decltype(&llama_kvmem_stash_free)> stash(
            llama_kvmem_stash_take(128, &rows), llama_kvmem_stash_free);
    require(stash && rows == 128, "boundary warm-prefix stash failed");
    llama_memory_clear(llama_get_memory(ctx.get()), true);
    llama_kvmem_pool_preserve_end(true);
    require(llama_kvmem_stash_fork(stash.get(), 128), "boundary warm-prefix fork failed");
    require(llama_state_seq_set_data_ext(ctx.get(), recurrent.data(), recurrent.size(), 0, flags) == recurrent.size(),
            "boundary recurrent restore failed");
    require(mem->tracking_gpu_writes(), "boundary regression lost write tracking");

    probe.mem = mem;
    decode(128, 257); // split_equal(256, keep_tail=6): 251 + 6
    probe.mem = nullptr;
    require(probe.widths == std::vector<int64_t>({251, 6}) && probe.completed == 385,
            "boundary fixture did not execute the required 251+6 ubatches");
    mem->harvest_flush();
    kvmem_transfer_test_access::flush(*mem);
    // Do not invalidate raw here: that would repair the bug under test.
    for (const auto & block : mem->store().blocks()) kvmem_transfer_test_access::save(*mem, block.block_id);
    kvmem_transfer_test_access::flush(*mem);
    size_t mismatches = 0;
    for (const auto & block : mem->store().blocks()) {
        require(block.gpu_slot >= 0, "boundary fixture unexpectedly evicted a block");
        for (int32_t il : mem->get_kv()->get_layer_ids()) {
            for (bool is_k : {true, false}) {
                auto * tensor = is_k ? mem->get_kv()->get_k_storage(il) : mem->get_kv()->get_v_storage(il);
                const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
                std::vector<uint8_t> gpu(size_t(block.n_tokens) * row_bytes), packed(gpu.size());
                ggml_backend_tensor_get(tensor, gpu.data(), size_t(block.gpu_slot) * kp.block_tokens * row_bytes, gpu.size());
                const bool valid = is_k ? mem->raw().copy_k_gpu(block.block_id, il, packed.data(), block.n_tokens)
                                        : mem->raw().copy_v_gpu(block.block_id, il, packed.data(), block.n_tokens);
                require(valid, "completed boundary rows were not harvested");
                if (gpu != packed) {
                    ++mismatches;
                    std::fprintf(stderr, "BOUNDARY_MISMATCH block=%u layer=%d kind=%c\n", block.block_id, il, is_k ? 'K' : 'V');
                }
            }
        }
    }
    std::printf("HARVEST_BOUNDARY block=128 prefix=128 appended=257 ubatches=251,6 first_completed=379 "
                "protected_rows=[379,384) final_completed=385 mismatches=%zu validity=%s\n",
                mismatches, probe.failure.empty() ? "ok" : probe.failure.c_str());
    llama_kvmem_set_pool_gpu_reuse(false);
    require(probe.failure.empty(), probe.failure.c_str());
    require(mismatches == 0, "packed raw differs from completed GPU KV");
    std::puts("PASS actual GPU harvest boundary: partial ubatch, D2H commit, packed K/V bytes");
}
