#include "kvmem/raw_kv_store.hpp"
#include "../../src/adapter/kvmem-resident-tags.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>

static void check(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}

static kvmem::RawKvStoreConfig config(uint32_t layers) {
    kvmem::RawKvStoreConfig c;
    c.n_layer = layers;
    c.n_embd_k = c.n_embd_v = 4;
    c.block_tokens = 128;
    c.k_gpu_row_bytes = 8;
    c.v_gpu_row_bytes = 4;
    return c;
}

static void fill(kvmem::RawKvStore & raw, uint32_t rows) {
    const auto & c = raw.config();
    for (uint32_t il = 0; il < c.n_layer; ++il) {
        std::vector<uint8_t> k(rows * c.k_gpu_row_bytes, uint8_t(il + 19));
        std::vector<uint8_t> v(rows * c.v_gpu_row_bytes, uint8_t(il + 31));
        raw.write_layer_k_gpu(0, rows, il, k.data());
        raw.write_layer_v_gpu(0, rows, il, v.data());
    }
}

static kvmem_resident_tags::payloads refs(const kvmem::RawKvStore & target,
                                        const kvmem::RawKvStore & mtp, uint32_t id, uint32_t rows) {
    kvmem_resident_tags::payloads out;
    for (const auto * raw : {&target, &mtp}) {
        for (uint32_t il = 0; il < raw->config().n_layer; ++il) {
            kvmem_resident_tags::payload k, v;
            if (!raw->packed_refs(id, il, rows, k, v)) return {};
            out.push_back(std::move(k));
            out.push_back(std::move(v));
        }
    }
    return out;
}

static void boundaries_and_writes() {
    kvmem::RawKvStore a(config(3)), ma(config(1));
    fill(a, 1037);
    fill(ma, 1037);
    for (uint32_t rows : {1u, 127u, 128u, 129u, 897u, 1025u, 1037u}) {
        auto b = a.clone_prefix(rows);
        auto mb = ma.clone_prefix(rows);
        kvmem_resident_tags tags;
        tags.resize(4);
        // Sparse set, including blocks outside a two-block semantic budget.
        uint32_t slot = 0;
        for (uint32_t id : {0u, 1u, 7u, 8u}) {
            if (id * 128 >= rows) continue;
            const uint32_t n = std::min(128u, rows - id*128);
            auto p = refs(*b, *mb, id, n);
            check(p.size() == 8, "target/MTP layer coverage");
            tags.publish(slot, id*128, n, p);
            check(tags.matches(slot, id*128, n, refs(*b, *mb, id, n)), "same-slot prefix miss");
            check(!tags.matches(slot, id*128+1, n, p), "wrong orig hit");
            check(!tags.matches(slot, id*128, n+1, p), "wrong nt hit");
            check(!tags.matches(slot+1, id*128, n, p), "different slot hit");
            ++slot;
        }
        auto p = refs(*b, *mb, 0, std::min(rows,128u));
        tags.invalidate(0); // Scheduled graph write, even if later cancelled.
        check(!tags.matches(0, 0, std::min(rows,128u), p), "write epoch did not revoke");
        const uint32_t n = std::min(rows,128u);
        b->invalidate_packed_block(0, 0);
        check(refs(*b, *mb, 0, n).empty(), "dirty host authority accepted");
        if (rows > 128) check(!refs(*b,*mb,1,std::min(128u,rows-128)).empty(), "unwritten history invalidated");
        std::vector<uint8_t> fresh_k(n*8, 91), fresh_v(n*4, 92);
        for (uint32_t il=0;il<3;++il) {
            b->write_layer_k_gpu(0,n,il,fresh_k.data());
            b->write_layer_v_gpu(0,n,il,fresh_v.data());
        }
        auto changed = refs(*b,*mb,0,n);
        check(changed.size()==8 && changed[0]!=p[0], "captured write did not COW");
        check(!tags.matches(0,0,n,changed), "unpublished capture hit");
        tags.publish(0,0,n,changed);
        check(tags.matches(0,0,n,changed), "completed capture did not publish");
        check(!tags.matches(0,0,n,p), "old branch hit after overwrite");
        tags.clear();
        check(!tags.matches(0,0,n,changed), "full reset retained proof");
    }
}

static void ancestry_and_mtp() {
    kvmem::RawKvStore a(config(2)), ma(config(1)), unrelated(config(2));
    fill(a,256); fill(ma,256); fill(unrelated,256);
    auto cow=a.clone_prefix(256), mcow=ma.clone_prefix(256);
    auto deep=a.clone_prefix_deep(256), mdeep=ma.clone_prefix_deep(256);
    auto p=refs(a,ma,0,128);
    kvmem_resident_tags tags;
    tags.resize(2); tags.publish(0,0,128,p);
    check(tags.matches(0,0,128,refs(*cow,*mcow,0,128)), "ancestor COW miss");
    check(!tags.matches(0,0,128,refs(*deep,*mdeep,0,128)), "deep clone incorrectly matched");
    check(!tags.matches(0,0,128,refs(unrelated,ma,0,128)), "equal text/bytes conflated with ancestry");
    std::vector<uint8_t> v(4,99);
    mcow->write_layer_v_gpu(0,1,0,v.data());
    check(!tags.matches(0,0,128,refs(*cow,*mcow,0,128)), "MTP-only divergence ignored");
    tags.publish(1,128,128,refs(a,ma,1,128));
    tags.invalidate(0);
    check(tags.matches(1,128,128,refs(a,ma,1,128)), "unwritten physical slot lost proof");
    tags.resize(3);
    check(!tags.matches(1,128,128,refs(a,ma,1,128)), "buffer-size generation change retained proof");
}

static void ownership_and_budget() {
    kvmem::RawKvStore a(config(2)), ma(config(1));
    fill(a,128); fill(ma,128);
    auto p=refs(a,ma,0,128);
    std::weak_ptr<const std::vector<uint8_t>> weak=p[0];
    const auto * identity=p[0].get();
    kvmem_resident_tags tags;
    tags.resize(1); tags.publish(0,0,128,p);
    std::vector<std::pair<const void*,size_t>> raw, cached;
    a.append_allocations(raw); ma.append_allocations(raw); tags.append_allocations(cached);
    std::map<const void*,size_t> raw_unique(raw.begin(),raw.end());
    size_t shared=0;
    for(const auto & item:cached) {
        const auto it=raw_unique.find(item.first);
        if(it!=raw_unique.end()) { ++shared; check(it->second==item.second,"inconsistent capacity identity"); }
    }
    check(shared==12,"target/MTP lease accounting failed to expose vector + byte identities");
    a.clear(); ma.clear(); p.clear();
    check(!weak.expired() && weak.lock().get()==identity,"GPU proof lease allowed pointer ABA");
    cached.clear(); tags.append_allocations(cached);
    check(std::any_of(cached.begin(),cached.end(),[identity](const auto & e){return e.first==identity;}),
          "last tag owner invisible to live budget");
    tags.clear();
    check(weak.expired(),"budget drop did not release last payload owner");
}

static void cross_ubatch_tail() {
    // 513 rows split 507+6: block [480,512) still has five unwritten rows
    // after the first graph. An older queued copy must finish BEFORE the
    // second graph invalidates validity; otherwise it can publish a stale tail.
    auto cfg = config(2);
    cfg.block_tokens = 32;
    kvmem::RawKvStore raw(cfg);
    std::vector<uint8_t> gpu_k(513*8, 0xee), gpu_v(513*4, 0xee);
    std::fill(gpu_k.begin(), gpu_k.begin()+507*8, 0x11);
    std::fill(gpu_v.begin(), gpu_v.begin()+507*4, 0x33);
    for (uint32_t il=0; il<2; ++il) {
        // Drain an old copy first. New code disables this early full-block
        // path, but write-time invalidation must remain safe even if queued.
        raw.write_layer_k_gpu(480,32,il,gpu_k.data()+480*8);
        raw.write_layer_v_gpu(480,32,il,gpu_v.data()+480*4);
    }
    raw.invalidate_packed_block(15,27); // actual second ubatch starts at 507
    for (uint32_t il=0; il<2; ++il) {
        check(!raw.has_k_gpu(15,il,32) && !raw.has_v_gpu(15,il,32), "507+6 stale full-block counter survived");
        check(raw.has_k_gpu(15,il,27) && raw.has_v_gpu(15,il,27), "completed prefix lost");
    }
    std::fill(gpu_k.begin()+507*8,gpu_k.end(),0x22);
    std::fill(gpu_v.begin()+507*4,gpu_v.end(),0x44);
    for (uint32_t il=0; il<2; ++il) {
        raw.write_layer_k_gpu(480,33,il,gpu_k.data()+480*8);
        raw.write_layer_v_gpu(480,33,il,gpu_v.data()+480*4);
        std::vector<uint8_t> k(32*8),v(32*4);
        check(raw.copy_k_gpu(15,il,k.data(),32) && raw.copy_v_gpu(15,il,v.data(),32), "completed block missing");
        check(std::equal(k.begin(),k.end(),gpu_k.begin()+480*8) &&
              std::equal(v.begin(),v.end(),gpu_v.begin()+480*4), "507+6 stale tail bytes survived");
        check(raw.has_k_gpu(16,il,1) && raw.has_v_gpu(16,il,1) &&
              !raw.has_k_gpu(16,il,2), "partial final block validity");
    }
}

int main() {
    try {
        boundaries_and_writes();
        ancestry_and_mtp();
        ownership_and_budget();
        cross_ubatch_tail();
        std::puts("PASS resident tags: prefix boundaries, sparse history, epochs, COW/MTP, ABA, owner budget");
    } catch (const std::exception & e) {
        std::fprintf(stderr,"FAIL: %s\n",e.what());
        return 1;
    }
}
