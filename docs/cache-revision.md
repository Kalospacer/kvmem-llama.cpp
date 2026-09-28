# Cache revision: rc3 production baseline

This branch starts at `v0.16.0-rc3` (`4773705`) and includes the previously
deployed message-boundary checkpoints and conversation pool. It intentionally
does not merge the later ROCm and upstream changes from the fork's master.

## Implemented

- Retain committed query state with a parked conversation. Validate its prefix,
  bounds and configuration before import, and retain a usable recurrent anchor.
- Truncate a restored conversation to the selected checkpoint before uploading
  its KV. Failed restores consume their stash and leave an empty live state.
- Share packed host K/V by layer with copy-on-write. Account for unique
  allocations and the bytes actually released by eviction, including live owners.
- Optionally reuse unchanged same-slot GPU blocks with shared ancestry. Target
  and MTP writes invalidate the reuse proof. In reuse mode, packed block harvest
  is deferred to completed-graph boundaries, avoiding premature block-tail reads.
- Support `KVMEM_WINDOWS_RESOURCE_POLICY=normal` before model loading. The
  Windows helpers apply/read back process and thread memory priority and preserve
  an identity-checked rollback journal. `inherit` retains the caller's policy.

Relevant options:

```text
--no-kvmem-pool-query
--kvmem-pool-restore full|prefix
--kvmem-pool-copy deep|cow
--kvmem-pool-gpu-reuse off|on
```

GPU reuse defaults to off. It was explicitly enabled in the validated deployment.

## PIC: functional, off by default, not beneficial yet

`--kvmem-pic on` now runs end to end: the adapter implements `pic_kv_commit`,
the server routes message-local 512-token segments through a live transaction,
and MTP state is carried across the splice. Off remains the default and the
validated deployment keeps it off. PIC is approximate; outputs are not
guaranteed to match exact prefill.

- M-RoPE models mark every row `spatial` (`n_pos >= 3`), including scalar text.
  Preparation now checks tokens and equal position axes instead of the flag.
- PIC relocation uses `kvmem_pic::hadamard_rows_fast` (butterfly WHT) instead of
  the dense O(n^2) `kvmem_hadamard_rows`. This cut splice preparation from
  14.6 s to 1.05 s for a 496-token body. Exact cache paths still use the
  original function.
- Per-step timings: `KVMEM_PIC_PREPARE` and `KVMEM_PIC_COMMIT` under KVMEM_TRACE.

Measured on V100 with the synthetic suite (9/9 checks passing, 3 cases per
quality category matching the exact baseline): a request that hits a segment
takes 7.3 s against 6.4 s with PIC off, and a multi-message request that builds
a segment takes 25.7 s against 11.6 s. With 512-token segments one hit can
save at most about 0.6 s of recompute, while the splice has more than 2 s of
fixed cost (GDN compose alone is 1.5 s) and each segment stores about 300 MB of
GDN transitions. Building runs synchronously inside the request (192
independent scans). Becoming beneficial needs message-level segments, building
off the request path, and batched GPU-resident compose/scan.

## Pool: deep branches and checkpoint retention

A request that shares only an early part of the live conversation used to drop
the rest when the dropped span had no checkpoint exactly on a message-start
token, e.g. one large system message followed by one user message. Returning to
the original then recomputed tens of thousands of tokens. Dropping at least
16384 rows now always parks the live conversation first; the per-turn rewritten
tail of a normal continuation stays far below that.

A parked entry keeps `--kvmem-pool-ckpts` recurrent checkpoints (default 5). It
now always keeps the oldest one, the Q anchor (or the newest row without one)
and the first checkpoint of the current turn. The remaining slots drop the
checkpoint closest to its neighbours, non-message rows first, so a later rewrite
of older context can resume inside the history instead of near the start.

Server logs add `kvmem pool: route` (why a dropped live tail is or is not
parked) and `kvmem pool: stash_phases` (stash timings, page faults, working set
and commit). With KVMEM_TRACE, `KVMEM_STASH_TAKE` breaks down the adapter side.

`scripts/cache-revision-regression.py --extended --branch-focus` covers a tail
branch, a completed mid-history branch, a cancelled mid-history branch and the
return to the original each time. Measured on V100 with a 72,892-token fixture:
returning after a mid-history branch went from 75.9 s to 1.5 s, returning after
a cancelled one from 94.7 s to 0.7 s, and the tail branch stayed at 1.0 s.

## Two conversation caches

This tree carries two ways to keep several conversations' KV in host RAM. They
cannot run together: the pool parks and restores host KV inside a request,
while `--kvmem-conversations` swaps the whole host store before it.

| | Conversation pool (default) | Multi-store cache (`--kvmem-conversations N`) |
|---|---|---|
| Unit | detached stash per parked branch | one whole host store per conversation |
| Shared system prompt | prefix shared copy-on-write | one copy per conversation |
| Restore | only rows up to the resume point | the whole working set |
| Query state (Q) | carried with the entry | reset on switch |
| Branch rule | message-start checkpoint or a deep drop | `rows - lcp <= last_n_gen + 64` |
| Disk tier | none | `--kvmem-session-nvme-gb` |
| Client id | none | `kvmem.conversation_id` |

The pool's branch rule suits clients that rewrite a temporary tail every turn:
the multi-store rule classifies such a turn as a branch and would park an
unusable copy each time. The multi-store cache suits append-only clients,
bounded disk spill and explicit conversation ids.

Selection at startup:

- No `--kvmem-conversations`: the pool runs as before.
- `--kvmem-conversations N > 1` with no `--kvmem-pool-*` flag: the pool is
  switched off and the server logs `KVMEM_STARTUP conversation pool disabled`.
- Both given explicitly: startup fails and asks for one of them, or for
  `--no-kvmem-pool`.

The pool's decisions live in `tools/kvmem-pool-policy.h`, freestanding like
`tools/kvmem-conversation-store.h`, and the server executes the plan it
returns. `tests/conversation-pool-test.cpp` runs without a model or GPU and
pins both production incidents above: each case also runs the pre-fix logic
and shows it fails.

## Validation

The source baseline was built for SM70 with MSVC and CUDA 12.9. The deployed
executable SHA256 was
`BE1B822256363AAD4070D7CB3EB792E66E7A68CA0488767BFF5CFBF9C3A4B031`.
That identifies the recorded deployment artifact, not a promise of reproducible
binary hashes across build environments or Git metadata.

- Host raw/COW and resident-tag tests passed, including a 507+6 ubatch-tail case.
- PIC store and GDN/RoPE CPU numerical tests passed, including fast vs dense Hadamard.
- Eight synthetic GPU checks passed: interleaved conversation routing, actual
  branch retention, output equality against the uninterrupted branch, and GPU
  reuse counters. The sequence recorded 84 reused blocks and 399 MiB of skipped
  target/MTP uploads in total; this is not a per-request speedup measurement.
- The extended >49K-token, cancellation/retry and small-pool GPU matrix was not
  run for this release. The regression script retains those optional modes.

Use `scripts/cache-revision-regression.py` only against a dedicated test instance.
The script uses synthetic data and contains no deployment credentials.

## Rebuilding and upgrading

Initialize the pinned llama.cpp submodule and run `scripts/apply-patches.sh`,
or use `scripts/windows/build.ps1`. The cumulative patch includes all core changes.
The cache revision upgrade supports a tree with the previous rc3 patch applied;
applying the maintained patches twice is supported. Do not change the submodule
pin to an unrecorded local commit.

Windows operations helpers are in `scripts/windows/`. Back up the executable,
matching launcher and task settings together before deployment. Do not pass new
options to an older executable during rollback. Process-policy journals are
specific to the captured process/thread creation identities.

AI assistance was used to implement and review this change. Production chat
content, credentials, private deployment logs and binary artifacts are not part
of this branch.
