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

## One conversation cache

The conversation pool is the only cache that keeps several conversations' KV
in host RAM. Upstream's multi-store cache (`--kvmem-conversations N`, one whole
host store per conversation, with an optional session disk tier) was removed
from the server after an A/B run on the V100 with the same binary and the same
18 requests, whose answers matched word for word:

| | Pool | Multi-store |
|---|---|---|
| Total wall time | 294 s | 474 s |
| Recomputed tokens | 148K | 273K |
| Process private memory | 10.8 GB | 27.7 GB |
| AstrBot-style turn (rewritten temporary tail) | ~11 s | 37-51 s |

The multi-store branch rule (`rows - lcp <= last_n_gen + 64`) classifies every
rewritten-tail turn as a new conversation and recomputes it in full, and it
cannot share a system prompt across stores.

Its flags still parse, so existing launch scripts start:

- `--kvmem-conversations N`: `N > 1` enables the pool with
  `--kvmem-pool-max N`; `N <= 1` does nothing.
- `--kvmem-conversations-gb G`: `G > 0` sets `--kvmem-pool-gb G`.
- An explicit `--kvmem-pool-max` / `--kvmem-pool-gb` wins over the mapped
  value. Either way the server prints one `KVMEM_STARTUP --kvmem-conversations`
  line saying what it did.
- `--kvmem-session-ram-gb`, `--kvmem-session-nvme-gb` and
  `--kvmem-session-cache-dir` are startup errors: the session tier was removed
  with the multi-store cache.
- `kvmem.conversation_id` in a request is accepted and ignored.

The pool's decisions live in `tools/kvmem-pool-policy.h`, freestanding, and the
server executes the plan it returns. `tests/conversation-pool-test.cpp` runs
without a model or GPU and pins both production incidents above: each case
also runs the pre-fix logic and shows it fails.

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

## State of this branch

`master` is upstream master plus the fork's cache revision, and it builds and
passes the host tests on Windows (NVMe off, sm_70):

- `kvmem_store_test`, `pinned_kv_tier_test`, `resident_tags_test` (COW, MTP,
  epochs, owner budget)
- `kvmem-conversation-pool-test` (pool policy, both incidents)
- `kvmem-session-snapshot-test` (RawKvStore snapshot roundtrip, bounds and
  checksum)

The multi-store cache's tests (`kvmem-conversation-store-test`,
`kvmem-session-transfer-test`) went with it. `kvmem-session-snapshot-test`
kept only its RawKvStore part, which still covers the kvmem library; the session
file part, which crashed with 0xC0000409 on Windows on upstream alone, was
removed with the session tier.

The Pool and PIC sections above describe the fork's features. PIC remains off
by default and is not beneficial at its current 512-token segment size; the
pool is on by default and is what the deployment uses.
