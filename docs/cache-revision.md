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

## PIC remains incomplete

The branch contains independently testable GDN transition/composition, capture,
state, packed RoPE relocation, KV preparation and segment-store components.
`tools/kvmem-pic-server.h` is an excluded integration draft. Adapter commit,
complete transaction/MTP integration and end-to-end quality validation remain
unfinished. `--kvmem-pic on` fails explicitly; off is the default. These components
must not be described as a functioning position-independent cache.

## Validation

The source baseline was built for SM70 with MSVC and CUDA 12.9. The deployed
executable SHA256 was
`BE1B822256363AAD4070D7CB3EB792E66E7A68CA0488767BFF5CFBF9C3A4B031`.
That identifies the recorded deployment artifact, not a promise of reproducible
binary hashes across build environments or Git metadata.

- Host raw/COW and resident-tag tests passed, including a 507+6 ubatch-tail case.
- PIC store and GDN/RoPE CPU numerical tests passed.
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
