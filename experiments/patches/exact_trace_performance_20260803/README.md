# LS4 file-window patch archive

Date indexed: 2026-08-03

This directory preserves the rejected `cuda_file_windowed` LS4 experiment as
mailbox patches. The mode was `strict_exact` on the completed 4B/256 trace, but
it is not retained in runtime code because both implementations are dominated
by an existing neighbor. These patches are recovery material, not an active or
sequentially selected configuration.

## Index

| Patch | Repository | SHA-256 | Purpose / result |
|---|---|---|---|
| `sibling/0001-Add-file-backed-CUDA-row-windows.patch` | `circuit-tracer_chunked` | `3e63b1058e241fa9a99d616d8333fbb1dc273404ec1c4778cfe5cf0bb54c09bd` | Adds a selectable canonical-memmap -> bounded pinned-host -> bounded CUDA provider. The full 4B/256 v1 trace was strict-exact, but took 189.64 s versus 129.40 s for mirrored `cuda_windowed` and peaked at 198.54 GiB cgroup usage. |
| `sibling/0002-Stream-file-windows-directly-into-pinned-memory.patch` | `circuit-tracer_chunked` | `95a21efe5899ed7ca5b5d89fda8be06f808993764187bb4c72c71de3b4d6091d` | Replaces mapped slicing with `preadv` into the pinned buffer and drops pages after reads. Process mapped-file RSS fell, but Phase 4 grew to 371.35 s; the 200 GiB guard fired at 201.05 GiB before packaging, so no scientific artifact was produced. |
| `project/0001-Register-LS4-file-window-candidate.patch` | `nlp_research_project` | `acb9b0e41c96a24d6aa9a538576656f68abde29c811dd96f6db25f090a653a7e` | Restores the explicit LS4 profile and registry tests. It does not change defaults or automatic mode selection. |

## Selection decision

The v1 provider removed the 6,186,403,212-byte signed CPU mirror from process
ownership, but repeated mapped reads accumulated as file cache. It was 46.6%
slower than mirrored CUDA and did not lower measured total cgroup usage. The v2
provider bounded the process file mapping as intended, but traded the cached
path for synchronous storage traffic: GPU utilization collapsed and Phase 4
was 2.28x slower than `cpu_exact` (371.35 versus 162.78 seconds). Its guarded
wrapper wall was 433.41 seconds and it crossed the declared host envelope.

Therefore neither version has a non-dominated runtime role. The existing
mirrored `cuda_windowed` mode remains selectable at its admitted 4B/129--256
scope, and `cpu_exact` remains the arbitrary exact route. Larger 4B/12B traces
must use the ordinary per-trace job machinery with a suitable RAM request;
they must not be forced into the current interactive allocation.

## Restoration

Start from the post-removal branches. Inspect every patch first. In the sibling
repository, apply the two sibling patches in numeric order with `git am`. Then
apply the project patch from the project repository. Re-run the focused row
store and campaign tests and repeat guarded performance measurement before
considering retention. The v1 result comparison is stored at
`ls4-4b-transfer-v1/comparisons/256-cpu-exact-vs-cuda-file-windowed-v1.json`
under the campaign scratch root; the v2 `trace_results.jsonl` is empty because
the resource guard stopped the run before packaging.
