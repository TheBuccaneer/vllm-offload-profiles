# Fix D interpretation notes

This is a small control block, not a new headline result. It asks whether generic GPU-only background load can reproduce the API-visible decode-timing regime that CPU offloading (12GB) induces in the profile-robustness block.

## llama
- concurrency=4:
  - GPU-only control did NOT reach a comparable median ITL to cpu_offload12 under the tested load grid (ratio=0.02).
  - median TTFT shift under load: 63.1ms -> 96.6ms (check whether delay moved into waiting/TTFT/tails instead of decode ITL).
  - Finding: generic GPU-only load did not reproduce the offload-level decode slowdown under the tested load grid; delay shifted elsewhere or the decode path saturated differently.
- concurrency=8:
  - GPU-only control did NOT reach a comparable median ITL to cpu_offload12 under the tested load grid (ratio=0.02).
  - median TTFT shift under load: 65.6ms -> 96.1ms (check whether delay moved into waiting/TTFT/tails instead of decode ITL).
  - Finding: generic GPU-only load did not reproduce the offload-level decode slowdown under the tested load grid; delay shifted elsewhere or the decode path saturated differently.

## qwen
- concurrency=4:
  - GPU-only control did NOT reach a comparable median ITL to cpu_offload12 under the tested load grid (ratio=0.01).
  - median TTFT shift under load: 60.3ms -> 73.7ms (check whether delay moved into waiting/TTFT/tails instead of decode ITL).
  - Finding: generic GPU-only load did not reproduce the offload-level decode slowdown under the tested load grid; delay shifted elsewhere or the decode path saturated differently.
- concurrency=8:
  - GPU-only control did NOT reach a comparable median ITL to cpu_offload12 under the tested load grid (ratio=0.02).
  - median TTFT shift under load: 71.5ms -> 117.4ms (check whether delay moved into waiting/TTFT/tails instead of decode ITL).
  - Finding: generic GPU-only load did not reproduce the offload-level decode slowdown under the tested load grid; delay shifted elsewhere or the decode path saturated differently.
