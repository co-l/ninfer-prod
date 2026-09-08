# ninfer cache-fix — graceful context-swap fallback

A minimal patch to [ninfer](https://github.com/Neroued/ninfer) that makes concurrent
**multi-session workloads with a RAM hot-KV tier** survive. Verified with
`agent-sim` (cache-pressure) on an RTX 5090 (32 GiB VRAM, 30 GiB RAM):
**4 concurrent 150K-token agent sessions, main contexts returned from RAM with
zero re-prefill.**

## The problem

ninfer's materialization planner picks *where each incoming request's cached
context lives*: on the GPU (device pool) or in the RAM hot-KV tier (offloaded
via pinned host memory). The device pool is VRAM-bound (~440-500K tokens on a
32 GiB card with a 27B model's weights resident), so at 4×150K one main
context always lives in the RAM tier and **swaps back when its session
resumes**.

When the swap requires freeing device space under peak pressure (device *and*
host both full), the planner's **bounded search** can fail to find the
demote-one-owner / restore-this-owner plan. Its fallback was catastrophic:

```cpp
selected_maximal_fallback  // release the ENTIRE inactive cache
```

— dropping every checkpoint (all sessions' prefixes) instead of evicting one
low-value owner to RAM. Result: massive re-prefills and collapsed caches.

## The fix (2 parts)

### 1. Graceful fallback plan (`graceful_fallback_target`)

New pressure-planner method: start from the *drop-everything* plan (always
feasible), then **greedily retain the most valuable victims** — keep on
device when it fits, demote to the host tier when it doesn't — as long as the
admission stays feasible. This always yields a minimal-damage plan.

The engine now builds this plan after every search and seals it whenever it
beats the incumbent, so a failed search degrades gracefully instead of nuking
the cache. (Runs even when the search succeeded — it only replaces the plan
when strictly cheaper.)

### 2. Search budget that matches reality

The materialization search space explodes combinatorially (per-victim eviction
choices, not "12 contexts"), so the stock 5 ms budget could evaluate only a
handful of targets at 4×150K scale:

| Constant | stock | patched | why |
|---|---|---|---|
| search time cap | 5 ms | 5 s | budget = `min(cap, incumbent_cost/20)`; normal requests still stop early via value-of-next-expansion |
| `kTargetBudget` | 4096 | 262144 | big-owner expansions fan out thousands of successors |
| `kGuidedBeamWidth` | 16 | 64 | broader guided-closure coverage |
| `kGuidedAssessmentBudget` | 32 | 256 | more closure candidates assessed |
| `kOptionalTargetCapacity` | 4096 | 262144 | sizes the target arena (same fan-out reason) |

## Build

Apply to a stock ninfer checkout, then build the image (rootful podman in the
reference setup; `--parallel 4` in the Dockerfile keeps the build under the
30 GiB RAM ceiling):

```bash
cd <ninfer-src>
git apply ninfer-cache-fix.patch
sudo podman build -t ninfer:local .
```

## Serve configuration (`ninfer-serve.sh`)

The reference launch flags that make 4×150K work, with rationale:

```
--kv-capacity 480000        # explicit device pool (auto picks ~440K; ~480K is the
                            # VRAM-safe max: 7904 page groups = 505,856 tokens is
                            # 200 MB over budget on this box)
--host-kv-mib 12288         # 12 GiB pinned host-KV tier (see memory note)
--host-state-slots 96       # host state images (96 x ~144 MiB ≈ 13.8 GiB pinned)
--device-state-slots 4      # 6 total device state images (C=2 + 4)
--max-concurrency 2
--max-private-continuations 64
--max-shared-prefixes 64
--kv-dtype nvfp4 --spec mtp --draft-tokens 4 --lm-head-draft --vision
```

No container memory limit is set on purpose: the workload's pinned footprint
(state 13.8 GiB + KV 12 GiB + process) plus the weights' page cache genuinely
needs the whole machine; any tighter cgroup limit causes mid-run OOM-kills.

## Memory note (the state tier is the hidden cost)

The RAM footprint is dominated by **two pinned arenas**, not the weights
(which live in VRAM + reclaimable page cache):

- host **state**: 96 slots × ~144 MiB = **13.8 GiB** (per-continuation
  recurrent-state images — the Gated-DeltaNet state)
- host **KV**: up to **12 GiB** (≈651K tokens at 18.4 KB/token)

Sweep result: state slots below 96 or KV below ~10 GiB breaks finalize
survival (4×150K needs ~380K tokens of RAM overflow + all concurrent states).
A "20 GiB RAM" configuration is therefore **not achievable** for this
workload; ~26-28 GiB is the practical floor. The one unused lever is the
weights' ~21.5 GB page cache: `posix_fadvise(POSIX_FADV_DONTNEED)` after
startup would reclaim it and make a cgroup limit viable.

## Results (agent-sim, 4 sessions × 150K main + 2×40K subs each, NVFP4 KV)

- Finalize survival (main context back from RAM after sub-agent work): **4/4**
  in ~90% of runs (occasional 1-2 finalizes re-prefill when a peak-pressure
  event drops a main — see "Known limitation").
- Main-continuation reuse: 116/116 (100%) in clean runs.
- Zero `selected_maximal_fallback` cache nukes.
- Transfer baseline: RAM→VRAM restore ≈ **1.1 s for 123K tokens**
  (~110K tok/s); VRAM→RAM demote is async and adds ~0 to the critical path.
- Cold 123K prefill for reference: ~35 s.

## Known limitation

At absolute peak (device + host simultaneously at capacity) a main can still
be dropped instead of demoted, cascading into a re-prefill of that session's
next turn and, occasionally, a finalize. The graceful fallback minimizes the
damage but doesn't yet synthesize its own demote-to-host decisions — that's
the next step (generate the demote decision in the fallback instead of relying
on the search having populated it).

## Verify

Use `cache-pressure`'s `agent_sim` against the served endpoint:

```bash
uv run python -m cache_pressure.agent_sim \
  --base-url http://<box>:8000/v1 --sessions 4 \
  --main-tokens 150000 --sub-tokens 40000 --sub-windows 2 \
  --ninfer-log <request-log>.jsonl
```

Ground truth for reuse comes from ninfer's `--request-log-jsonl`
(`prefix_cache_hit_tokens` + `prefix_reuse_path`); the API's `cached_tokens`
conflates device-hits with RAM restores.
