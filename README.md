# ninfer cache-fix — deterministic value-ranked planner (sort, not search)

A patch to [ninfer](https://github.com/Neroued/ninfer) that replaces the
bounded combinatorial *eviction search* in the materialization planner with a
**deterministic, value-ranked assignment policy**. No search, no budget, no
catastrophic fallback: every pressure decision is an O(n log n) sort of the
cached contexts by value (hit count, shared credit, retention weight,
recency), then a waterfall that assigns the highest-value contexts to the
device pool, the next to the host tier (state + KV), and drops only what fits
nowhere.

## The problem

ninfer's materialization planner decides where each incoming request's cached
context lives: GPU (device pool, ~440-500K tokens on a 32 GiB card with a 27B
model resident) or the RAM hot-KV tier (pinned host memory). Under peak
pressure (device and host simultaneously full) the stock planner's **bounded
search** had three failure modes:

1. **5 s search tax** — every pressure request ground the search budget to
   prove a trivial plan (hydrates took 6 s TTFT).
2. **Drop-1 churn** — when the device was over by more than one context's
   worth, the greedy couldn't *chain* demotes, so it dropped one context per
   admission (a sliding window; cache-pressure retention 1/65).
3. **Root-maximal nuke** — the search's failure fallback released the ENTIRE
   inactive cache (`selected_maximal_fallback`), collapsing every session's
   prefixes.

All three come from the same disease: *eviction as search*.

## The fix: `deterministic_target` (sort, not search)

New pressure-planner method that builds exactly one plan, deterministically:

1. **Rank** every victim by value — `preferred_owner_ids` is already sorted by
   `selected_hit_count`, shared credit, retention weight, recency (least
   valuable first).
2. **Phase 1 — device waterfall.** While the device is over-committed, demote
   the lowest-value victims whose retained, non-evicting decision strictly
   reduces device over-commitment. Demotes chain (several victims can be
   demoted to close a deficit no single demote covers), and Host impact is
   *deferred* — a momentarily full Host tier does not block a device fix.
3. **Phase 2 — Host payoff.** While the plan (or Host) is still over-committed,
   drop the state checkpoints of the lowest-value victims (the context's KV
   stays cached, its prefix stays reusable) until the residual closes. This
   pays the Host debt deferred by phase 1.
4. **Phase 3 — evict.** Last resort: evict the lowest-value victims until the
   plan is feasible (both tiers full, everything retainable retained).

The engine seals the resulting plan directly — no queue, no beam, no guided
closure, no budget, no fallback races. `deterministic_target` is always
feasible-or-drops-to-minimum.

### Host-feasibility is the allocator's answer, not the byte model

The planner's internal byte model of the Host tier is optimistic in two ways
the real extent allocator is not, and both caused the same failure (a plan
declared feasible that compose's allocator then rejected → `blocked_host` →
every candidate infeasible → maximal fallback → root re-prefill):

1. **It credits Host frees the hot tier never releases.** Restores keep the
   host copy (that is the RAM hot tier's point), so a "restore main, demote
   others" plan cannot count the main's host pages as freed. The byte model
   did; the allocator didn't (run 44: 2.79 GiB demote demand vs 2.48 GiB
   free, planner residual host=0).
2. **It is blind to the Host the plan itself frees by eviction.** An eviction
   is deterministic (the victim's host extents are destroyed), unlike a
   `DropHostDuplicate` which the hot tier may never materialize.

`projected_residual` therefore asks the real allocator
(`program->host_kv_requests_fit`, a conservative no-releases check of the
current free extents) and accepts a plan when **either** the allocator fits
**or** the plan's own evictions free enough Host to cover the demote demand
(`added_host_kv <= host_free + host_freed_by_evictions`). When neither holds,
the residual carries the true Host demand as pressure and Phase 3 converts
demotes to evictions (same device relief, zero Host demand) instead of
compose failing the whole plan. This is what makes the finalize of the
last session survive at full occupancy: it evicts the already-finalized
dead-weight mains instead of being blocked by a full Host tier.

Supporting changes:

- `kTargetBudget` / `kOptionalTargetCapacity` raised to 262144 (arena sizing
  for big owner sets; the planner itself no longer searches).
- `ProgramImplCore::host_kv_requests_fit` — real-extent feasibility check
  used by the planner (mirrors compose's Host admission, conservative).
- The old `graceful_fallback_target` / `guided_closure_target` methods and the
  expansion machinery are retained in the session API but no longer called by
  the planner (dead code, kept to minimize the patch's surface).

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

The reference launch flags, with rationale:

```
--kv-capacity 480000        # explicit device pool (auto picks ~440K; ~480K is the
                            # VRAM-safe max: 7904 page groups = 505,856 tokens is
                            # 200 MB over budget on this box)
--host-kv-mib 12288         # 12 GiB pinned host-KV tier (see memory note)
--host-state-slots 96       # host state images (96 x ~144 MiB ≈ 13.8 GiB pinned)
--device-state-slots 4      # 6 total device state images (C=2 + 4)
--max-concurrency 2
--max-private-continuations 128   # >64: 65x8K working set needs >64 catalog slots
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
- host **KV**: up to **12 GiB** (≈651K tokens at ~18.4 KB/token)

At 65×8K cached contexts each context carries an endpoint + rewrite state, so
the state tier (device 6 + host 96 = 102 slots) is the binding constraint, not
KV: the state waterfall (phase 2) exists precisely so a full Host state tier
never forces an eviction when dropping a redundant state checkpoint suffices.

## Planner-level behavior (verified on the RTX 5090 box)

- Hydrate TTFT stays ~1.1 s (no 5 s search tax).
- Steady-state retention-bench plans are `demoted=4 dropped=0` — chained
  demotes instead of the old `demoted=1 dropped=1`.
- Under 4×150K agent-sim load the planner emits `dropped=0` plans throughout
  (up to 17 chained demotes at peak), zero `selected_maximal_fallback`.
- Deterministic and O(n log n); planning_elapsed stays in the single-digit ms
  range even at 200+ victims.

## Verified results (2026-09-11, clean production build)

**Both benches pass on the same build, and agent-sim is stable across runs:**

- **cache-pressure retention: 65/65 (108.46%)** — hydrate TTFT ~1.1 s, all 65
  contexts restorable from the RAM tier in reverse order.
- **agent-sim 4×150K: 4/4 finalizes at ≥98.6% reuse** (`private_turn_closure`),
  mains ~96.5% at every step (the natural 5K-token step growth), **zero
  `selected_maximal_fallback`**, in 4 consecutive clean runs (46–48 + the
  clean-build re-verify). The historical 2/4 coin-flip is gone: the finalize
  of the last session now evicts the already-finalized dead-weight mains
  (deterministic eviction frees) instead of being blocked by the full Host
  tier and falling back to a root re-prefill.

## Verify

Use `cache-pressure`'s `agent_sim` and `cache_pressure` against the served
endpoint (restart the server between benches — the caches accumulate):

```bash
uv run python -m cache_pressure.agent_sim \
  --base-url http://<box>:8000/v1 --sessions 4 \
  --main-tokens 150000 --sub-tokens 40000 --sub-windows 2 \
  --ninfer-log <request-log>.jsonl

uv run python -m cache_pressure --base-url http://<box>:8000/v1 --kv-size 480000
```

Ground truth for reuse comes from ninfer's `--request-log-jsonl`
(`prefix_cache_hit_tokens` + `prefix_reuse_path`); the API's `cached_tokens`
conflates device-hits with RAM restores.
