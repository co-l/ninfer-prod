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

Supporting changes:

- `kTargetBudget` / `kOptionalTargetCapacity` raised to 262144 (arena sizing
  for big owner sets; the planner itself no longer searches).
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

## Known limitation — engine-level demote-execution data loss (open)

Both validation benches (cache-pressure retention **and** agent-sim 4×150K)
currently fall short because of an issue **below the planner**: after a
demote-heavy plan executes, the demoted contexts lose fast reuse — re-queries
show `cache 0%` / full re-prefill, and the cache drops from ~94 to ~10
contexts in a single admission. The planner's plan says "keep 94, demote 4",
but the executed cache does not retain the demoted contexts on the host tier
for prefix/continuation reuse.

This reproduces on every pressure-capable build (stock search, the previous
graceful build, and this one), so it predates the planner rewrite and points
at the pressure *execution* path (demote → host-tier store → restore on
resume), not at the policy. Investigation notes:

- Host arenas are allocated and pinned ("host state pinned 13.8 GiB", "host KV
  pinned 12.0 GiB").
- A single 8K context's host restore is well under the bench's 0.54 s hit
  threshold, yet misses show full re-prefill timing — the KV is not found on
  the host tier at all.
- Suggested next step: instrument `publish_pressure_*` / the demote commit to
  confirm the host copy is actually stored (and not released on the next
  admission), then verify the host→device restore path is consulted for
  prefix reuse on fresh requests, not only for active continuations.

Until that is resolved, the deterministic planner is a strict improvement at
the policy level (fast, deterministic, no nukes) but does not by itself make
the two benches pass end-to-end.

## Verify

Use `cache-pressure`'s `agent_sim` and `cache_pressure` against the served
endpoint:

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
