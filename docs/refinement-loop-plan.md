# Efficient 50–100 Game Strategy Refinement Loop

This plan replaces manual 10-game trial-and-error with a decoupled batch workflow:

**Unattended 50–100 Game Farm → Automated Retention → Deterministic `< 30 KB` Batch Digest → Single LLM Review & Patch → Offline Counterfactual Gates → Paired SPRT.**

The phone and GPU run headless without waiting on an LLM; deterministic code compresses 50–100 matches (~9–18 GB of raw data) into one high-signal failure dossier; one LLM pass diagnoses *all* recurring failure clusters at once and ships a candidate strategy; and offline gates prove the candidate fixes those failures before risking another hour on live matches.

---

## 1. Current bottlenecks (measured in Batch 1 & Batch 2)

1. **Statistical noise in 10-game batches:** At `N = 10`, Wilson 95% confidence intervals span ~40–53 percentage points (`ladder-vision-v4` `9W–1L` = `[0.596, 0.982]`; `ladder-vision-v5` `5W–5L` = `[0.237, 0.763]`). Detecting a real `+6%` to `+8%` win-rate gain reliably needs ~50–100+ games or paired blocks.
2. **The 89% triage blackout in [`flywheel/triage.py`](../src/droid_harness/flywheel/triage.py):**
   - `PERCEPTION_WARNINGS = {"elixir_discrepancy", "hand_unreadable", "affordability_mismatch", "tower_hp_increase"}` with `perception_warning_frames = 2`.
   - Every time the bot plays a card, the slot animates for ~300 ms (~15–18 frames) and emits transient `hand_unreadable` warnings. Any engagement touching a play transition crosses `>= 2` frames and is tagged `PERCEPTION` instead of `STRATEGY` or `OK`.
   - Measured result in live runs: Batch 1 had `triage_mix: {PERCEPTION: 0.904, OK: 0.089, STRATEGY: 0.007}` and Batch 2 had `{PERCEPTION: 0.889, STRATEGY: 0.034, OK: 0.077}`, leaving `play_value_by_rule` with only **1 scored decision** (`cycle.cheap_cycle: n=1`) across 10 games. That blinds both `analyze` and `flywheel/loss_stories.py`.
3. **Raw video disk growth at 50–100 games:** Captured `stream.h264` grows at ~60 MB/min (~180 MB per 3-minute game, **9–18 GB** for 50–100 games). By contrast, compact logs (`meta.json` + `summary.json` + `decisions.jsonl`) take ~1.5 MB/match (confirmed `15 MB` total across Batch 2's 10 matches, ~75–150 MB for 50–100 matches).
4. **Matchmaking & trophy-climb confounding across sequential batches:** Running `v4` for 10 games climbed trophies (`9W–1L`), pushing `v5` into higher-king/card-level opponents (`Lv4–Lv5`). Sequential unpaired batches mix strategy deltas with ladder climb.
5. **No single-command failure aggregator yet:** Batch 2's regression (`5W–5L` vs `9W–1L`) took a custom Python inspection of `decisions.jsonl` to catch that `t=1.1–1.4s` back-Giants at `7.2–7.5` elixir caused three ~95 s `0–3` losses (`98b80f1060b2`, `d1485a87825e`, `d88cb32961f4`). At 100 games, reading raw `decisions.jsonl` manually is impossible without deterministic clustering.

---

## 2. Locked-In 3-Speed Architecture (Agent-Harness Driven)

Instead of generating a *batch* of LLM strategies to test on the phone (which hits multi-day SPRT wall-clock limits or small-sample Winner's Curse) or using an LLM to guess continuous floats, strategy optimization runs at **three decoupled speeds**:

1. **Speed 1 — Offline Counterfactual Search (`droid-harness tune-offline` + `shadow`, CPU seconds, 0 Phone Games, 0 LLM):**
   - Samples 100–500 continuous parameter candidates around the champion and scores each across all logged matches using true `Engagement.value` (now including calibrated `digits/phone-a/` tower HP deltas and post-fix `triage.py`).
   - Enforces a `replay-diff` change guardrail (`1–18%`) and selects **only the single #1 candidate** that flips losing engagements without breaking winning ones.
2. **Speed 2 — Persistent Two-Stage Live Paired Gate (`droid-harness experiment`, Phone hours, 0 LLM):**
   - Runs inside **one persistent `BotRunner`** (cached ROCm `TorchDetector` weights + single `scrcpy` stream, hot-swapping `CRPolicy` per match) in randomized 2-game blocks (`--test pairs`) to cancel ladder trophy/level drift.
   - **Stage A (Dense-Margin Futility Stop, `--futility-pairs 12`):** Scores all pairs (including `W–W` and `L–L`) by crown + `tower_hp_diff` + speed margin and aborts losing candidates early when `UCB95 < 0`.
   - **Stage B (Paired SPRT Promotion):** Candidates surviving Stage A run to paired SPRT acceptance/rejection in [`ledger/STRATEGY_LEDGER.md`](../ledger/STRATEGY_LEDGER.md).
3. **Speed 3 — On-Demand Structural Agent Handoff (`droid-harness digest`, 1 Agent Turn Only When Plateaued):**
   - No internal LLM SDK loop is needed: `droid-harness digest --data data/matches --out data/agent_digest.md` emits a deterministic `< 30 KB` **Agent Handoff Artifact** (+ `.json`) for the external coding-agent harness (OpenCode / Antigravity) to add **1 structural rule + unit test** when failure clusters show a missing mechanic, then hands numerical tuning back to Speed 1.
4. **Endgame Tie-Breaker — M8 v2 Board-Aware Residual Ranker (`learn/label_from_video.py --version 2` + `train_ranker.py`):**
   - Encodes per-lane board value, threat ETA, and opponent cycle (`droid-harness-ranker/2`), trains with engagement-credit Advantage-Weighted / Filtered BC (`--cards ... --min-credit -0.5`), and blends via bounded residual (`ranker_mode: residual`) without overwriting safety/counter gates.

---

## 3. Prerequisite code fixes (before farming 50–100 games)

### Fix A — Stop card-play transitions from poisoning `triage.py`

In [`src/droid_harness/flywheel/triage.py`](../src/droid_harness/flywheel/triage.py), update `triage()` so routine card-play slide transitions no longer swipe 89% of engagements into `PERCEPTION`:

1. Collect timestamps `play_t_ms` of confirmed own plays (`tl.plays` with `side == "me"`).
2. Ignore `hand_unreadable` and `affordability_mismatch` warnings within `[play_t_ms - 50, play_t_ms + 500]` ms of a confirmed play (the ~300 ms card-slide window).
3. Require **sustained** perception corruption rather than 2 random frames: mark `PERCEPTION` only when **either** `elixir_discrepancy` / `tower_hp_increase` appears `>= 3` times in the engagement, **or** non-transition `hand_unreadable` / `affordability_mismatch` affects `>= 10` frames **and** `>= 15%` of the engagement's sampled frames.
4. Keep `eval/inclusion.py` unchanged: win/loss SPRT statistics always count all completed matches regardless of triage tag.

### Fix B — Retention & compaction (`droid-harness retain`)

Add a subcommand (and optional `--prune-win-video` flag on `run`) so 50–100 game runs stay small without losing diagnostic evidence:

- Keep `meta.json`, `summary.json`, `decisions.jsonl`, and DuckDB `warehouse.duckdb` rows for **100%** of matches (~1.5 MB/match).
- On clean completed wins (`result == "win"` and no `LATENCY`/`EXECUTION` flags), delete raw `stream.h264`, `stream.pts`, and `frames.jsonl` once `analyze` (and optional `frames` training export) has ingested them.
- On losses, draws, or flagged `LATENCY`/`EXECUTION` engagements, extract 1 keyframe 1 s before each lost tower + settled result frame (`~50 KB` JPEG/PNG each) and optionally transcode `stream.h264` to 1 Mbps H.265 or prune after 7 days.
- Net disk budget per 100-game campaign: **~150–400 MB** instead of **18 GB**.

---

## 4. Exact spec for `droid-harness digest` (what the LLM actually reads)

When you hand 50–100 matches to an LLM, **never** feed raw `decisions.jsonl` or `frames.jsonl`. Add `droid-harness digest --data <dir> --db data/warehouse.duckdb --out data/digest.md` to emit a single `< 30 KB` report containing 6 deterministic sections:

1. **Batch Health & Drift Slices (chronological quartiles: Q1..Q4):**
   - Match count, W–L–D + Wilson 95% CI per quartile (detects trophy-climb difficulty shifts across a 100-game session).
   - Crown distribution (`3-0`, `1-0`, `0-1`, `0-3`, etc.), median duration, `0` vs non-zero aborts, `loop_p50_ms` / `loop_p99_ms`, retry rate, `opp_deck_seen == 8` lock rate, and post-fix `triage_mix`.
2. **Opponent Card & Archetype Matchup Matrix:**
   - Group matches by cards observed in `opp_deck_seen` (already populated in 10/10 recent matches): win rate and average crown differential when facing `giant`, `mini_pekka`, `valkyrie`, `barbarians`, `goblin_cage`, `tombstone`, `prince`, `baby_dragon`, `arrows`, `fireball`. Tells the LLM immediately which enemy cards beat our current strategy.
3. **Rule & Posture Scorecard:**
   - From `flywheel/report.py::batch_report` (now unblocked by the `triage.py` fix): per-`chosen_rule` and per-`posture` play count, share of total plays, mean engagement value, total engagement value, and mean `(my_elixir, opp_elixir)` when triggered.
4. **6 Automated Failure-Pattern Detectors (computed deterministically across all 50–100 `decisions.jsonl`):**
   - **Cluster 1 — Opening Overcommitment (`opening_throw`):** `t_ms < 15_000` plays that leave `my_elixir < 3.0` while `opp_elixir[1] >= 6.5`, followed by taking tower damage or losing first tower before `t = 60s`. *(Would have flagged 3/5 losses in Batch 2 automatically.)*
   - **Cluster 2 — Win-Condition Starvation (`wincon_starved`):** Matches where `giant` (or primary `win_conditions`) was played `<= 1` time despite lasting `>= 120s`, paired with share of frames where `my_elixir >= 5.0`.
   - **Cluster 3 — Low-Elixir Defense Bleed (`over_defend_bleed`):** Matches where `my_elixir < 5.0` for `> 75%` of frames due to spending `3–4` elixir on repeated minor threats.
   - **Cluster 4 — Wasted / Un-thrown Spells (`spell_inefficiency`):** Spell casts yielding negative engagement value alongside frames where `>= 3` clumped enemies existed untouched by `arrows`/`fireball`.
   - **Cluster 5 — Bad Cycle Timing vs. Tracked Counters (`bad_counter_timing`):** Win-condition plays where `opp_cards_until_back(counter) == 0` AND `opp_elixir[1] >= cost(counter)` vs. win-condition rate when counters were out of cycle (`>= 2` away).
   - **Cluster 6 — Card Utilization Skew & Unused Options:** Per-card play count, average hold time in hand, and fraction of plays triggered by `tier0.leak_guard`.
5. **Top 8–12 Richest `STRATEGY` Loss Stories (upgrading [`loss_stories.py`](../src/droid_harness/flywheel/loss_stories.py)):**
   - Extend `loss_story()` lines from `t opp/we card @ tile rule play_value` to also print `(my_el -> after | opp_el[lo,hi] | opp_hand_ready)` and the **top-2 losing/runner-up candidate scores** from `decisions.jsonl` (`why Giant wasn't picked` or `why defense fired early`).
6. **Optional Visual Pack (`<= 5` keyframes, `< 250 KB` total):**
   - One annotated arena crop from 1–2s before a lost tower falls across the top 3–5 unexplainable losses (only if a perception blind spot is suspected).

---

## 5. Offline screening + paired SPRT (zero wasted phone hours)

Once the LLM reads the `< 30 KB` digest and produces a single consolidated candidate (for example upgrading rules + YAML parameters + counter weights together in `ladder-vision-v6.yaml`), run **three fast offline checks before touching `phone-a`**:

1. **Invariant tests (`pytest -ra && ruff check src/ tests/`):** Proves rules behave as intended on scripted board states (`opponent_model`, `offense`, `defend`, `spells`, `cycle`).
2. **Whole-batch counterfactual check (`droid-harness replay-diff`):** Run across all 50–100 logged `decisions.jsonl` files (`--data data/batch50`). Inspect `by_champion_rule` and `examples` to verify:
   - Bad cluster triggers drop as expected (for example `t=1.1s` naked Giants go to `0`).
   - Emergency and core defenses are preserved.
3. **Engagement-weighted shadow report (wire [`flywheel/shadow.py`](../src/droid_harness/flywheel/shadow.py) into CLI):** Run candidate vs. champion across all 50–100 matches and sort disagreements by engagement `value` (`flywheel/shadow.py::report()` already aggregates disagreements × `value`!). A good candidate should disagree predominantly on **negative-value** (losing) engagements, not positive-value winning ones.
4. **Live validation (`droid-harness run-experiment --test pairs`):** Run 10–30 randomized pairs (20–60 matches alternating champion vs. candidate) on `phone-a`, canceling trophy/arena drift cleanly. Every experiment records into [`ledger/experiments.jsonl`](../ledger/experiments.jsonl) and [`ledger/STRATEGY_LEDGER.md`](../ledger/STRATEGY_LEDGER.md) so neither a human nor an LLM re-proposes rejected changes.

### Synergy with M8 Learned Policy & M6 Optuna

Every 50–100 win farm batch pays twice:
- Once as diagnostic fuel for the deterministic digest + LLM rule/strategy patch.
- Once as clean labeled training data for [`learn/label_from_video.py`](../src/droid_harness/learn/label_from_video.py) → [`learn/train_ranker.py`](../src/droid_harness/learn/train_ranker.py) (M8 tie-breaker ranker once 50+ wins are collected) and numerical fine-tuning via [`flywheel/tuning/optuna_study.py`](../src/droid_harness/flywheel/tuning/optuna_study.py).

---

## 6. Implementation roadmap

| Phase | Scope & files | Deliverable |
| --- | --- | --- |
| **Phase 1 (Immediate high ROI)** | `flywheel/triage.py`, new `flywheel/digest.py`, `cli.py` (`digest`, `retain`) | Fixes the 89% `PERCEPTION` triage blackout; turns any 50–100 match directory into a `< 30 KB` failure dossier + safe disk pruning |
| **Phase 2 (Richer offline signal)** | `flywheel/loss_stories.py`, `flywheel/shadow.py`, `cli.py` (`shadow`), `flywheel/coach/coach.py` | Adds elixir/cycle context & runner-up candidates to `loss_story()`, adds `droid-harness shadow`, and wires `droid-harness coach --from-digest` |
| **Phase 3 (Closed unattended flywheel)** | `cli.py` (`run-experiment`), `flywheel/tuning/optuna_study.py`, `learn/train_ranker.py` | Overnight paired SPRT gates, Optuna parameter sweeps around winning candidates, and M8 ranker training on 50+ wins |
