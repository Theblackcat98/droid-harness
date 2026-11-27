# droid-harness

A deterministic, low-latency Android automation and Clash Royale harness that runs **without any LLM in the loop**: scrcpy H.264 stream → computer-vision perception → rule-based utility policy → timed ADB/scrcpy touches.

Every strategy parameter, threshold, counter table, and rule lives in versioned YAML/CSV files under [`configs/strategies/`](configs/strategies/), while the analytics flywheel (`analyze`, `replay-diff`, `run-experiment`, `sprt`, `ledger`) lets you measure, tune, and promote strategies statistically using only local Python CLI tools.

> **Account scope.** Supercell's terms treat bots as cheating and can ban accounts. Use a throwaway account (Training Camp, Practice, or two-phone friendly battles; ladder only on throwaway accounts you are willing to lose). The general Android core carries none of this risk.

---

## Current status (verified on `phone-a`)

| Component | Current measured state |
| --- | --- |
| **Transport & loop** | `scrcpy-server-v2.4` over USB + PyAV H.264 decode; battle loop `2.51–4.76 ms` `p50` / `3.95–12.38 ms` `p99` with `--async-detect` |
| **31-class YOLO26-n v2 detector** | Trained across 31 classes ([`models/classes.txt`](models/classes.txt)): `mAP50=0.728`, `mAP50-95=0.474`, `P=0.742`, `R=0.632`; PyTorch-ROCm `TorchDetector` `4.14 ms` `p50` (`4.23 ms` mean) on RX 7900 XT |
| **Opponent & crown perception** | Card/cost/cycle tracker locks 8-card opponent decks across Arena 1–3 (`10/10` matches in both recent batches); Arena 2 golden-sand false crown bug fixed in [`perception/screens.py`](src/droid_harness/domains/clash_royale/perception/screens.py) |
| **Perception additions** | Tower HP digits (`digits/phone-a/`), countdown clock, King HP seeding; triage `PERCEPTION` share ~89% → ~18% on recorded matches |
| **Verified live batches (`phone-a`)** | [`ladder-vision-v4.yaml`](configs/strategies/ladder-vision-v4.yaml) **9W–1L–0D (90.0%)** (champion); [`ladder-vision-v5.yaml`](configs/strategies/ladder-vision-v5.yaml) **5W–5L–0D (50.0%)** (rejected candidate — compacted telemetry in [`data/matches/`](data/matches/)) |
| **Unit tests & lint** | `pytest -ra` all passing (incl. `tests/test_three_speed_flywheel.py`); `ruff check src/ tests/` clean |

---

## 1. Install & smoke-test (no phone needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

pytest -ra
ruff check src/ tests/

# Synthetic closed-loop simulation (~3 ms loop latency, no phone)
droid-harness sim --seconds 10 --out data/sim
droid-harness analyze --data data/sim --cards tests/fixtures/cards_min.json

# View SPRT sample-size table for win-rate testing
droid-harness sprt
```

Optional dependencies live in extras (`[detector]`, `[rocm]`, `[onnx]`, `[tuning]`). You do **not** need `[coach]` unless you run an offline LLM coach.

---

## 2. Run matches on an already-calibrated phone (`phone-a`)

`phone-a` (OnePlus N10 `741be240`) already has checked-in configs under [`configs/`](configs/). Verify ADB sees the device (`adb devices`), unlock the screen onto Clash Royale's Home screen, and run:

### One-command Ladder or Training Camp play (recommended)

`--device phone-a` is the default preset and auto-resolves `profile`, `rois`, `tiles`, `arena`, `templates`, `screens`, `towers`, `digits`, `cards.json`, `models/yolo26n-cr.pt`, `models/classes.txt`, and `--async-detect` (short strategy aliases like `v6` or `v4` also resolve automatically):

```bash
# 1-second pre-flight check (calibration, 10/10 digits, 8/8 deck templates, ROCm GPU, ADB)
.venv/bin/droid-harness doctor --live

# Play 10 ladder matches with pro starter beatdown v6 + auto win-video pruning
.venv/bin/droid-harness play --strategy v6 --mode ladder --matches 10 --prune-win-video

# Persistent-session paired evaluation (v4 vs v6; loads GPU & scrcpy ONCE, Stage-A futility + SPRT)
.venv/bin/droid-harness experiment --champion v4 --candidate v6 --mode ladder --prune-win-video \
  --title "v6 pro rules + tower HP digits" --hypothesis "Eliminate opening throws and add cycle-lock/chip/punish"

# 3-Speed Offline Flywheel commands (0 phone time, 0 LLM calls):
.venv/bin/droid-harness tune-offline --champion v6 --data data/matches --trials 100 --out configs/strategies/ladder-vision-v7-opt.yaml
.venv/bin/droid-harness shadow --champion v4 --candidate v6 --data data/matches
.venv/bin/droid-harness digest --data data/matches --out data/agent_digest.md
.venv/bin/droid-harness retain --data data/matches
```

Explicit paths (`droid-harness run --profile ... --layout ...`) remain supported for custom setups:

```bash
# Explicit-flag invocation (equivalent to `droid-harness play --strategy v4`)
.venv/bin/droid-harness run \
  --profile configs/device_profiles/phone-a.yaml \
  --layout configs/domains/clash_royale/rois/phone-a.yaml \
  --tiles configs/domains/clash_royale/tiles/phone-a.yaml \
  --arena configs/domains/clash_royale/arena/phone-a.yaml \
  --templates configs/domains/clash_royale/templates/phone-a \
  --screens configs/domains/clash_royale/screens/phone-a \
  --towers configs/domains/clash_royale/towers/phone-a \
  --digits configs/domains/clash_royale/digits/phone-a \
  --cards configs/domains/clash_royale/cards.json \
  --strategy configs/strategies/ladder-vision-v4.yaml \
  --detector models/yolo26n-cr.pt \
  --classes models/classes.txt \
  --async-detect \
  --mode ladder \
  --matches 10 \
  --balance-epoch 2026-11 \
  --data data/matches-batch
```

Switch `--mode ladder` to `--mode training_camp` to farm against Trainer bots instead of ladder opponents.

### Key runtime flags

| Flag | Purpose |
| --- | --- |
| `--detector <path>` | `.pt` runs `TorchDetector` (PyTorch GPU/ROCm); `.onnx` runs `OnnxDetector` (MIGraphX EP if available, else CPU). Omit only when running a blind strategy such as [`training-camp-blind-v1.yaml`](configs/strategies/training-camp-blind-v1.yaml). |
| `--classes models/classes.txt` | Cross-checks embedded model class names and fails fast on mismatch. |
| `--async-detect` | Runs object detection on a dedicated background worker so the 60 Hz HUD/policy/touch loop never blocks on inference. |
| `--detect-every N` | Runs detector every Nth frame (`1` default; units extrapolate between frames). |
| `--det-conf` / `--det-iou` | Detection confidence threshold (`0.35`) and NMS IoU (`0.5`). |
| `--matches N` | Plays `N` completed matches unattended (menu navigation, battle queue, end screen OK dismissal, Trophy Road popups). Press `Ctrl+C` to stop cleanly. |
| `--data <dir>` | Output directory. Each match creates `<dir>/<match_id>/` with `meta.json`, `summary.json`, `decisions.jsonl`, `frames.jsonl`, and `stream.h264` + `stream.pts`. |

---

## Improvement loop (what you do to keep improving the bot)

Three speeds: seconds on CPU, hours on the phone, and an agent only when plateaued. No LLM runs inside the harness.

```bash
# 0. Pre-flight before every session (phone plugged in, unlocked, on Clash Royale home)
.venv/bin/droid-harness doctor --live

# 1. Collect games (champion = v4 until v6 is proven)
.venv/bin/droid-harness play --strategy v4 --mode ladder --matches 30 --prune-win-video

# 2. Paired live test: candidate vs champion (randomized pairs cancel ladder drift;
#    clearly worse candidates stop early via dense-margin futility; result -> ledger/)
.venv/bin/droid-harness experiment --champion v4 --candidate v6 --mode ladder --prune-win-video \
  --title "v6 pro rules" --hypothesis "fewer opening throws"
.venv/bin/droid-harness ledger        # renders ledger/STRATEGY_LEDGER.md

# 3. Speed 1 - offline, between phone runs (seconds, no phone, no LLM)
.venv/bin/droid-harness analyze --data data/matches
.venv/bin/droid-harness tune-offline --champion v6 --trials 200 --out configs/strategies/ladder-vision-v7.yaml
.venv/bin/droid-harness shadow --champion v6 --candidate v7
.venv/bin/droid-harness replay-diff --champion v6 --candidate v7 --expected-pct 5
# then repeat step 2 with --champion v6 --candidate v7

# 4. Speed 3 - when progress plateaus: hand an artifact to your coding agent
.venv/bin/droid-harness digest --data data/matches --out data/agent_digest.md
# The agent adds ONE structural rule + unit test in a NEW strategy file (never edit a strategy that has played matches).

# 5. Housekeeping
.venv/bin/droid-harness retain --data data/matches   # prune old win videos
```

Notes:
- `tune-offline` enforces a replay-diff guardrail (`--min-diff-pct`, `--max-diff-pct`); if it finds nothing, raise `--trials`, lower `--min-diff-pct`, or collect more matches.
- Use a new `--balance-epoch` after a game update; experiments are voided across updates.
- Short names resolve automatically (`v6` → `configs/strategies/ladder-vision-v6.yaml`).

**Things only you can do**
- **Card levels:** upgrade cards that survive into the Hog EQ deck (Musketeer, Knight, Fireball); keep Giant/Mini P.E.K.K.A/Archers/Minions/Arrows at minimum. Reach King Level 2 early.
- **New cards after arena unlocks:** capture `affordable` + `greyed` hand templates (`droid-harness template`); `doctor` flags missing deck templates. Then fine-tune the detector (section 6).
- **Hog Rider (Arena 5):** switch to the Hog EQ blueprint in [`docs/pro-strategy-research.md`](docs/pro-strategy-research.md).
- **M8 ranker (after ~50+ post-digit ladder matches):** `python -m droid_harness.learn.label_from_video --version 2 --cards configs/domains/clash_royale/cards.json`, then `train_ranker`, then test it with `experiment` like any candidate.
- Ladder botting can get the account banned; that risk is yours.

Deferred: live GUI / replay inspector (spec and alignment question in [`todo.md`](todo.md)).

---

## 3. First-time setup for a new phone or card database

See [`docs/phone-a-bringup.md`](docs/phone-a-bringup.md) for a worked example. For a new device:

1. **Pin `scrcpy-server-v2.4`** into `vendor/` and create `configs/device_profiles/<phone>.yaml` from [`phone-a.example.yaml`](configs/device_profiles/phone-a.example.yaml) (`serial`, `local_port`, `scrcpy_server_jar`). Each concurrent phone needs a unique `local_port`.
2. **Test the H.264 stream and measure touch round-trip latency:**
   ```bash
   droid-harness stream --profile configs/device_profiles/<phone>.yaml --seconds 10 --save-frame data/frame.png
   droid-harness loopback --profile configs/device_profiles/<phone>.yaml --x 30 --y 400
   ```
   Copy the reported `p50` into `exec_latency_ms` in your device profile. Static screens emit no H.264 frames; tap or animate the screen while testing. To test multi-minute stability without playing:
   ```bash
   droid-harness stream --profile configs/device_profiles/<phone>.yaml --duration 1800
   ```
3. **Generate `cards.json`** (only needed when game card stats change or setting up from scratch):
   ```bash
   git clone --depth 1 https://github.com/RoyaleAPI/cr-api-data ~/src/cr-api-data
   python tools/build_card_db.py ~/src/cr-api-data --out configs/domains/clash_royale/cards.json
   ```
4. **Calibrate ROIs & tile homography:**
   - Copy [`configs/domains/clash_royale/rois/phone-a.yaml`](configs/domains/clash_royale/rois/phone-a.yaml) and measure normalized fractions `[x0, y0, x1, y1]` for hand slots, `next_card`, `elixir_bar`, `clock`, tower HPs, `arena`, and menu buttons (`battle`, `ok`, `dismiss`, `trophy_ok`, etc.).
   - Populate landmark pixel coordinates and fit the pixel-to-`18×32` tile homography (check max landmark error `< 0.5` tiles):
     ```bash
     droid-harness calibrate-tiles --landmarks <landmarks.yaml> --out configs/domains/clash_royale/tiles/<phone>.yaml
     ```
5. **Capture card, tower, and screen templates** from saved frames:
   ```bash
   # Capture BOTH affordable and greyed states for each card in your 8-card deck (plus slot "next")
   droid-harness template --frame <battle.png> --layout <rois.yaml> --slot 0 --card knight --state affordable --out configs/domains/clash_royale/templates/<phone>
   droid-harness template --frame <battle_low_elixir.png> --layout <rois.yaml> --slot 0 --card knight --state greyed --out configs/domains/clash_royale/templates/<phone>

   # Tower alive / rubble states
   droid-harness tower-template --frame <battle.png> --layout <rois.yaml> --tower all --state alive --out configs/domains/clash_royale/towers/<phone>

   # Menu & end screens (HOME, MATCHMAKING, TROPHY_ROAD, POPUP, IN_BATTLE, RESULT_WIN, RESULT_LOSS)
   droid-harness screen --frame <home.png> --name HOME --out configs/domains/clash_royale/screens/<phone>
   ```

---

## 4. Analyze matches & inspect telemetry (offline)

All analytics run deterministically on recorded directories with zero API calls:

```bash
# Segment lane engagements, tag triage causes (PERCEPTION/EXECUTION/LATENCY/STRATEGY/OK), ingest into DuckDB, print report
droid-harness analyze --data data/matches-batch --cards configs/domains/clash_royale/cards.json --db data/warehouse.duckdb

# Plain-text flywheel dashboard (win rate CI by strategy & game version, rule/posture value, triage mix, loop p99)
python notebooks/dashboard.py data/matches-batch
```

### Quick per-match CLI inspection

Each match folder contains:
- `summary.json`: `result`, `my_crowns`, `opp_crowns`, `strategy_id`, `status`, `duration_s`, `elixir_leaked`, `loop_p50_ms`, `loop_p99_ms`, `perception_warnings`, `opp_deck_seen`.
- `decisions.jsonl`: every decision tick with `t_ms`, `posture`, `chosen_rule`, `chosen_card`, `tile`, `score`, `candidates`, and `state` snapshot (`my_elixir`, `opp_elixir`, `hand`, `opp_deck_seen`, `opp_last_four`, `towers`).
- `frames.jsonl` / `stream.h264`: per-frame stage timings and raw captured video.

### Re-running perception on saved videos (`replay`)

To test a new detector checkpoint or perception change offline against past videos without touching the phone:

```bash
droid-harness replay \
  --data data/matches-batch \
  --layout configs/domains/clash_royale/rois/phone-a.yaml \
  --tiles configs/domains/clash_royale/tiles/phone-a.yaml \
  --arena configs/domains/clash_royale/arena/phone-a.yaml \
  --templates configs/domains/clash_royale/templates/phone-a \
  --towers configs/domains/clash_royale/towers/phone-a \
  --cards configs/domains/clash_royale/cards.json \
  --detector models/yolo26n-cr.pt --classes models/classes.txt \
  --out data/replay/run-new.json \
  --baseline data/replay/run-prev.json
```

---

## 5. Edit, screen & promote strategies manually (no LLM)

**Immutability rule:** Never modify a strategy YAML or counter CSV once it has logged matches. Copy to a new version (for example `ladder-vision-v4.yaml` → `ladder-vision-v6.yaml`), update `strategy:` and `parent:`, and point `counters:` at its own CSV. Every strategy gets a 12-char `strategy_id` content hash covering [`policy/`](src/droid_harness/domains/clash_royale/policy/), the YAML, and its counter CSV.

### Strategy file structure ([`policy/strategy.py`](src/droid_harness/domains/clash_royale/policy/strategy.py))

| Section | Key fields | Meaning |
| --- | --- | --- |
| **Identity** | `strategy`, `parent`, `deck`, `win_conditions` | Strategy name, lineage, exact 8 card keys matching [`cards.json`](configs/domains/clash_royale/cards.json), win-condition card(s). |
| **Tempo & defense** | `act_threshold`, `elixir_weight`, `defend_trigger_tiles`, `defend_horizon_s`, `opening_ms`, `min_action_gap_ms` | Minimum utility score to play, elixir penalty weight, defense activation radius & ETA horizon. |
| **Beatdown & bridge gating** | `min_elixir_to_attack`, `press_opp_elixir_max`, `max_opp_elixir_to_bridge`, `min_elixir_advantage_to_bridge`, `attack_blocked_by_threat_eta_s` | Elixir floors and enemy-elixir ceilings before playing a bridge win condition. |
| **Counter-push & back build** | `counterpush_min_value`, `counterpush_tank_ahead`, `counterpush_tank_bonus`, `giant_back_build`, `giant_safe_build_elixir`, `giant_back_build_score`, `giant_build_block_tiles`, `defense_reserve`, `reserve_penalty` | Snowball surviving defenders behind a win-condition tank; back-build a Giant above safe elixir while keeping reserve defense elixir. |
| **Card / cycle / spell IQ** | `counter_gate`, `wc_counters`, `counter_in_hand_mult`, `counter_out_mult`, `counter_unaffordable_punish_mult`, `back_build_counter_in_hand_mult`, `cycle_punish_partial_deck`, `spell_aware`, `kill_aware_spells`, `save_counters_for_in_hand_threats`, `spell_value_threshold`, `spell_cycle_max_casts` | Scales pushes against opponent deck/cycle & elixir affordability (`wc_counters`), kills vs chip value on spells, avoids throwing Minions into in-hand Arrows or wasting tank killers in cycle. |
| **Tier 0 reflexes (`tier0:`)** | `lethal_margin_hp`, `spell_tower_damage`, `leak_threshold`, `leak_ms`, `leak_card_preference`, `leak_tiles`, `emergency_eta_s`, `emergency_hp_frac` | Unconditional tower-finish spells, sub-`35%` HP emergency defense, and anti-leak dumps above `leak_threshold`. Set `spell_tower_damage` to **your** card levels. |
| **Placement (`placement:`)** | `building_pull`, `ranged_standoff_tiles`, `bridge_tiles`, `back_tiles` | Anchor tiles and standoff distance on the `18×32` arena grid. |
| **Rule weights & postures** | `weights`, `posture_multipliers` | Per-rule multipliers plus posture scaling across `OPENING`, `BALANCED`, `DEFEND_LEAD`, `PRESS`, and `SPELL_CYCLE`. |
| **Counters CSV (`counters:`)** | `card,threat,effectiveness` (`0..1`) | Defense matchup matrix paired with `Card.can_hit(flying)`. |

### Manual promotion funnel (4 deterministic gates)

1. **Gate 1 — Unit tests & lint:**
   ```bash
   pytest -ra && ruff check src/ tests/
   ```
2. **Gate 2 — Offline counterfactual diff (`replay-diff`):** Checks how often your candidate changes logged real-match decisions before running a single phone match.
   ```bash
   droid-harness replay-diff \
     --champion configs/strategies/ladder-vision-v4.yaml \
     --candidate configs/strategies/ladder-vision-v6.yaml \
     --data data/matches \
     --cards configs/domains/clash_royale/cards.json \
     --arena configs/domains/clash_royale/arena/phone-a.yaml \
     --expected-pct 5.0 --tolerance 4.0
   ```
3. **Gate 3 — Shadow scoring / Optuna sweeps (optional, no LLM):** `flywheel/shadow.py` compares candidate vs. champion on identical `GameState` snapshots; `flywheel/tuning/optuna_study.py` offers TPE hyperparameter search (`pip install -e ".[tuning]"`).
4. **Gate 4 — Paired SPRT (`run-experiment`):** Alternates champion and candidate in randomized blocks of two (`--test pairs`) so matchmaking/trophy drift cancels cleanly, logs every result to [`ledger/experiments.jsonl`](ledger/experiments.jsonl), and renders [`ledger/STRATEGY_LEDGER.md`](ledger/STRATEGY_LEDGER.md).
   ```bash
   droid-harness run-experiment \
     --profile configs/device_profiles/phone-a.yaml \
     --layout configs/domains/clash_royale/rois/phone-a.yaml \
     --tiles configs/domains/clash_royale/tiles/phone-a.yaml \
     --arena configs/domains/clash_royale/arena/phone-a.yaml \
     --templates configs/domains/clash_royale/templates/phone-a \
     --screens configs/domains/clash_royale/screens/phone-a \
     --towers configs/domains/clash_royale/towers/phone-a \
     --cards configs/domains/clash_royale/cards.json \
     --champion configs/strategies/ladder-vision-v4.yaml \
     --candidate configs/strategies/ladder-vision-v6.yaml \
     --detector models/yolo26n-cr.pt --classes models/classes.txt \
     --async-detect --mode ladder --balance-epoch 2026-11 \
     --test pairs --title "v6 safer back-build" --hypothesis "Raise safe build elixir to 8.6 to eliminate t=1s losses"

   droid-harness ledger
   ```
   *(Never use `--test fixed` to promote a strategy; always use `--test pairs` or `--test h2h` for two-phone self-play via `selfplay-schedule`.)*

---

## 6. Retrain or expand the detector (when adding cards)

When you unlock new cards or promote a frequent enemy unit out of `other_ground_opp` / `other_air_opp`:

1. Add its `<unit>_me` / `<unit>_opp` class name to [`models/classes.txt`](models/classes.txt) and alias mapping to [`models/katacr_alias.json`](models/katacr_alias.json) (keep indices `0..30` stable or re-export cleanly).
2. Rebuild synthetic + KataCR + own-frame datasets and verify zero train/val overlap:
   ```bash
   python tools/fetch_katacr.py --out data/katacr --part2
   python tools/extract_backgrounds.py --data data/matches-batch --out data/backgrounds

   # --synthesize-missing-sides shifts opp sprites -> blue _me sprites with blue HP bars when _me sprites are missing
   droid-harness synth \
     --backgrounds data/backgrounds data/katacr/backgrounds \
     --sprites data/katacr/segment --bars data/katacr/bars \
     --classes models/classes.txt --synthesize-missing-sides \
     --out data/synth --n 4500 --imgsz 640 384

   droid-harness convert-katacr --part2 data/katacr/_repo/images/part2 --classes models/classes.txt --out data/katacr-real
   droid-harness frames --data data/matches-batch --layout configs/domains/clash_royale/rois/phone-a.yaml \
     --detector models/yolo26n-cr.pt --classes models/classes.txt --out data/own

   droid-harness dataset \
     --train data/synth/data.yaml:train data/katacr-real/data.yaml:train data/own/data.yaml:train \
     --val data/katacr-real/data.yaml:val data/own/data.yaml:val \
     --out data/train.yaml
   ```
3. Train (fast ROCm settings) and export ONNX + check per-class recall gate:
   ```bash
   python -m droid_harness.learn.train_detector \
     --data data/train.yaml --model models/yolo26n-cr.pt \
     --epochs 30 --batch 64 --cache ram --workers 8 --patience 8 \
     --imgsz 640 --export-imgsz 640 384 --name yolo26n-cr-next --out models/yolo26n-cr-next.onnx

   droid-harness eval-detector \
     --model models/yolo26n-cr-next.onnx --classes models/classes.txt \
     --data data/katacr-real/data.yaml --min-threat-recall 0.70 --min-support 20
   ```
   Copy `models/runs/yolo26n-cr-next/weights/best.pt` → `models/yolo26n-cr.pt` (backing up your previous `.pt`/`.onnx` first) once `eval-detector` and `replay` pass.

---

## 7. Troubleshooting & operational limits

- **Phone sleeps or drops ADB:** Verify Developer Options *Stay awake* is enabled while plugged into USB; re-check `adb devices`. If `scrcpy-server` is busy after a crash, wait ~3 s for the previous socket on `local_port` to close or kill stray `adb`/`droid-harness` processes.
- **Deck changed in Clash Royale:** Capture new `affordable` and `greyed` templates via `droid-harness template` into a clean directory, update `deck:` in your strategy YAML, and re-run `replay` / `pytest` — stale hand templates produce `hand_unreadable` / wrong card plays.
- **Disk usage on long runs:** Raw H.264 video grows at **~60 MB/min** (~2–4 GB per 10 matches). After running `analyze` and exporting any frames you want via `droid-harness frames`, keep only `meta.json`, `summary.json`, and `decisions.jsonl` (~1.5 MB/match) for completed matches whose video you no longer need.
- **AMD GPU vs CPU ONNX:** ONNX Runtime removed its ROCm EP upstream. Pass `--detector models/yolo26n-cr.pt` for GPU inference via PyTorch-ROCm (`~4.14 ms p50` at `640×384` on RX 7900 XT); `.onnx` runs via `MIGraphXExecutionProvider` if built, or falls back to CPU (`~26.8 ms p50`).
- **License:** Ultralytics YOLO (`[detector]` / training) is AGPL-3.0; keep the runtime MIT/private or swap in an Apache-2.0 detector such as RF-DETR before public/hosted redistribution.

---

## Companion documentation

- [`docs/refinement-loop-plan.md`](docs/refinement-loop-plan.md) — Scaling from 10-match batches to a 50–100 match farm → single deterministic digest → single LLM review → offline gate → paired SPRT workflow.
- [`docs/pro-strategy-research.md`](docs/pro-strategy-research.md) — Top Clash Royale macro/micro mechanics mapped to `droid-harness` capabilities + ready-to-load strategy YAML blueprints (`ladder-vision-v6.yaml` and archetype templates).
- [`docs/phone-a-bringup.md`](docs/phone-a-bringup.md) — Hardware bring-up and calibration measurements for `phone-a`.
- [`docs/plan.md`](docs/plan.md) & [`docs/review.md`](docs/review.md) — Architecture design spec and engineering review rationale.
- [`todo.md`](todo.md) — Milestone tracker and chronological test/experiment log.
