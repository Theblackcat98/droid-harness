# Overnight, LLM-free plan

Goal: leave the phone running Training Camp matches overnight, record and analyze every
match, and improve the decision strategy through deterministic experiments and SPRT —
without an LLM in the control loop.

An LLM (coach) is optional and offline only. It is not required for capture, perception,
policy, touch injection, recording, triage or SPRT.

## Current status (2026-11-26, post-`fc94e6b`)

### Works on the real phone

- OnePlus N10 `741be240`, Android 11, 1080×2400, 90 Hz mode.
- Throwaway account `theblackcat99` (King Level 1, Archers Lv3, Arena 1), `--mode ladder` approved & working end-to-end.
- Real deck: Knight, Archers, Minions, Arrows, Fireball, Giant, Musketeer, Mini P.E.K.K.A.
- `scrcpy-server-v2.4` pinned. Stream: `464×1024` H.264, ~58 fps moving, decode p50 ≈ 3.8 ms, p99 ≈ 15 ms.
- Tracked per-phone calibration: `device_profiles/phone-a.yaml`, `rois/phone-a.yaml` (incl. `battle`, `dismiss`, `trophy_ok`), `tiles/phone-a.yaml`, `arena/phone-a.yaml`, rebuilt 18-PNG `templates/phone-a/`, 10 `towers/phone-a/` alive/rubble templates, and `screens/phone-a/` (`HOME`, `MENU`, `TRAINING_CONFIRM`, `LOADING`, `MATCHMAKING`, `TROPHY_ROAD`, `IN_BATTLE`, `RESULT_WIN`, `RESULT_LOSS`). Local `cards.json` (118 cards, `arrows` patched).
- Strategies: `configs/strategies/ladder-vision-v2.yaml` and `ladder-vision-v3.yaml` (plus `training-camp-v3` and `training-camp-blind-v1`).
- Trained M3 detector: epoch-40 YOLO26-n (`best.pt` epoch 36 `mAP50=0.797` / `mAP50-95=0.514`; local gitignored `models/yolo26n-cr.pt` & static `640×384` `models/yolo26n-cr.onnx`). On `veritas` (RX 7900 XT), `TorchDetector` PyTorch-ROCm runs **`7.90 ms p50 / 9.83 ms p95`** vs CPU `OnnxDetector` `26.78 ms p50 / 53.40 ms p95`.
- Perception/session reliability: `CardMatcher` `local_znorm` + mask, `read_result_detail()` left+right banner checks, `battle_evidence` battle-lock & timeout invariants, match-start elixir sync, low-conf generic tower-skin FP suppression, and `MENU`/`TROPHY_ROAD` recovery.
- **Verified live ladder run (`ladder-verify`, `ladder-vision-v3` + PyTorch-ROCm):** **9W–1L–0D (90.0%, 95% CI `[0.596, 0.982]`)**, `loop_p50_ms` **`2.48–2.77 ms`**, `loop_p99_ms` **`4.01–4.36 ms`**, `0` aborts, `0` orphaned match dirs, `0–1` `elixir_discrepancy`/match. Test suite: **179 passed, 1 skipped**.

### Known issues

1. **Detector weak/untested enemy classes.** On `data/katacr-real/data.yaml` val, `archer_opp` (`R=0.27`) and `knight_opp` (`R=0.43`) lag, and `giant_opp`, `mini_pekka_opp`, `minion_opp`, and `arrows_effect` have `n=0` in KataCR val. Needs real-frame labeling via `droid-harness frames` + fine-tuning.
2. **No tower-HP digits yet** (`digits/phone-a/`), so mid-match tower HP stays inferred except destroyed-tower zeroing from `TowerStateReader` (`towers/phone-a/`); `my_crowns`/`opp_crowns` in summaries come reliably from `read_result_detail()`.
3. **Transition-window `hand_unreadable` / `affordability_mismatch` warnings** remain (still the bulk of `PERCEPTION` triage entries; no `RESULT_DRAW`/POPUP refs yet, and selected-state card templates missing).
4. **Loopback / display-off / thermals unverified overnight.** OxygenOS hides synthetic `show_touches` dots (50/50 loopback timeout; kept `exec_latency_ms: 40`), and `turn_display_off: true` thermal behavior over hours hasn't been tested.
5. **No live SPRT or video retention job yet.** `ExperimentRunner` (`run-experiment --test pairs`) is implemented and unit-tested, but hasn't run live; raw `.h264` recordings (~60 MB/min) still need automated M5 pruning/transcoding before multi-hour unattended runs.

## Architecture of the overnight loop

```
scrcpy H.264 ──► HUD + screen classifier + detector ──► CRPolicy ──► touch events
      │                                                        │
      └──────────────► Recorder (video + JSONL) ◄──────────────┘
                              │
                              ▼
                  analyze / triage / warehouse
                              │
                              ▼
                     SPRT champion vs candidate
                              │
                              ▼
                     ledger + next experiment
```

No LLM is in the loop. The optional coach runs offline on triaged loss stories and still
has to pass replay-diff + SPRT before a change is accepted.

## Phase 0 — make the runner trustworthy

Before leaving it overnight:

1. **Fix result settling.**
   - Detect result banners/crowns.
   - Wait 1.0–1.5 s.
   - Re-read the final screen.
   - Only then call `_end_match`.
   - Add a test with a synthetic intermediate crown frame and the final frame.
2. **Fix result-screen recovery.**
   - Try `ok`.
   - If the screen is still result after N seconds, `am force-stop` + relaunch.
   - Never let the liveness BACK key open the exit-match confirmation during a live
     battle.
3. **Add popup handling.**
   - One `close` action per popup type (chest, offer, season pass, “next reward”).
   - Capture 2–3 frames per popup.
4. **Add tower-HP digits** (or crown-count fallback) so summaries contain real crowns.
5. **Add a watchdog.**
   - `adb devices` check.
   - App process check.
   - Screen-stuck check.
   - Reconnect scrcpy, force-stop/relaunch, resume.
6. **Test display-off** (`turn_display_off: true`) and phone thermals before an
   overnight run.

## Phase 1 — detector (the main blocker)

Until the bot sees troops, strategy tuning is blind. Build M3:

1. `models/classes.txt` for your deck + trainer decks + common threats, spell effects
   and tower states.
2. Synthetic generator (`learn/synth.py`) + KataCR data + 30–50 recorded battles.
3. Train YOLO26-n, export static-shape ONNX, export `classes.txt`.
4. Benchmark ONNX Runtime providers; keep the fastest.
5. Crop to `layout.arena` and offset boxes back.
6. Add flying-unit foot offsets.
7. Calibrate HP-bar ownership on real frames.
8. Map spell effects into `SpawnGrouper`.
9. Add a replay-regression script over recorded `.h264` matches.

Minimum acceptance before overnight tuning:

- Per-class recall target met on held-out real frames.
- Detector ≤ 10 ms p95 on the target GPU/CPU.
- Full-loop p99 < 100 ms in battle.

## Phase 2 — deterministic strategy loop

### Batch runner

Run N matches, then analyze:

```bash
droid-harness run \
  --profile configs/device_profiles/phone-a.yaml \
  --layout configs/domains/clash_royale/rois/phone-a.yaml \
  --tiles configs/domains/clash_royale/tiles/phone-a.yaml \
  --arena configs/domains/clash_royale/arena/phone-a.yaml \
  --templates configs/domains/clash_royale/templates/phone-a \
  --screens configs/domains/clash_royale/screens/phone-a \
  --strategy configs/strategies/training-camp-v1.yaml \
  --cards configs/domains/clash_royale/cards.json \
  --data data/overnight-$(date +%F) \
  --matches 50 \
  --balance-epoch 2026-11
```

Wrap it with a supervisor:

```bash
while true; do
  droid-harness run ... --matches 50 || true
  sleep 10
done
```

Use `tmux` or a systemd user service so an SSH disconnect does not stop it.

### Analysis

```bash
droid-harness analyze \
  --data data/overnight-$(date +%F) \
  --cards configs/domains/clash_royale/cards.json
```

Read:

- win rate with CI by strategy × opponent × game version,
- `perception_warnings`,
- `loop_p99_ms`,
- triage mix and loss stories,
- elixir leaks and execution failures.

### SPRT

Use the existing pieces:

- `flywheel/eval/sprt.py`
- `flywheel/eval/inclusion.py`
- `flywheel/ledger.py`
- `flywheel/warehouse.py`
- `flywheel/eval/replay_diff.py`

Implement the missing `flywheel/eval/runner.py`:

1. Alternate champion and candidate per match.
2. Tag `experiment_id`/`arm` on every match.
3. Check `game_version`; abort the experiment on a game update.
4. Stop on an SPRT decision.
5. Write to the ledger and DuckDB `experiments` table.

Then the overnight loop is:

```
champion:  training-camp-v1.yaml
candidate: one small change (placement, spell, defense, counter table)
run until SPRT accepts or rejects
repeat
```

### Strategy changes worth testing first

- Placement tiles: bridge, back, building pull.
- Spell value threshold and tower chip.
- Defense trigger distance and emergency HP threshold.
- Counter table for the trainer decks.
- Execution retry (re-tap if the card did not leave the hand).
- Match clock and double-elixir timing.
- Destroyed-tower / pocket logic.

## Phase 3 — optional decision model (still no LLM)

If you want a learned decision model instead of hand-written rules:

1. Collect state/action labels from recorded matches (M8).
2. Train a small ranker (BC → filtered BC → offline RL).
3. Export ONNX.
4. Plug it into `CRPolicy(ranker=...)` via the `learned_ranker` field.
5. Gate every step by replay-diff + SPRT.

This is a decision model, not an LLM. It still needs the detector and the same
evaluation funnel.

## Operational guardrails

- **Disk:** raw video is ~60 MB/min. A full night can be tens of GB. Add the M5
  retention job or prune `data/` after analysis.
- **Thermals:** watch `loop_p99_ms` drift and decode time. Use a fan/cooling clip.
- **Game updates:** re-run the golden set, bump `balance_epoch`, and void any running
  experiment that spans an update.
- **Account rule:** Training Camp, Practice Mode and friendly battles only. Never
  ladder.
- **Phone state:** keep it plugged in, disable sleep, and test display-off before an
  unattended run.
- **One process:** only one scrcpy client per phone/port.

## Definition of done for the overnight loop

- 50+ unattended matches complete without manual intervention.
- Every match has `summary.json`, video, decisions and a correct result.
- `analyze` ingests all matches into DuckDB with low `perception_warnings`.
- At least one candidate strategy has been accepted or rejected by a real SPRT.
- The ledger and warehouse both show the experiment.
- Win rate is measured with a confidence interval, not a single lucky match.

## Recommended next three steps

1. Export & label real frames (`droid-harness frames --data data/ladder-verify ...`) and fine-tune YOLO26-n on `veritas` (with shorter epochs / faster settings) so weak and missing `*_opp` classes pass `eval-detector`.
2. Test `turn_display_off: true` / thermal stability over longer batches and implement M5 video retention/pruning before multi-hour unattended runs.
3. Run the first live paired SPRT (`droid-harness run-experiment --test pairs`) comparing `configs/strategies/ladder-vision-v3.yaml` against a single controlled candidate change (placement tuning, exact level-calibrated `spell_tower_damage`, or cycle gating).
