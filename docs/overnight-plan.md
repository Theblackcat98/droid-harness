# Overnight, LLM-free plan

Goal: leave the phone running Training Camp matches overnight, record and analyze every
match, and improve the decision strategy through deterministic experiments and SPRT —
without an LLM in the control loop.

An LLM (coach) is optional and offline only. It is not required for capture, perception,
policy, touch injection, recording, triage or SPRT.

## Current status (2026-11-26)

### Works on the real phone

- OnePlus N10 `741be240`, Android 11, 1080×2400, 90 Hz mode.
- Throwaway account `theblackcat99`, 0 trophies, Training Camp unlocked.
- Real deck: Knight, Princess, Minions, Arrows, Fireball, Giant, Musketeer,
  Mini P.E.K.K.A (3.6 avg elixir).
- scrcpy-server **v2.4** pinned. Stream: 464×1024 H.264, ~58 fps, decode p50 ≈ 3.8 ms,
  p99 ≈ 15 ms.
- Calibrated per-phone files:
  - `configs/domains/clash_royale/rois/phone-a.yaml`
  - `configs/domains/clash_royale/tiles/phone-a.yaml`
  - `configs/domains/clash_royale/arena/phone-a.yaml`
  - `configs/domains/clash_royale/templates/phone-a/`
  - `configs/domains/clash_royale/screens/phone-a/`
  - `configs/domains/clash_royale/cards.json`
  - `configs/strategies/training-camp-v1.yaml`
- Menu path verified: HOME → hamburger → Training Camp → confirm → battle.
- Card matcher is exact (1.00 NCC) on saved hand/next frames.
- `droid-harness run` supports `--arena` and real-device matches.
- **Genuine win once:** bootstrap controller won **3–1** vs Trainer Jonas
  (`screens/phone-a/RESULT_WIN__0.png`).
- Full-policy run reached loop p50 **45 ms**, p99 **97 ms** over 176 s.

### Known issues

1. **Result settling is not reliable yet.** On the last full-policy run the final screen
   was Trainer Jonas 2 – 1 theblackcat99 (a loss), but the summary recorded a win. The
   verifier caught an intermediate crown-reveal frame. Fix: after a result is detected,
   wait ~1.0–1.5 s (or require ~20 stable frames), re-read the final crowns, then end the
   match.
2. **No detector.** The bot cannot see units, so it cannot defend. This is the gating
   item for a repeatable win.
3. **No tower-HP digits**, so `my_crowns`/`opp_crowns` are 0/0 even on a real result.
4. **High perception warnings** (`hand_unreadable`, `affordability_mismatch`) during
   intro/selected/greyed states.
5. **Recovery is incomplete.** A result screen can ignore injected taps; the liveness
   BACK key opens a “Return to the Main Menu?” dialog. Overnight runs need a dedicated
   result/popup flow or a force-stop fallback.
6. **No experiment runner.** `flywheel/eval/sprt.py` and `inclusion.py` exist, but
   `flywheel/eval/runner.py` is missing, so champion/candidate runs are not automated.

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

1. Fix result settling + re-read, then rerun one match and confirm the summary matches
   the final screen.
2. Add result/popup recovery so the process cannot hang overnight.
3. Start M3 detector work. It is the gating item for a consistently winning,
   self-improving bot.
