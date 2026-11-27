# droid-harness

A low-latency harness for agents that operate an Android phone. Clash Royale is the reference game; the same core drives general Android apps.

- **Reflex loop without an LLM:** scrcpy H.264 stream → perception → policy → timed touches. In the synthetic closed loop it takes ~3 ms from packet received to action scheduled (p50) on a laptop CPU. The phone's encoder and the game's frame time come on top; measure those with `droid-harness loopback`.
- **Codified strategy:** Tier 0 reflex invariants → postures → typed, unit-tested utility rules, all parameterized by a versioned, content-hashed strategy file.
- **Iteration you can trust:** every decision is logged with the candidates it beat. Lane engagements get a value and a root-cause tag (perception / execution / latency / strategy). Changes are accepted only by an SPRT, after a replay diff and a ledger check.
- **LLMs only offline:** an optional coach reads STRATEGY-tagged loss stories and proposes one guarded patch at a time.

The full design is in [`docs/plan.md`](docs/plan.md); why it looks this way is in [`docs/review.md`](docs/review.md).

> **Scope.** Supercell's terms treat bots as cheating and can permanently ban the account. Use a throwaway account, and only Training Camp, Practice Mode and friendly battles between your own accounts. Never ladder against real people. The general-Android parts carry none of this.

## Status

| Works now (verified CI + live `phone-a`) | Still open / next |
|---|---|
| `scrcpy-server-v2.4` client, low-delay PyAV decode (`3.8 ms` p50), touch executor, pre-roll H.264/PTS recorder, liveness + watchdog | Real-frame labeling (`droid-harness frames`) & fine-tuning for weak/missing `*_opp` classes |
| Tracked `phone-a` calibration (`configs/device_profiles/phone-a.yaml`, `rois/`, `tiles/`, `arena/`, `templates/`, `screens/`, `towers/`) | Tower-HP digits (`digits/phone-a/`), 200+ HUD golden frames, `RESULT_DRAW`/POPUP refs, display-off thermal check |
| `CardMatcher` (`local_znorm` + mask), HP-bar ownership + tracker, spawn grouping, `TowerStateReader`, match clock, `TorchDetector` PyTorch-ROCm (`7.90 ms` p50 / `9.83 ms` p95) | First live paired SPRT (`run-experiment --test pairs`), retention/transcoding job, live `--shadow` CLI wiring |
| M3 data pipeline (`fetch_katacr`, `synth`, `convert-katacr`, `dataset`, `eval-detector`, `replay`) + trained epoch-40 YOLO26-n (`best.pt` epoch 36 `mAP50=0.797`) | Strategy E6/E7/E8, `coach` CLI + API client, Optuna `play_block` wiring |
| Live `--mode ladder` verified (`ladder-vision-v3` + PyTorch-ROCm): **9W–1L–0D** (`4.01–4.36 ms` loop p99, `179 passed, 1 skipped`) | M8 ranker training on 50+ wins, M9 live advisor test, M7 two-phone self-play, M10 DroidRun Portal |

## Quick start (no phone)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                           # 179 passed, 1 skipped (~14 s)
droid-harness sim --seconds 10      # synthetic closed loop; prints plays and loop latency
droid-harness analyze --data data/sim --cards tests/fixtures/cards_min.json
droid-harness sprt                  # sample-size table for strategy tests
```

## Phone setup (plan phases P0–P1)

1. **Server.** Download the `scrcpy-server` file from the scrcpy release you pin into `vendor/`. Copy `configs/device_profiles/phone-a.example.yaml` to `phone-a.yaml` and fill it in. Each phone needs its own `local_port`.
2. **Stream + latency.**
   ```bash
   droid-harness stream --profile configs/device_profiles/phone-a.yaml --seconds 10 --save-frame data/frame.png
   droid-harness loopback --profile configs/device_profiles/phone-a.yaml --x 30 --y 400
   ```
   Put the loopback p50 into `exec_latency_ms`. Static screens send no frames; that's normal.
3. **Card database.** It isn't committed (source has no license, last updated 2023):
   ```bash
   git clone --depth 1 https://github.com/RoyaleAPI/cr-api-data ~/src/cr-api-data
   python tools/build_card_db.py ~/src/cr-api-data
   ```
   Then set spell tower damage for *your* card levels in the strategy file (`tier0.spell_tower_damage`).
4. **Calibrate** (per phone, from saved frames):
   - ROIs: copy `configs/domains/clash_royale/rois/example.yaml` and measure your regions as fractions of the frame.
   - Tiles: fill `landmarks/example.yaml` with pixel positions, then run `droid-harness calibrate-tiles --landmarks ... --out configs/domains/clash_royale/tiles/phone-a.yaml`.
   - Card templates: run `droid-harness template --frame ... --layout ... --slot 0 --card hog_rider --state greyed --out configs/domains/clash_royale/templates/phone-a`. Capture each deck card affordable *and* greyed.
   - Screens: run `droid-harness screen --frame ... --name HOME --out configs/domains/clash_royale/screens/phone-a` for HOME, MENU, TRAINING_CONFIRM, IN_BATTLE, RESULT_WIN/LOSS/DRAW and POPUP.
   - Digits (optional): save `0.png`…`9.png` crops for tower HP.
5. **Run.**
   ```bash
   droid-harness run --profile configs/device_profiles/phone-a.yaml \
     --layout configs/domains/clash_royale/rois/phone-a.yaml \
     --tiles configs/domains/clash_royale/tiles/phone-a.yaml \
     --arena configs/domains/clash_royale/arena/phone-a.yaml \
     --templates configs/domains/clash_royale/templates/phone-a \
     --screens configs/domains/clash_royale/screens/phone-a \
     --towers configs/domains/clash_royale/towers/phone-a \
     --strategy configs/strategies/ladder-vision-v3.yaml \
     --cards configs/domains/clash_royale/cards.json \
     --detector models/yolo26n-cr.pt --classes models/classes.txt \
     --async-detect --mode ladder --matches 15 --balance-epoch 2026-11
   ```
   Pass a `.pt` checkpoint to run `TorchDetector` (PyTorch-ROCm GPU on AMD) or `.onnx` for `OnnxDetector`. Without `--detector`, the bot sees the HUD but no units: vision strategies can't track opponent elixir without vision, so play a blind strategy such as `configs/strategies/training-camp-blind-v1.yaml`. With a detector, `--async-detect` (and optionally `--detect-every N`) keeps the battle loop at ~2.5 ms p50 / ~4.3 ms p99; units are extrapolated between detections.
6. **Train the detector (M3, no phone needed).**
   ```bash
   python tools/fetch_katacr.py --out data/katacr --part2        # sprites (named + catch-alls), bars, arenas, real frames
   python tools/extract_backgrounds.py --data data/matches --out data/backgrounds   # your arena crops
   droid-harness synth --backgrounds data/backgrounds data/katacr/backgrounds \
     --sprites data/katacr/segment --bars data/katacr/bars --out data/synth --n 5000
   droid-harness convert-katacr --part2 data/katacr/_repo/images/part2 --out data/katacr-real   # split by sequence
   droid-harness frames --data data/matches --layout ... --out data/own [--detector models/detector.onnx]  # to label
   droid-harness dataset --train data/synth/data.yaml:train data/katacr-real/data.yaml:train data/own/data.yaml:train \
     --val data/katacr-real/data.yaml:val data/own/data.yaml:val --out data/train.yaml   # refuses overlap
   droid-harness train-detector --data data/train.yaml --out models/detector.onnx        # ultralytics + ROCm
   droid-harness eval-detector --model models/detector.onnx --data data/own/data.yaml   # per-class; gates on *_opp recall
   droid-harness replay --data data/matches --layout ... --tiles ... --templates ... --cards ... \
     --detector models/detector.onnx --out data/replay/detector-v1.json [--baseline data/replay/<previous>.json]
   ```
   Class names live in the ONNX metadata; `--classes` only cross-checks them and refuses to load on a mismatch.

## Iterating on strategy (P4–P5)

```bash
droid-harness analyze --data data/matches --cards configs/domains/clash_royale/cards.json   # engagements, triage, report
droid-harness selfplay-schedule --games 200 --a phone-a:throwaway-1 --b phone-b:throwaway-2
droid-harness run-experiment --champion configs/strategies/training-camp-v3.yaml \
  --candidate configs/strategies/training-camp-v3-e4-cycle-gate.yaml ...                      # --test pairs (default)
droid-harness sprt --wins 118 --losses 86                                                     # accept / reject / continue
droid-harness ledger
```

`run-experiment` defaults to `--test pairs`: matches are assigned in randomized blocks of two (one per arm), and the SPRT counts only pairs where one arm scored better, so a candidate is promoted only for beating the champion. `--test h2h` is for self-play (candidate vs champion directly); `--test fixed` compares against a fixed rate and must never gate a promotion.

The promotion funnel runs unit tests → `flywheel/eval/replay_diff.py` → shadow mode → screening → SPRT → ledger. The coach (`flywheel/coach/`, `pip install -e ".[coach]"`) may only patch `configs/strategies/**`, the rules package and `tests/coach/**`.

## Layout

```
src/droid_harness/
  core/            transport (scrcpy protocol/client, decoder, adb), actuate (gestures, executor),
                   runtime (brain loop, liveness, intent bus), telemetry (recorder, latency, loopback)
  domains/
    clash_royale/  arena, cards, elixir, opponent model, state, perception/, policy/ (tier0, posture,
                   rules/, placement, constraints, utility), actions, menus, session
    android_os/    settle detection, compiled-skills cache, Set-of-Mark overlay, a11y tree sources
  flywheel/        timeline, engagements, triage, loss stories, report, warehouse (DuckDB),
                   eval/ (SPRT, stats, self-play, inclusion, replay diff), coach/, ledger, tuning/
  sim/             fake scrcpy server, synthetic Clash Royale-like scene
configs/           device profiles, ROIs, landmarks, strategies + counter tables
tools/             build_card_db.py
docs/              plan.md, review.md
```

## Notes

- **License:** Ultralytics YOLO is AGPL-3.0. Keeping this repo private is fine; before publishing it or hosting it as a service, switch the detector to an Apache-2.0 model such as RF-DETR.
- **AMD GPU:** ONNX Runtime's ROCm EP was removed upstream; pass `.pt` to use `TorchDetector` directly on PyTorch-ROCm (measured on RX 7900 XT at `640×384`: **`7.90 ms p50 / 9.83 ms p95`** vs CPU `OnnxDetector` **`26.78 ms p50 / 53.40 ms p95`**). `OnnxDetector` prefers `MIGraphXExecutionProvider` when built/installed and otherwise falls back to CPU.
