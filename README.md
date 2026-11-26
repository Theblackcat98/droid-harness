# droid-harness

A low-latency harness for agents that operate an Android phone. Clash Royale is the reference game; the same core drives general Android apps.

- **Reflex loop without an LLM:** scrcpy H.264 stream → perception → policy → timed touches. In the synthetic closed loop it takes ~3 ms from packet received to action scheduled (p50) on a laptop CPU. The phone's encoder and the game's frame time come on top; measure those with `droid-harness loopback`.
- **Codified strategy:** Tier 0 reflex invariants → postures → typed, unit-tested utility rules, all parameterized by a versioned, content-hashed strategy file.
- **Iteration you can trust:** every decision is logged with the candidates it beat. Lane engagements get a value and a root-cause tag (perception / execution / latency / strategy). Changes are accepted only by an SPRT, after a replay diff and a ledger check.
- **LLMs only offline:** an optional coach reads STRATEGY-tagged loss stories and proposes one guarded patch at a time.

The full design is in [`docs/plan.md`](docs/plan.md); why it looks this way is in [`docs/review.md`](docs/review.md).

> **Scope.** Supercell's terms treat bots as cheating and can permanently ban the account. Use a throwaway account, and only Training Camp, Practice Mode and friendly battles between your own accounts. Never ladder against real people. The general-Android parts carry none of this.

## Status

| Works now (tested in CI, no phone needed) | Needs your phone / data |
|---|---|
| scrcpy wire protocol, client, low-delay decoder, executor, recorder, liveness, loopback test (all against a fake scrcpy server streaming real H.264) | Pin a scrcpy server jar and verify the structs against its `doc/develop.md` |
| HUD readers, card matching (greyed/compressed), digits, HP-bar ownership, tracker, spawn grouping, screen IDs, ONNX detector decoding | Calibrate ROIs, tile landmarks, card/digit templates and reference screens per phone |
| Elixir + opponent models, cycle tracking, lead targeting, Tier 0, postures, rules, constraints, policy | Train the YOLO26 detector (synthetic sprites + KataCR data + your frames) |
| Closed loop: synthetic scene → H.264 → bot → touches → scene reacts | Tune the counter table and parameters on real games |
| Engagements, triage, loss stories, reports, DuckDB warehouse, SPRT, self-play schedule, inclusion rule, replay diff, ledger, coach guardrails | MIGraphX vs PyTorch-ROCm benchmark on your RX 7900 XT; DroidRun Portal on your phones |

## Quick start (no phone)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                           # ~80 tests, ~10 s
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
     --layout configs/domains/clash_royale/rois/phone-a.yaml --tiles configs/domains/clash_royale/tiles/phone-a.yaml \
     --templates configs/domains/clash_royale/templates/phone-a --screens configs/domains/clash_royale/screens/phone-a \
     --strategy configs/strategies/hog-cycle-v1.yaml --cards configs/domains/clash_royale/cards.json \
     --detector models/yolo26n-cr.onnx --classes models/classes.txt --matches 20 --balance-epoch 2026-11
   ```
   Without `--detector` the bot sees the HUD but no units, which is enough to test menus and the match lifecycle.

## Iterating on strategy (P4–P5)

```bash
droid-harness analyze --data data/matches --cards configs/domains/clash_royale/cards.json   # engagements, triage, report
droid-harness selfplay-schedule --games 200 --a phone-a:throwaway-1 --b phone-b:throwaway-2
droid-harness sprt --wins 118 --losses 86                                                     # accept / reject / continue
droid-harness ledger
```

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
- **AMD GPU:** ONNX Runtime's ROCm EP was removed upstream; `OnnxDetector` prefers `MIGraphXExecutionProvider` when installed and falls back to CPU. Benchmark it against PyTorch-ROCm.
