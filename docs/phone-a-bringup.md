# phone-a bring-up (2026-11-26)

This is the first real-device run. Everything below was measured on the connected
phone, not the synthetic scene.

## Device / account

| item | value |
|---|---|
| phone | OnePlus N10 (BE2028), Android 11, 1080×2400, 90 Hz mode |
| serial | `741be240` |
| game | Clash Royale, package `com.supercell.clashroyale` |
| account | `theblackcat99` (0 trophies, No Clan, throwaway) |
| mode | Training Camp (`Trainer Jonas`, Royal Trainers) |
| first real win | **3–1** vs Trainer Jonas, 2026-11-26 |
| proof frame | `configs/domains/clash_royale/screens/phone-a/RESULT_WIN__0.png` |

## Stream

`scrcpy-server-v2.4` is pinned in `vendor/`. The v2.4 wire format is the one the
existing client/tests implement (12-byte codec meta, no per-session packet). v5.0.1
adds a 12-byte session packet after the 4-byte codec id, so do **not** point the
profile at v5 without updating `scrcpy_bridge.py` first.

```bash
.venv/bin/droid-harness stream \
  --profile configs/device_profiles/phone-a.yaml \
  --seconds 15 --save-frame /tmp/opencode/screens/stream_home.png
```

Result: `BE2028 464x1024 h264`, 867 frames / 15 s (57.8 fps while the screen moved),
decode p50 3.8 ms / p99 15 ms.

## Calibration files

| file | what it holds |
|---|---|
| `configs/domains/clash_royale/rois/phone-a.yaml` | hand ×4, next, elixir, clock, six tower HPs, arena, menu buttons, HSV |
| `configs/domains/clash_royale/tiles/phone-a.yaml` | pixel → 18×32 tile homography (max landmark error 0.31 tiles) |
| `configs/domains/clash_royale/arena/phone-a.yaml` | river, bridge and tower tiles fitted from the same landmarks |
| `configs/domains/clash_royale/templates/phone-a/` | all 8 deck cards + next; affordable/greyed |
| `configs/domains/clash_royale/screens/phone-a/` | HOME, MENU, TRAINING_CONFIRM, IN_BATTLE, RESULT_WIN, RESULT_LOSS |
| `configs/domains/clash_royale/cards.json` | minimal card DB for the starter deck |
| `configs/strategies/training-camp-v1.yaml` | strategy whose deck matches the account |

Important classifier fix: the default screen regions are the top and bottom halves,
but the bottom half changes with the hand. `phone-a.yaml` uses the **top 18% only**
(clock/opponent/name), which classifies all saved battle frames correctly even when
the hand is completely different.

Card matching on the saved frames (normalized cross-correlation) is 1.00 on every
occupied hand/next slot. Example: `battle2_start` → Knight, Mini P.E.K.K.A, Princess,
Fireball; next Giant.

## Menu path

HOME → hamburger (`menu`, top-right) → Training Camp (`training_camp`) →
“Do you want to start a training match?” (`confirm` / OK) → battle.

## Win

A bootstrap controller (`tools/bootstrap_win_bot.py` in this repo) used the harness
`ScrcpyClient` for frames, read the elixir bar from the frame, and played every
affordable card at the bridges through `adb shell input tap`. It won **3–1** in
about 90 s (12 card plays), taking all three enemy towers. This validates the
capture, HUD read, hand rotation and real touch path before the detector exists.

The full `CRPolicy` path is being validated separately; without a detector it can
navigate and cycle cards but cannot see or defend against units.

## Known rough edges

- The game can leave the post-match result screen ignoring injected taps. Recovery
  used here: `adb shell am force-stop com.supercell.clashroyale` then relaunch.
  The liveness monitor's BACK key on that screen opens a “Return to the Main Menu?”
  confirmation, so result-screen recovery needs a dedicated flow (tap OK; if that
  fails, force-stop).
- The first full-policy run proved that a top-only dhash cannot distinguish a live
  battle from the result screen (the opponent name/HP are in the same place). It
  ended a match at 56.9 s and mislabeled the eventual loss as a win. Fixed:
  `perception/screens.py::read_result` verifies the pink/blue result banners and
  counts gold crowns, and `CRSession` requires 3 stable result observations plus a
  post-result gate before a new match can start.
- The `--arena` path is loaded by `BotRunner`, but the policy's default placement
  tiles were not re-tuned against it yet.
- No detector, no tower-HP digits, no match clock read yet.

## Updated calibrated state (2026-11-26, commit `fc94e6b`)

- **Tracked `phone-a` calibration** (`.gitignore` exceptions keep these versioned without `git add -f`):
  - `configs/device_profiles/phone-a.yaml` (`serial: "741be240"`, `account: "theblackcat99"`, `exec_latency_ms: 40`)
  - `configs/domains/clash_royale/rois/phone-a.yaml` (adds `battle`, `dismiss`, and `trophy_ok` buttons)
  - `configs/domains/clash_royale/tiles/phone-a.yaml` and `arena/phone-a.yaml`
  - `configs/domains/clash_royale/templates/phone-a/`: rebuilt 18 clean PNGs across all 8 deck cards (removed zoomed `archers__0` and duplicates)
  - `configs/domains/clash_royale/towers/phone-a/`: 10 `alive`/`rubble` tower templates (`me_*`, `opp_*`)
  - `configs/domains/clash_royale/screens/phone-a/`: adds `LOADING__0`, `MATCHMAKING__0`, `TROPHY_ROAD__0`, and `TROPHY_ROAD__1` (Bone Pit Arena-lock modal) alongside `HOME`, `MENU`, `TRAINING_CONFIRM`, `IN_BATTLE`, `RESULT_WIN`, and `RESULT_LOSS`
- **Deck identity settled & card DB built:** starter slot is **Archers** (not Princess; upgraded Lv2 → Lv3 on `theblackcat99`). Built 118-card local `configs/domains/clash_royale/cards.json` (`arrows` patched to `damage: 144, radius_tiles: 3.5, ct_damage_pct: 20.8`).
- **OxygenOS loopback finding:** synthetic `show_touches` dots are hidden on BE2028 (`droid-harness loopback` timed out 50/50), so `exec_latency_ms: 40` is kept from configuration rather than loopback measurement.
- **King Level 1 arena cap:** near ~275–299 trophies, Clash Royale displays the Bone Pit Arena-lock modal (`TROPHY_ROAD__1.png`); registered and automatically dismissed via `trophy_ok` so ladder runs continue uninterrupted.
- **Live detector-driven validation:** YOLO26-n trained to epoch 40 (`best.pt` epoch 36 `mAP50=0.797`) and executed via `TorchDetector` PyTorch-ROCm (`7.90 ms p50 / 9.83 ms p95`) + `--async-detect`. On `ladder-vision-v3`, `data/ladder-verify` finished **9W–1L–0D (90.0%)** across 10 completed matches (`loop_p50_ms` `2.48–2.77 ms`, `loop_p99_ms` `4.01–4.36 ms`, `0` aborts, `0` orphaned dirs, `0–1` `elixir_discrepancy`/match). Tower-HP digits (`digits/phone-a/`) and display-off thermal testing remain open.
