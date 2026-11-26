# todo.md — from this repo to a bot that plays real matches

**Target:** the bot plays Clash Royale **Training Camp** matches on a real phone, unattended. Every match is logged, analyzed and triaged, and a strategy change has been accepted or rejected by a real SPRT. Stretch goals follow after that (Practice gauntlet, two-phone self-play, coach, learned policy, general Android).

**Where things stand (Nov 2026):** core, perception, policy and flywheel code exist, with 80 passing tests against a fake scrcpy server and a synthetic scene. **Update 2026-11-26:** the harness has now streamed from a real OnePlus N10 (phone-a), phone-a ROIs/screens/card templates/arena geometry are calibrated, and a bootstrap controller driving the real UI through the harness scrcpy client won a Training Camp match **3–1**. No detector is trained yet. Several real-game pieces are missing (listed per milestone, with file pointers).

Tags: **[you]** needs your hands, phone or account · **[code]** pure code work · **[data]** collecting or labeling.
Each milestone ends with **Done when** — don't start the next one until it's true.

> Account rule for everything below: throwaway accounts only; Training Camp, Practice Mode and friendly battles between your own accounts only. Never ladder.

---

## M0 — Hardware and accounts

- [x] **[you]** Pick a dedicated Android 12+ phone with a hardware H.264 encoder and a 90–120 Hz screen. Enable Developer options → USB debugging, Stay awake, *Pointer location* (for debugging).
      *Used the provided OnePlus N10 (BE2028), Android 11, 1080×2400, 90 Hz mode; not 12+, but everything below works. USB debugging was already on.*
- [x] **[you]** USB 3 cable direct to the PC (no hub at first). Something to keep the phone cool (fan or cooling clip).
      *Direct USB connection (`adb devices` → `741be240`). No cooling clip available in this environment.*
- [x] **[you]** Create throwaway account #1 (its own Supercell ID). Install Clash Royale, finish the tutorial, unlock Training Camp.
      *Account `theblackcat99` (0 trophies, No Clan) was already installed and unlocked.*
- [x] **[you]** Pick the v1 deck in the game. It must match `configs/strategies/hog-cycle-v1.yaml` (no champion, no evolutions), or edit the deck list there.
      *The account only owns the starter deck, so `configs/strategies/training-camp-v1.yaml` was added with the real 8 cards: Knight, Princess, Minions, Arrows, Fireball, Giant, Musketeer, Mini P.E.K.K.A (3.6 avg). `hog-cycle-v1.yaml` is left as the reference strategy.*
- [ ] **[you]** Later, for M7: a second phone and throwaway account #2, friends with #1 (or in one clan) so they can send friendly battles.

**Done when:** `adb devices` lists the phone, and Training Camp opens on account #1.

---

## M1 — Connect for real (plan P0)

- [x] **[you]** Download the `scrcpy-server` file from one scrcpy release into `vendor/scrcpy-server-v<ver>`. Copy `configs/device_profiles/phone-a.example.yaml` → `phone-a.yaml`; set `serial`, `scrcpy_server_version`, `scrcpy_server_jar`.
      *Pinned `vendor/scrcpy-server-v2.4`; profile `configs/device_profiles/phone-a.yaml` sets serial `741be240`, version `2.4`, jar `vendor/scrcpy-server-v2.4`, local port 27183.*
- [x] **[code]** Check `src/droid_harness/core/transport/scrcpy_protocol.py` against that release's `doc/develop.md`: socket order, dummy byte, 64-byte device meta, 12-byte codec meta, frame-header flag bits, the 32-byte touch struct, the `SET_DISPLAY_POWER` payload, keycode/text layouts. Update `tests/test_scrcpy_protocol.py` to pin whatever changed.
      *Checked v2.4: `writeVideoHeader` is exactly 12 bytes (codec id + width + height), device meta 64 bytes, touch struct 32 bytes, frame header 12 bytes — matches the code and tests. Also checked v5.0.1: it adds a 12-byte session packet after the 4-byte codec id, so the current client is intentionally pinned to v2.4 rather than silently misreading v5 frames.*
- [x] **[code]** Check `H264Decoder` (`core/transport/decoder.py`): does the `low_delay` option apply on your PyAV version? If not, set it the way your version supports.
      *PyAV 19.0.2: the `flags=low_delay` option is accepted; live decode p50 ≈ 3.8 ms, p99 ≈ 15 ms at 464×1024.*
- [ ] **[you]** Run `droid-harness stream --profile configs/device_profiles/phone-a.yaml --seconds 30 --save-frame data/frame_home.png`. Expect frames while the screen moves and none while it's static; both are normal.
- [ ] **[you]** Run `scrcpy --list-encoders` and set `video_encoder` to the hardware H.264 encoder. A/B with and without `codec_options`.
- [ ] **[you]** Loopback test: `droid-harness loopback --profile ... --x <quiet spot> --y <quiet spot> --samples 200`. Record p50/p99 and set `exec_latency_ms` to the p50.
- [ ] **[you]** Check display-off: set `turn_display_off: true`, stream a Training Camp battle, and confirm frames keep coming and the game keeps animating with the panel dark. If they don't, set it to false and note the phone model.
- [ ] **[you]** Resolution sweep: `max_size` 720 / 1024 / 1280. Record loopback p50 and decode time for each. Pick the final size after M3's detector check.
- [ ] **[code]** Add a `--duration` soak mode to `stream` that runs 30 min and reports reconnects, so you can confirm static menus don't trigger false reconnects.

**Done when:** the 30-minute stream has 0 false reconnects, loopback p99 is recorded, and a tap at known pixels lands where *Pointer location* shows it.

---

## M2 — See the HUD and drive the menus (plan P1)

Calibration, per phone, from saved frames:

- [ ] **[you]** Save frames of each screen: HOME, the menu that holds Training Camp, the Training Camp confirm, LOADING, IN_BATTLE (early, mid, overtime), RESULT_WIN / RESULT_LOSS / RESULT_DRAW, and common POPUPs (chests, offers, season pass).
      *Captured HOME, MENU, TRAINING_CONFIRM, IN_BATTLE (early + mid), RESULT_WIN and RESULT_LOSS. Still missing LOADING, overtime, RESULT_DRAW and the common POPUPs.*
- [x] **[you]** ROIs: copy `configs/domains/clash_royale/rois/example.yaml` → `rois/phone-a.yaml`. Measure hand slots ×4, next card, elixir bar, clock, 6 tower-HP fields, arena rectangle and menu buttons as fractions of the frame. Tune `elixir_hsv_lo/hi` on a real frame.
      *`rois/phone-a.yaml` calibrated on 1080×2400 frames; elixir HSV [135,80,120]–[175,255,255]. The stable screen-ID region is the top 18% (clock/opponent), not the old top/bottom halves.*
- [ ] **[you]** Templates: run `droid-harness template ... --state affordable|greyed|selected` for all 8 deck cards in every state, plus the next-card slot.
      *All 8 deck cards + next-card templates exist in `templates/phone-a` (affordable and several greyed). Selected-state templates are still missing; the card matcher is exact (1.00 NCC) on all saved hand/next frames.*
- [ ] **[you]** Screens: run `droid-harness screen ... --name <SCREEN>` for every screen above (2–3 frames each).
      *Registered `HOME` (2), `MENU`, `TRAINING_CONFIRM`, `IN_BATTLE` (3), `RESULT_WIN` (2), `RESULT_LOSS`; missing LOADING/DRAW/POPUP references.*
- [ ] **[you]** Digits: crop `0.png`…`9.png` from tower-HP text into `configs/domains/clash_royale/digits/phone-a/`.

Code gaps to close in this milestone:

- [x] **[code]** **Menu flow for the real UI.** Update `DEFAULT_FLOWS["training_camp"]` in `domains/clash_royale/menus.py` to the button names you calibrated. The June 2025 update moved Training Camp into the menu, so verify the path. Add popup handling (one `close` per popup type).
      *Verified path: HOME → `menu` (hamburger, top-right) → `training_camp` (MENU) → `confirm` (TRAINING_CONFIRM dialog) → battle. The existing `DEFAULT_FLOWS` names already match the layout; `POPUP → close` exists and `close` is in the layout. The only popup seen so far (Next Reward) was non-modal.*
- [ ] **[code]** **Destroyed towers.** `perception/pipeline.py::_read_towers` never sets a tower to 0; unreadable digits just keep the last value. Add destroyed-tower detection (template of the rubble/crown state per tower ROI, or detector classes). Crowns, `pocket_open` and the king/princess targeting all depend on it.
- [ ] **[code]** **Match clock.** The layout has a `clock` ROI but the pipeline never reads it. Read it at 1 Hz to align match t=0 with the game (screen detection lags a few frames, and the intro countdown shifts it). Flag drift over 1 s as a perception warning. Double elixir at 2:00 depends on this.
- [x] **[code]** **Arena geometry.** `app.py::BotRunner` uses `Arena()` defaults. Add `--arena configs/domains/clash_royale/arena/phone-a.yaml` (load with `Arena.load`) and calibrate river, bridges and tower tiles from the same landmarks as the homography.
      *Added `--arena` to `droid-harness run` and `RunConfig`; `configs/domains/clash_royale/arena/phone-a.yaml` was fitted from the same six landmarks as `tiles/phone-a.yaml` (river y 15.0–16.5, bridges x 3.6/14.4).*
- [ ] **[code]** **Golden frames.** Commit 200+ labeled real frames (hand, elixir, towers, screen ID) under `tests/golden/phone-a/` (gitignored if large; keep an index). Add a test that fails if HUD accuracy drops below 99.5%.
- [ ] **[code]** Elixir start: the first HUD read seeds both models; confirm on real frames that the first `IN_BATTLE` frame shows the starting bar and not the intro.

Run it:

- [ ] **[you]** `droid-harness run ... --matches 5` **without** `--detector`. The bot can't see units yet, but it should navigate menus, recognize battles and results, read the hand and elixir, and cycle cards with the leak guard.
- [ ] **[you]** Run `droid-harness analyze --data data/matches --cards ...` and read the warning counts. Fix calibration until `perception_warnings` is near zero.

**Done when:** 20 unattended Training Camp matches complete (menus, battle, result, next match) with correct results in `summary.json` and HUD golden-set accuracy ≥ 99.5% in every card state.

---

## M3 — See the units (plan P2)

The detector is the biggest missing piece; nothing in `learn/` exists yet.

- [ ] **[code]** **Class list.** Write `models/classes.txt`: units of your deck + the Training Camp bot's decks + common threats, spell effects (`fireball_effect`, `log_effect`, …), and tower states. Names must match `unit_classes` in the card DB (`tools/build_card_db.py` output); add an alias map if KataCR names differ.
- [ ] **[code]** **Synthetic generator** `src/droid_harness/learn/synth.py`: paste sprite cut-outs on empty-arena backgrounds (random position, scale, overlap, HP bars, effects), write YOLO labels. Seed it from KataCR's generative dataset approach.
- [ ] **[data]** Pull KataCR's open detection dataset; check the license and which classes it covers (snapshot from May 2024, so newer cards are missing). Map its class names onto yours.
- [ ] **[data]** Record 30–50 of your own battles (the recorder already saves `stream.h264`). Extract frames, auto-label them with the first model, fix the labels in CVAT/Label Studio, and repeat.
- [ ] **[code]** `src/droid_harness/learn/train_detector.py`: Ultralytics YOLO26-n training on PyTorch-ROCm, exported to ONNX with a static shape. Also export `classes.txt`.
- [ ] **[you]** Install the ONNX Runtime MIGraphX EP (AMD's ROCm docs). Benchmark `OnnxDetector` on MIGraphX vs CPU vs PyTorch-ROCm at your input size, and keep the fastest. Check `detector.provider` in the logs.
- [ ] **[code]** Crop to the arena: pass only `layout.arena` to the detector and offset the boxes back. This cuts latency and removes false hits on the HUD.
- [ ] **[code]** Flying units: the box bottom isn't the ground point for air units. Add a per-class foot offset (from the card DB `flying` flag, tuned on frames).
- [ ] **[code]** Calibrate the HSV ranges in `perception/ownership.py` on real HP bars. Find out whether real troops show bars before they take damage; if not, rely on the spawn-side prior.
- [ ] **[code]** If crowded fights cause ID switches (`tracker.id_switch_suspects`), swap `perception/tracker.py` for ByteTrack (`supervision`) behind the same interface.
- [ ] **[code]** Spell plays: map detected spell effects to plays in `SpawnGrouper`, so opponent elixir and cycle tracking see spells. If effects stay unreliable, set `OpponentElixir.miss_widen_per_s` > 0.
- [ ] **[code]** Replay regression: a script that re-runs recorded `.h264` matches through perception and diffs the spawn/play/tower events against the previous model.

**Done when:** you meet your per-class recall target on held-out real frames, the detector runs ≤ 10 ms p95 on the 7900 XT, and full-loop p99 is under 100 ms in battle (`loop_p99_ms` in summaries).

---

## M4 — Play real games (plan P3)

- [ ] **[you]** Build the real card DB with `python tools/build_card_db.py ~/src/cr-api-data`. Check the cost and deploy time of every deck and threat card against the current game, and fix stale entries by hand (the source dates from 2023).
- [ ] **[you]** Fill `tier0.spell_tower_damage` in the strategy file from **your** card levels (in-game card info).
- [ ] **[code]** Verify motion units: compare tracker velocities with the DB `speed_tiles_s` (the builder assumes speed/60 = tiles/s). Check spell travel times against recorded video and adjust `projectile_speed_tiles_s`.
- [ ] **[code]** Tune actuation on the device: tap hold/gap in `CRActionMapper` (currently 20/20 ms), tap-tap vs drag, and when pre-selection helps. Watch *Pointer location* and the execution failures in triage.
- [ ] **[code]** **Execution retry:** if the card hasn't left the hand within ~300 ms, re-tap once (log it). Right now a missed tap only blocks new actions until `pending_timeout_ms`.
- [ ] **[code]** Seed the counter table (`configs/strategies/counters/hog-cycle-v1.csv`) for the Training Camp bot's decks. Current values are placeholders.
- [ ] **[code]** Calibrate placement tiles (`placement.building_pull`, `bridge_tiles`, `back_tiles`, `tier0.leak_tiles`) on the real arena.
- [ ] **[you]** Play 50 Training Camp matches. Run `analyze`. Read the triage mix, the lowest-value rules and a few loss stories, then fix the obvious problems.

**Done when:** the bot beats Training Camp consistently over 50 matches, and every decision is logged with its candidates, posture, predicted tile and latency.

---

## M5 — Evaluation you can trust (plan P4)

- [x] **[code]** **Experiment runner** `src/droid_harness/flywheel/eval/runner.py`: alternates champion/candidate strategy files per match, tags `experiment_id`/`arm` on every match, checks `game_version`, and stops on an SPRT decision (`flywheel/eval/sprt.py`, `inclusion.py`). Write results to the ledger and the warehouse `experiments` table.
      *Implemented `ExperimentRunner` (alternation, SPRT stop, game-version void, ledger + warehouse writes) with unit tests, plus `droid-harness run-experiment` which plays one Training Camp match per arm with `--experiment-id`/`--arm` until the SPRT decides. Shadow mode / replay-diff CLI / retention / dashboard still open.*
- [ ] **[code]** **Shadow mode:** run the candidate policy alongside the champion on the same `GameState` and log its choice (`shadow_strategy_id`, `shadow_choice_json`, already in the schema). Add a report: disagreements × engagement value.
- [ ] **[code]** **Replay-diff CLI:** `droid-harness replay-diff --champion A.yaml --candidate B.yaml --data data/matches --expected-pct N`, wrapping `flywheel/eval/replay_diff.py`.
- [ ] **[code]** **Retention job:** after 7 days, transcode losses and flagged engagements to ~1 Mbps H.265, export anomaly keyframes for labeling, and delete the rest. Raw video is ~60 MB/min.
- [ ] **[code]** A dashboard notebook (`notebooks/dashboard.py`, marimo) over DuckDB: win rate with CI by strategy × opponent × game version, play value by rule × posture, triage mix, loop p99 per day.
- [ ] **[you]** Run the first real SPRT: champion vs one small, hand-made change, bounds [0.50, 0.58]. Expect roughly 200 games.

**Done when:** one SPRT runs end-to-end unattended, its result is in the ledger, and no match in it spans a game update.

---

## M6 — Coach and tuning (plan P5)

- [ ] **[code]** `droid-harness coach` CLI: analyze → STRATEGY-only loss stories → `coach.context_pack` → `AnthropicClient` → `check_guardrails` → `apply_patch` on a branch → replay-diff gate → queue the experiment.
- [ ] **[code]** Wire Optuna's `play_block` (`flywheel/tuning/optuna_study.py`) to the experiment runner. Tuning produces candidates only; they still go through M5.
- [ ] **[you]** Accept or reject at least one coach proposal and one tuned candidate via SPRT.

**Done when:** ≥ 1 coach proposal and ≥ 1 tuned candidate have gone through the full funnel.

---

## Definition of done: "fully running harness in a real bot game"

- [ ] The bot plays Training Camp matches unattended for hours with automatic recovery (reconnect, popups, app restart).
- [ ] Loop p99 is under 100 ms in battle; HUD accuracy ≥ 99.5%; the detector meets its recall target.
- [ ] Every match is recorded, analyzed, triaged and in the warehouse, with a game version and balance epoch.
- [ ] At least one strategy change has been accepted or rejected by SPRT and recorded in the ledger.

---

## After that (stretch)

- **M7 — Practice gauntlet + self-play.** Add a Practice Mode flow (battle log → opponent → Training) and a friendly-battle invite/accept flow (`menus.py`; the `friendly` flow is a stub). Run two `BotRunner`s with distinct ports, swapping strategy ↔ (phone, account) per `flywheel/eval/selfplay.py`. Check `seat_effect` for device bias.
- **M8 — Learned policy (plan P6).** `learn/label_from_video.py` (card left hand + matching spawn → state/action), BC → filtered BC → offline RL, ONNX ranker plugged into `CRPolicy(ranker=...)`. Gate each step by SPRT.
- **M9 — L2 advisor (optional).** A separate process publishing `Intent` to `IntentBus`, running off-GPU or via API. Measure p99 with it on. Keep it only if it wins an SPRT, then distill it.
- **M10 — General Android (plan P7).** Test DroidRun Portal on your phones (`domains/android_os/a11y.py`, with uiautomator fallback). Add a GUI-model fallback with the Set-of-Mark overlay. Measure step latency and cache-hit rate on an everyday task suite.

## Risks to watch

- **Game updates** change UI, cards and balance. Re-run the golden set after each update, bump `balance_epoch`, and void running tests.
- **scrcpy upgrades** can change the protocol. Re-pin the version, re-check the structs, and re-run M1.
- **Thermals:** watch `loop_p99_ms` and loopback drift over long runs.
- **Perception errors** look like strategy errors. Trust triage, and fix perception before tuning strategy.
