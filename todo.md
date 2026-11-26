# todo.md — from this repo to a bot that plays real matches

**Target:** the bot plays Clash Royale **Training Camp** matches on a real phone, unattended. Every match is logged, analyzed and triaged, and a strategy change has been accepted or rejected by a real SPRT. Stretch goals follow after that (Practice gauntlet, two-phone self-play, coach, learned policy, general Android).

**Where things stand (Nov 2026):** core, perception, policy and flywheel code exist, with 89 passing tests against a fake scrcpy server and a synthetic scene. **Update 2026-11-26:** the harness has now streamed from a real OnePlus N10 (phone-a), phone-a ROIs/screens/card templates/arena geometry are calibrated, and a bootstrap controller driving the real UI through the harness scrcpy client won a Training Camp match **3–1**. M3 detector foundations are committed (`models/classes.txt`, `models/katacr_alias.json`, `learn/synth.py`, `learn/train_detector.py`, `tools/fetch_katacr.py`, `tools/extract_backgrounds.py`), and a read-only detector review found silent-failure bugs to fix *before* the first training run — folded into M3 below, ordered by severity. No detector is trained yet; this host's venv is CPU-only (no torch/ultralytics, ORT CPU-only), so training and the MIGraphX benchmark need a ROCm host. Several real-game pieces are missing (listed per milestone, with file pointers).

**Update 2026-11-26 (PR `m3/phone-free-fixes`):** everything below that didn't need the phone is done: a no-vision strategy that attacks (`training-camp-blind-v1`), CI green on a clean checkout, every detector-review item, side-aware classes, spell-side attribution, tower state from fixed ROIs, async/strided detection, the training-data tools (synth in the inference geometry, KataCR sprites + real-frame converter, trainer), replay regression, execution retry, a real counter table, a stream soak mode and an app watchdog. 120 tests (118 pass, 2 skip without local calibration).

### Priority order to an unattended win vs the Training Camp bot

1. **[you]** Run `training-camp-blind-v1` for 5 matches, then 20 unattended (M2 *Run it*). Confirm the scrcpy taps land (Pointer location) and read `decisions.jsonl` for `offense.blind_push` and `retry:` reasons. If it wins most of them, you already have an autonomous win; everything after this makes it robust.
2. **[you]** Unattended reliability: LOADING/DRAW/POPUP reference screens, `stream --duration 1800` soak, tower `alive`/`rubble` templates (`droid-harness tower-template`).
3. **[you]** Watch 3 recorded matches and list the trainer's cards → add them to `models/classes.txt` (as `<unit>_me`/`<unit>_opp`), `models/katacr_alias.json` and `counters/training-camp-v2.csv`.
4. **[you/data]** `tools/fetch_katacr.py --part2`, `extract_backgrounds.py`, `droid-harness synth`, `convert-katacr`, then train on the ROCm host and gate on recall over the converted real frames.
5. **[you]** `droid-harness replay` with the detector over recorded matches, then live with `training-camp-v2` + `--async-detect`; 50 matches + `analyze`.
6. Then the flywheel (M5/M6).

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
- [x] **[code]** Add a `--duration` soak mode to `stream` that runs 30 min and reports reconnects, so you can confirm static menus don't trigger false reconnects.
      *`droid-harness stream --profile ... --duration 1800` (`core/telemetry/soak.py`): runs liveness with `in_battle=False`, reconnects like `BotRunner`, reports frames, reconnects and findings by kind. Tested against the fake server for a static stream (0 reconnects) and a killed session (reconnects).*

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
      *Code done: `perception/towers.py::TowerStateReader` template-matches each tower ROI against per-phone `alive`/`rubble` references and marks a tower destroyed after 2 rubble reads in a row (permanent for the match; digits can't revive it). `run --towers DIR` wires it in. A `tower_rubble` detection (older models) also marks the nearest tower. Crowns, `pocket_open` and king targeting follow from `towers[name] == 0`.*
- [ ] **[you]** Capture tower templates: `droid-harness tower-template --frame <battle frame> --layout rois/phone-a.yaml --tower all --state alive --out configs/domains/clash_royale/towers/phone-a`, then `--tower opp_left --state rubble` (etc.) from frames after a tower falls.
- [ ] **[code]** **Match clock.** The layout has a `clock` ROI but the pipeline never reads it. Read it at 1 Hz to align match t=0 with the game (screen detection lags a few frames, and the intro countdown shifts it). Flag drift over 1 s as a perception warning. Double elixir at 2:00 depends on this.
- [x] **[code]** **Arena geometry.** `app.py::BotRunner` uses `Arena()` defaults. Add `--arena configs/domains/clash_royale/arena/phone-a.yaml` (load with `Arena.load`) and calibrate river, bridges and tower tiles from the same landmarks as the homography.
      *Added `--arena` to `droid-harness run` and `RunConfig`; `configs/domains/clash_royale/arena/phone-a.yaml` was fitted from the same six landmarks as `tiles/phone-a.yaml` (river y 15.0–16.5, bridges x 3.6/14.4).*
- [ ] **[code]** **Golden frames.** Commit 200+ labeled real frames (hand, elixir, towers, screen ID) under `tests/golden/phone-a/` (gitignored if large; keep an index). Add a test that fails if HUD accuracy drops below 99.5%.
- [ ] **[code]** Elixir start: the first HUD read seeds both models; confirm on real frames that the first `IN_BATTLE` frame shows the starting bar and not the intro.

Run it:

- [ ] **[you]** `droid-harness run ... --strategy configs/strategies/training-camp-blind-v1.yaml --matches 5` **without** `--detector`. It navigates menus, recognizes battles and results, reads hand and elixir, and attacks blind (Giant at the bridge, support behind, spells on the tower). `training-camp-v1` cannot attack without a detector: its opponent-elixir estimate never narrows, so the attack rule scores −0.1.
- [ ] **[you]** Run `droid-harness analyze --data data/matches --cards ...` and read the warning counts. Fix calibration until `perception_warnings` is near zero.

**Done when:** 20 unattended Training Camp matches complete (menus, battle, result, next match) with correct results in `summary.json` and HUD golden-set accuracy ≥ 99.5% in every card state.

---

## M3 — See the units (plan P2)

The detector is the biggest missing piece: no model is trained yet, and this host's venv is CPU-only (no torch/ultralytics; ORT exposes only `CPUExecutionProvider`), so training and the MIGraphX benchmark need a ROCm host. The foundations are in place — `models/classes.txt` (14-class side-aware closed set), `models/katacr_alias.json` (keys verified against upstream `label_list.py`, MIT), `learn/synth.py`, `learn/train_detector.py`, `tools/fetch_katacr.py`, `tools/extract_backgrounds.py`, and `tests/test_detector_m3.py` (including a `SceneDetector` test double that honors the arena-crop contract).

**Detector review (2026-11-26, read-only) — fix these before spending a training run, in this order:**

- [x] **[code]** **Class-name loading corrupts indices in the live path.** `app.py` loads `Path(cfg.classes).read_text().split()`, which keeps the `#` comment lines in `models/classes.txt` — measured: 53 "class names", `knight` at index 41. A model trained on the real 12-class list decodes every box with a wrong name → `card_for_class` returns `None` → units, spawns, threats and opponent-elixir tracking all silently die. Same silent death when `--detector` runs without `--classes` (empty names → `str(c)` fallback, no warning). Strip comments like `tests/test_detector_m3.py::_read_names`, warn on empty names, and cross-check the provided names against the ONNX model metadata at load time (Ultralytics embeds them there).
      *Done: `OnnxDetector` takes names from the ONNX metadata; `--classes` is parsed with `classes.load_class_names` (comments stripped, duplicates rejected) and must equal the embedded list or the detector refuses to load. No names at all → refuses. `BotRunner` warns when a deck card's unit classes aren't in the model.*
- [x] **[code]** **Synth geometry ≠ inference geometry.** `learn/synth.py::compose` resizes *full background frames* to 640×640 (~2.2:1 vertical squash of 464×1024), and `tools/extract_backgrounds.py` mines *full frames*, while inference letterboxes the *arena crop* (aspect-preserving, padded) — and sprites can be pasted over HUD regions inference never sees. Make `compose` reproduce the inference transform (crop to `layout.arena` → `detector.letterbox` → paste) and mine crops, not full frames.
      *Done: `synth.compose` letterboxes an arena-crop background with `detector.letterbox` at the export shape, pastes only inside the image area, and scales sprites by `our_arena_width / 568 × r`. `extract_backgrounds.py` writes arena crops.*
- [x] **[code]** **Per-class NMS.** `perception/detector.py::decode` runs `cv2.dnn.NMSBoxes` across all classes at once (class-agnostic), so overlapping unit-vs-tower and effect-vs-unit boxes suppress each other — unlike Ultralytics' per-class default, and unlike the end-to-end branch which NMSes in-graph. Add class offsets (Ultralytics-style) or make agnostic an explicit choice. Note: e2e exports bake conf/iou at export time, so the runtime `iou` is dead in that branch.
      *Done: class-offset NMS (Ultralytics default); `agnostic_nms=True` is an explicit option. The e2e branch only filters by conf.*
- [x] **[code]** **Decide e2e vs raw at load time.** The decode heuristic (`out.shape[-1] == 6 and out.shape[0] != 4 + nc`) misroutes an e2e output with exactly `4 + nc` rows to the raw branch. Pin the branch once from the session's output shape (or the export mode set in `train_detector.py`).
      *Done: `detector.output_mode` pins `e2e`/`raw`/`raw_t` from the output shape (and the `end2end` metadata key) at load; symbolic shapes pin on the first output. `(16, 6)` with 12 classes is now e2e.*
- [x] **[code]** **Consume tower classes.** `perception/pipeline.py::_detect` lets `tower_princess`/`tower_king`/`tower_rubble` flow into the tracker as ordinary units (opponent towers surface as side "opp" in `state.units`/`threats()` and escape the defend rules only via distance filtering), while nothing ever reads `tower_rubble` — so crowns, `pocket_open` and the destroyed-tower gap stay open even with a trained detector. Route tower classes to a small tower-state estimator and keep them out of the unit tracker.
      *Done, and towers left the class list: they never move, and every synthetic background contains unlabeled towers, so training them contradicted the data. `tower_*` detections (older models) go to tower state, never the tracker.*
- [x] **[code]** **Wire `river_mid_y`.** `perception/tracker.py::Track.side` hardcodes the river at `y > 16.0` while the tracker is constructed with `river_mid_y` (phone-a ≈ 15.75) that is then never used. Works today by calibration coincidence; landmine for phone-b or any arena recalibration.
      *Done: each `Track` carries the tracker's calibrated `river_mid_y`.*
- [x] **[code]** **Detector cadence knob.** `detect_every` exists on `CRPerception` but has no CLI/`RunConfig` flag, and `self.units` hold last-*detected* positions with no extrapolation between detections. The measured loop p99 of 97 ms was *without* detector cost, so every-frame inference will breach the 100 ms budget. Expose `--detect-every` (plus conf/iou/imgsz) on `run`/`run-experiment` and extrapolate units the way the tracker already does for gating.
      *Done: `--detect-every`, `--det-conf`, `--det-iou`, `--async-detect` on `run`/`run-experiment`. `AsyncDetector` runs inference on its own thread against the newest arena crop; results are mapped with the frame they came from. Units are extrapolated along velocity between detections (capped at 500 ms).*
- [x] **[code]** **Training-path nits.** `train_detector.train()` needs `exist_ok=True` (Ultralytics auto-increments run dirs and `best.pt` is then read from the stale run); `synth.generate_dataset` mutates the caller's `sprites` list in place and reads `models/classes.txt` from CWD (silent `class_i` fallback) — pass class names explicitly; stop labeling sprites clipped up to ⅓ off-canvas (tiny sliver boxes hurt small-unit recall, the known weak spot at 464×1024); `Detection.foot` is dead code (the pipeline computes its own foot with the flying offset); `PREFERRED_PROVIDERS` still lists the upstream-removed `ROCMExecutionProvider`.
      *Done: `exist_ok=True` and `best.pt` from the trainer; `generate_dataset` copies the sprite list and requires class names; mostly-hidden or clipped sprites aren't labeled; `Detection.foot` and `ROCMExecutionProvider` removed; `write_classes` needs an explicit path.*

- [x] **[code]** **Side-aware classes** (review 2026-11-26, Claude): units are `<unit>_me`/`<unit>_opp` (your units face up, theirs down; KataCR labels side the same way, `_0_`/`_1_` in sprite names, `bel` column in labels). The pipeline strips the suffix; the class side beats the HP bar and the spawn prior, and `knight_me`/`knight_opp` are never associated into one track. Fixes enemies re-acquired on your half being called "me".
- [x] **[code]** **Spell effects attributed to the right player.** Effects get their side from your own recent plays of that spell (3.5 s window), not from where they land (spells land on the *other* half: your fireball on their tower used to charge them 4 elixir). Effects produce play events but never `units`.
- [x] **[code]** **Raw-head decode crashed on empty frames** (`IndexError` from NMS indexing when nothing passed conf, i.e. every empty-arena frame). Returns `[]` now.
- [x] **[code]** **Rectangular static input.** `letterbox`/`OnnxDetector` take `(h, w)`; the trainer exports 640×384 (arena crops are ~0.6 aspect, so 640×640 was ~40% padding). The model's static input shape wins over `imgsz`.
- [x] **[code]** **KataCR real frames as a held-out set.** `learn/katacr.py` + `droid-harness convert-katacr` map the 12-column labels (`cls cx cy w h bel …`) of the ~7,400 hand-labeled frames in `images/part2` to our side-aware classes; `tools/fetch_katacr.py --part2` fetches them (plus HP-bar sprites and empty arenas). Note: KataCR has no your-side sprites for minion, mini-pekka or princess (only 6 arrows, 17 princess); cut those from your recordings.
- [x] **[code]** **Class list.** Write `models/classes.txt`: units of your deck + the Training Camp bot's decks + common threats, spell effects (`fireball_effect`, `log_effect`, …), and tower states. Names must match `unit_classes` in the card DB (`tools/build_card_db.py` output); add an alias map if KataCR names differ.
      *Now 14 classes: 6 starter-deck units × 2 sides + 2 spell effects (no towers, no `minions`: Minions spawn three `minion` units; make sure your local `cards.json` says `unit_classes: [minion]`). The KataCR alias map is `models/katacr_alias.json`, its keys verified against upstream `models/katacr_label_list.py` (MIT).*
- [x] **[code]** **Synthetic generator** `src/droid_harness/learn/synth.py`: paste sprite cut-outs on empty-arena backgrounds (random position, scale, overlap, HP bars, effects), write YOLO labels. Seed it from KataCR's generative dataset approach.
      *`droid-harness synth`: inference geometry, tile-scaled side-aware KataCR sprites, back-to-front by foot y, real KataCR HP-bar sprites on units only, occlusion/clip-aware labels, Ultralytics layout + `data.yaml` + `label_counts.yaml`. Needs backgrounds: KataCR's empty arenas plus your own crops (only 1 mined so far).*
- [ ] **[data]** Pull KataCR's open detection dataset; check the license and which classes it covers (snapshot from May 2024, so newer cards are missing). Map its class names onto yours.
      *`tools/fetch_katacr.py` is ready (sprites for the alias map, HP bars, empty arenas, and `--part2` real frames; MIT confirmed); nothing is pulled yet — `data/katacr/` is empty.*
- [ ] **[data]** Record 30–50 of your own battles (the recorder already saves `stream.h264`). Extract frames, auto-label them with the first model, fix the labels in CVAT/Label Studio, and repeat.
- [x] **[code]** `src/droid_harness/learn/train_detector.py`: Ultralytics YOLO26-n training on PyTorch-ROCm, exported to ONNX with a static shape. Also export `classes.txt`.
- [ ] **[you]** Install the ONNX Runtime MIGraphX EP (AMD's ROCm docs). Benchmark `OnnxDetector` on MIGraphX vs CPU vs PyTorch-ROCm at your input size, and keep the fastest. Check `detector.provider` in the logs.
      *This host's venv is CPU-only — install PyTorch-ROCm and the MIGraphX EP first (the MIGraphX EP is not in stock pip wheels; expect a custom ORT build or container).*
- [x] **[code]** Crop to the arena: pass only `layout.arena` to the detector and offset the boxes back. This cuts latency and removes false hits on the HUD.
      *Implemented in `perception/pipeline.py::_detect` and proven by `tests/test_detector_m3.py::test_arena_crop_and_flying_offset` (crop coordinates offset back to full-image pixels before ownership sampling and tile mapping).*
- [ ] **[code]** Flying units: the box bottom isn't the ground point for air units. Add a per-class foot offset (from the card DB `flying` flag, tuned on frames).
      *Mechanism exists (`flying_foot_frac` + the `flying` flag, tested at 0.5); tuning the offset on real frames is pending a trained model.*
- [ ] **[code]** Calibrate the HSV ranges in `perception/ownership.py` on real HP bars. Find out whether real troops show bars before they take damage; if not, rely on the spawn-side prior.
- [ ] **[code]** If crowded fights cause ID switches (`tracker.id_switch_suspects`), swap `perception/tracker.py` for ByteTrack (`supervision`) behind the same interface.
- [x] **[code]** Spell plays: map detected spell effects to plays in `SpawnGrouper`, so opponent elixir and cycle tracking see spells. If effects stay unreliable, set `OpponentElixir.miss_widen_per_s` > 0.
- [x] **[code]** Replay regression: a script that re-runs recorded `.h264` matches through perception and diffs the spawn/play/tower events against the previous model.
      *`droid-harness replay --data ... --out run.json [--baseline old.json]` (`domains/clash_royale/replay.py`). The recorder now writes `stream.pts` (device pts per packet) so replays use exact timing; older recordings fall back to 60 fps.*

**Done when:** you meet your per-class recall target on held-out real frames, the detector runs ≤ 10 ms p95 on the 7900 XT, full-loop p99 is under 100 ms in battle (`loop_p99_ms` in summaries — note the current 97 ms was measured *without* detector cost), and the detector refuses to load when its class list disagrees with the ONNX model metadata.

---

## M4 — Play real games (plan P3)

- [ ] **[you]** Build the real card DB with `python tools/build_card_db.py ~/src/cr-api-data`. Check the cost and deploy time of every deck and threat card against the current game, and fix stale entries by hand (the source dates from 2023).
- [ ] **[you]** Fill `tier0.spell_tower_damage` in the strategy file from **your** card levels (in-game card info).
- [ ] **[code]** Verify motion units: compare tracker velocities with the DB `speed_tiles_s` (the builder assumes speed/60 = tiles/s). Check spell travel times against recorded video and adjust `projectile_speed_tiles_s`.
- [ ] **[code]** Tune actuation on the device: tap hold/gap in `CRActionMapper` (currently 20/20 ms), tap-tap vs drag, and when pre-selection helps. Watch *Pointer location* and the execution failures in triage.
- [x] **[code]** **Execution retry:** if the card hasn't left the hand within ~300 ms, re-tap once (log it). Right now a missed tap only blocks new actions until `pending_timeout_ms`.
      *`CRPolicy(retry_after_ms=400)`: one re-tap, only if the current frame still shows that card in that slot and it's affordable (a late hand read can't make it tap whatever slid in). Logged with a `retry:` reason; `policy.retries` counts them. Caveat until selected-state templates exist: if the first tap selected the card but the placement missed, the retry's card tap may deselect it.*
- [x] **[code]** Seed the counter table (`configs/strategies/counters/hog-cycle-v1.csv`) for the Training Camp bot's decks. Current values are placeholders.
      *`counters/training-camp-v2.csv` (used by `training-camp-v2.yaml` and the blind strategy): 71 rows of your 8 cards vs your own deck's units and common low-arena threats. v1 pointed at the hog-cycle copy, where nothing countered Giant, Knight, Mini P.E.K.K.A or Princess. Values are hand-set starting points to tune; extend once you know the trainer's cards.*
- [ ] **[code]** Calibrate placement tiles (`placement.building_pull`, `bridge_tiles`, `back_tiles`, `tier0.leak_tiles`) on the real arena.
- [ ] **[you]** Play 50 Training Camp matches. Run `analyze`. Read the triage mix, the lowest-value rules and a few loss stories, then fix the obvious problems.

**Done when:** the bot beats Training Camp consistently over 50 matches, and every decision is logged with its candidates, posture, predicted tile and latency.

---

## M5 — Evaluation you can trust (plan P4)

- [x] **[code]** **Experiment runner** `src/droid_harness/flywheel/eval/runner.py`: alternates champion/candidate strategy files per match, tags `experiment_id`/`arm` on every match, checks `game_version`, and stops on an SPRT decision (`flywheel/eval/sprt.py`, `inclusion.py`). Write results to the ledger and the warehouse `experiments` table.
      *Implemented `ExperimentRunner` (alternation, SPRT stop, game-version void, ledger + warehouse writes) with unit tests, plus `droid-harness run-experiment` which plays one Training Camp match per arm with `--experiment-id`/`--arm` until the SPRT decides. Shadow mode / replay-diff CLI / retention / dashboard still open.*
- [ ] **[code]** **Shadow mode:** run the candidate policy alongside the champion on the same `GameState` and log its choice (`shadow_strategy_id`, `shadow_choice_json`, already in the schema). Add a report: disagreements × engagement value.
- [x] **[code]** **Replay-diff CLI:** `droid-harness replay-diff --champion A.yaml --candidate B.yaml --data data/matches --expected-pct N`, wrapping `flywheel/eval/replay_diff.py`.
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

## Recovery

- [x] **[code]** App watchdog: `BotRunner` checks `pidof com.supercell.clashroyale` every 10 s and relaunches a dead game (aborting the match as `app_died`). It only acts after it has seen the app running once, so a phone without `pidof` can't cause a relaunch loop.

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
