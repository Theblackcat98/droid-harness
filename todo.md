# todo.md — plan, status and run log

**Target:** the bot plays Clash Royale unattended on a real phone; every match is logged, analyzed and triaged; strategy changes are accepted or rejected by a real paired SPRT. Stretch goals follow (Practice gauntlet, two-phone self-play, learned policy, general Android).

How to use this file: the top is what to do next, the middle is the milestone checklist, the bottom is the dated run log (newest first). How things work, how to run them and the strategy playbook are in [`README.md`](README.md).

Tags: **[you]** needs your hands, phone or account · **[code]** pure code work · **[data]** collecting or labeling. Each milestone ends with **Done when**; don't start the next one until it's true.

> Account rule: the repo plan says throwaway-only, never ladder. **Operator override 2026-11-26:** ladder IS approved on `theblackcat99` (throwaway, nothing to lose). Revisit if the account gains real trophies or Supercell flags anything.

---

## Where things stand (2026-11-27)

- **Works on the real phone (`phone-a`):** unattended ladder and Training Camp play with menus, result settling, stuck-screen recovery (calibrated taps → visual buttons → BACK → app restart), a 30 s battle no-progress guard, batch-end settle to HOME. Loop p50 2.5–4.8 ms, p99 4–12 ms with the async PyTorch-ROCm detector (31 classes).
- **Results:** `ladder-vision-v4` went 9W–1L near 0 trophies. At ≈476 trophies, EXP-0001 (v4 vs v6, paired, 16 matches) went champion 2–6, candidate 2–5 (+1 abort): discordant pairs 2–2, LLR −0.05, **inconclusive**. The v6 opening guard demonstrably works (v4 back-built Giant at 1.1–1.9 s in six matches, v6 never did); eight pairs are too few to measure win rate.
- **Why the win rate fell (review, 2026-11-27):** your Princess Towers read 1512 HP (**King Level 2**); every opponent whose towers were read had 1750–1890 (**King Level 4–5**). A 2–3 level gap is roughly a 20–30 % stat deficit on every card, and the policy doesn't model levels at all.
- **Measurement problems found in the review:** opponent tower HP is read in only some frames (10 of the 16 EXP-0001 matches, never in the older `data/matches/` batch), so `tower_hp_diff` has the wrong sign in several losses (a 0–3 loss scored +337, a 0–1 loss +2620) and `tower_hp_increase` fires hundreds of times per match. That difference feeds engagement value, triage, Speed-1 scoring and the Stage-A futility margin.
- **Built 2026-11-27 (plan steps 1–6, see the run log):** trustworthy tower HP, account levels and level-aware policy (v7), a counter prior, Practice/friendly flows, the dense mSPRT, targeted Speed 1, the detector's next class list and unseen-card eval, template harvesting, a value model, and the `autopilot` with a systemd unit and a weekly coding-agent job. Everything new in `policy/` is opt-in and default off; it changed every strategy ID (v4 `61f2df2e1ec1`, v6 `434fb3d8d263`, v7 `f42458bd6f37`) but not a single logged decision of v4 or v6.
- **Not yet run on the phone.** None of the new code has played a match. The first real run is the next step.
- **Tests:** 243 passed, 1 skipped (torch); `ruff` clean.

## Priority queue

1. **[you] Fill in your levels.** Update [`configs/accounts/theblackcat99.yaml`](configs/accounts/theblackcat99.yaml): every deck card's level (only Archers is recorded) and `king_level` after each upgrade session. `doctor` then checks the strategies' spell damage against them.
2. **[you] Close the level gap.** Spend gold on cards that survive into Hog EQ (Musketeer, Knight, Fireball; README §7) and push King Level. Opponents at ~476 trophies were King Level 4–5 against your 2.
3. **[you/live] Validate tower HP on real video.** On `veritas`, where the original match folders kept their `stream.h264` (the repo copies don't): `droid-harness hp-audit --data data/matches-exp-v4-v6 --crops data/hp-crops`, then play 3–5 matches and check `tower_hp_reads` in `summary.json` covers all six towers. Only then trust `tower_hp_diff` and the dense test.
4. **[you/live] First autopilot cycle, watched.** `droid-harness autopilot --testbed training_camp --test dense --cycles 1 --ui`. Check the report in `reports/latest.md`, then install the service (`tools/systemd/droid-autopilot.service`), set an ntfy topic in it, and run with `--cycles 0`.
5. **[you] Calibrate Practice Mode** (battle-log Practice buttons, `PRACTICE_CONFIRM` screen; `doctor --mode practice` lists them) so the autopilot can test on fixed real decks instead of the Trainer.
6. **[you/live] v7 vs v4.** Once levels are filled in: `droid-harness experiment --champion v4 --candidate v7 --mode training_camp --test dense --ui`, then confirm on the ladder.
7. **[data/you] Next detector.** Train with `models/classes-next.txt` + `katacr_alias-next.json` + `katacr_generic-next.json` on `veritas` (README §10); gate with `eval-detector` and compare `eval-unseen` before/after.
8. **[you] Review the weekly agent's PRs** (Saturdays). Merge only what you'd defend; the autopilot then tests it.

## Review proposals R1–R8 (from 2026-11-27; status)

Each is a separate, testable change. Anything under `policy/` changes every strategy ID, so those shipped as one versioned release.

- **R1 [code] Opponent tower HP. Done in code; live check open.** Per-tower label search with locked offsets, backoff and misread correction; `tower_hp_diff` only when all six towers were read; the dense margin skips an unknown HP term; `hp-audit` CLI. Open: run `hp-audit` on real video (priority 3).
- **R2 [code] Account levels as data. Done.** `configs/accounts/<account>.yaml`; `account_levels` in every `meta.json`; experiments void on a level change; `doctor` checks levels against `tier0.spell_tower_damage`. Open: fill in the levels (priority 1).
- **R3 [code, policy] Level-aware numbers. Done (opt-in).** `level_aware` scales HP/damage for kill-aware spells and Tier 0; King Levels inferred from Princess HP on both sides; `level_gap_respect` leans to defense when under-leveled. In `ladder-vision-v7`. Open: test it (priority 6).
- **R4 [you/code] Lower-variance testbed. Code done; calibration open.** `practice` flow (both arms of a pair face the same battle-log entry), `friendly_host`/`friendly_guest` flows, per-device flow overrides, `doctor --mode`. Open: calibrate the buttons (priority 5); a second phone for self-play.
- **R5 [code] Speed 1 as a targeted filter. Done.** `tune-targeted` with five pattern predicates, 1–2 knobs, a removal floor and a collateral budget; the autopilot uses it.
- **R6 [data/code] Unseen cards. Partly done.** `classes-next.txt` (86 classes incl. five spell effects), `check-classes`, `eval-unseen`. Skipped on purpose: estimating an unknown spawn's cost from unit count and size, because the play event carries no size signal and naming the cards is the real fix. Open: train the next detector (priority 7); a leave-one-card-out model for `eval-unseen`.
- **R7 [code, policy] Counters that generalize. Done (opt-in).** `counter_prior` gives missing CSV pairs an effectiveness from card attributes; a CSV row always wins. Blueprints in README §7 now say to turn it on.
- **R8 [code] One hypothesis per experiment. Done in process.** Targeted candidates change 1–2 knobs; the weekly agent adds one rule per PR; `--test dense` uses a continuous paired outcome.

---

## M0 — Hardware and accounts

- [x] **[you]** Pick a dedicated Android 12+ phone with a hardware H.264 encoder and a 90–120 Hz screen. Enable Developer options → USB debugging, Stay awake, *Pointer location* (for debugging).
      *Used the provided OnePlus N10 (BE2028), Android 11, 1080×2400, 90 Hz mode; not 12+, but everything below works. USB debugging was already on.*
- [x] **[you]** USB 3 cable direct to the PC (no hub at first). Something to keep the phone cool (fan or cooling clip).
      *Direct USB connection (`adb devices` → `741be240`). No cooling clip available in this environment.*
- [x] **[you]** Create throwaway account #1 (its own Supercell ID). Install Clash Royale, finish the tutorial, unlock Training Camp.
      *Account `theblackcat99` (0 trophies, No Clan) was already installed and unlocked.*
- [x] **[you]** Pick the v1 deck in the game. It must match `configs/strategies/hog-cycle-v1.yaml` (no champion, no evolutions), or edit the deck list there.
      *The account only owns the starter deck, so `configs/strategies/training-camp-v1.yaml` (and later `ladder-vision-v1`..`v3`) was added with the real 8 cards (later corrected from `Princess` to `Archers` and upgraded Archers Lv2 → Lv3 on `theblackcat99`). `hog-cycle-v1.yaml` is left as the reference strategy.*
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
- [x] **[you]** Run `droid-harness stream --profile configs/device_profiles/phone-a.yaml --seconds 30 --save-frame data/frame_home.png`. Expect frames while the screen moves and none while it's static; both are normal.
      *Verified on `phone-a`: streamed `464×1024 h264` at ~57.8 fps moving (decode p50 ≈ 3.8 ms) and across 30+ live matches (see the bring-up entry in the run log).*
- [ ] **[you]** Run `scrcpy --list-encoders` and set `video_encoder` to the hardware H.264 encoder. A/B with and without `codec_options`.
- [ ] **[you]** Loopback test: `droid-harness loopback --profile ... --x <quiet spot> --y <quiet spot> --samples 200`. Record p50/p99 and set `exec_latency_ms` to the p50.
      *Attempted on `741be240`: OxygenOS hides synthetic `show_touches` dots (timed out 50/50); kept `exec_latency_ms: 40` in `phone-a.yaml`.*
- [ ] **[you]** Check display-off: set `turn_display_off: true`, stream a Training Camp battle, and confirm frames keep coming and the game keeps animating with the panel dark. If they don't, set it to false and note the phone model.
- [ ] **[you]** Resolution sweep: `max_size` 720 / 1024 / 1280. Record loopback p50 and decode time for each. Pick the final size after M3's detector check.
- [x] **[code]** Add a `--duration` soak mode to `stream` that runs 30 min and reports reconnects, so you can confirm static menus don't trigger false reconnects.
      *`droid-harness stream --profile ... --duration 1800` (`core/telemetry/soak.py`): runs liveness with `in_battle=False`, reconnects like `BotRunner`, reports frames, reconnects and findings by kind. Tested against the fake server for a static stream (0 reconnects) and a killed session (reconnects).*

**Done when:** the 30-minute stream has 0 false reconnects, loopback p99 is recorded, and a tap at known pixels lands where *Pointer location* shows it.

---

## M2 — See the HUD and drive the menus (plan P1)

Calibration, per phone, from saved frames:

- [ ] **[you]** Save frames of each screen: HOME, the menu that holds Training Camp, the Training Camp confirm, LOADING, IN_BATTLE (early, mid, overtime), RESULT_WIN / RESULT_LOSS / RESULT_DRAW, and common POPUPs (chests, offers, season pass).
      *Captured and committed HOME, MENU, TRAINING_CONFIRM, LOADING, MATCHMAKING, TROPHY_ROAD (incl. Bone Pit Arena-lock modal), IN_BATTLE (early + mid), RESULT_WIN and RESULT_LOSS. Still missing overtime, RESULT_DRAW and common POPUPs.*
- [x] **[you]** ROIs: copy `configs/domains/clash_royale/rois/example.yaml` → `rois/phone-a.yaml`. Measure hand slots ×4, next card, elixir bar, clock, 6 tower-HP fields, arena rectangle and menu buttons as fractions of the frame. Tune `elixir_hsv_lo/hi` on a real frame.
      *`rois/phone-a.yaml` calibrated on 1080×2400 frames (and committed); elixir HSV [135,80,120]–[175,255,255], plus `battle`, `dismiss`, and `trophy_ok` buttons. The stable screen-ID region is the top 18% (clock/opponent), not the old top/bottom halves.*
- [ ] **[you]** Templates: run `droid-harness template ... --state affordable|greyed|selected` for all 8 deck cards in every state, plus the next-card slot.
      *18 clean templates across all 8 deck cards (affordable/greyed; stale zoomed `archers__0` and duplicates removed) committed in `templates/phone-a/`, paired with `CardMatcher` `local_znorm` + bottom-center/border mask (dropped active-battle misses on `calib01` from 46.45% → 0.14%). Selected-state templates are still missing.*
- [ ] **[you]** Screens: run `droid-harness screen ... --name <SCREEN>` for every screen above (2–3 frames each).
      *Registered `HOME` (2), `MENU`, `TRAINING_CONFIRM`, `IN_BATTLE` (3), `LOADING`, `MATCHMAKING`, `TROPHY_ROAD` (2), `RESULT_WIN` (3), `RESULT_LOSS` (2); missing DRAW/POPUP references.*
- [ ] **[you]** Digits: crop `0.png`…`9.png` from tower-HP text into `configs/domains/clash_royale/digits/phone-a/`.

Code gaps to close in this milestone:

- [x] **[code]** **Menu flow for the real UI.** Update `DEFAULT_FLOWS["training_camp"]` in `domains/clash_royale/menus.py` to the button names you calibrated. The June 2025 update moved Training Camp into the menu, so verify the path. Add popup handling (one `close` per popup type).
      *Verified path: HOME → `menu` (hamburger, top-right) → `training_camp` (MENU) → `confirm` (TRAINING_CONFIRM dialog) → battle. The existing `DEFAULT_FLOWS` names already match the layout; `POPUP → close` exists and `close` is in the layout. The only popup seen so far (Next Reward) was non-modal.*
- [x] **[code]** **Destroyed towers.** `perception/pipeline.py::_read_towers` never sets a tower to 0; unreadable digits just keep the last value. Add destroyed-tower detection (template of the rubble/crown state per tower ROI, or detector classes). Crowns, `pocket_open` and the king/princess targeting all depend on it.
      *Code done: `perception/towers.py::TowerStateReader` template-matches each tower ROI against per-phone `alive`/`rubble` references and marks a tower destroyed after 2 rubble reads in a row (permanent for the match; digits can't revive it). `run --towers DIR` wires it in. A `tower_rubble` detection (older models) also marks the nearest tower. Crowns, `pocket_open` and king targeting follow from `towers[name] == 0`.*
- [x] **[you]** Capture tower templates: `droid-harness tower-template --frame <battle frame> --layout rois/phone-a.yaml --tower all --state alive --out configs/domains/clash_royale/towers/phone-a`, then `--tower opp_left --state rubble` (etc.) from frames after a tower falls.
      *Done: 10 `alive`/`rubble` templates captured and committed in `configs/domains/clash_royale/towers/phone-a/`.*
- [x] **[code]** **Match clock.** The layout has a `clock` ROI but the pipeline never reads it. Read it at 1 Hz to align match t=0 with the game (screen detection lags a few frames, and the intro countdown shifts it). Flag drift over 1 s as a perception warning. Double elixir at 2:00 depends on this.
      *Done: `pipeline.py` reads the `clock` ROI at most 1 Hz, parses `M:SS`, exposes `match_t_s` + `clock_drift_s` on the state, warns `clock_drift` past 1 s. Tested on synthetic frames.*
- [x] **[code]** **Arena geometry.** `app.py::BotRunner` uses `Arena()` defaults. Add `--arena configs/domains/clash_royale/arena/phone-a.yaml` (load with `Arena.load`) and calibrate river, bridges and tower tiles from the same landmarks as the homography.
      *Added `--arena` to `droid-harness run` and `RunConfig`; `configs/domains/clash_royale/arena/phone-a.yaml` was fitted from the same six landmarks as `tiles/phone-a.yaml` (river y 15.0–16.5, bridges x 3.6/14.4).*
- [ ] **[code]** **Golden frames.** Commit 200+ labeled real frames (hand, elixir, towers, screen ID) under `tests/golden/phone-a/` (gitignored if large; keep an index). Add a test that fails if HUD accuracy drops below 99.5%.
- [x] **[code]** Elixir start: the first HUD read seeds both models; confirm on real frames that the first `IN_BATTLE` frame shows the starting bar and not the intro.
      *Fixed in `pipeline.py`: skips `frame_i == 0` reset when `hud < 1.0` and syncs both `elixir` & `opp_elixir` upward during `t <= 3.0s` before the first event (`0–1` `elixir_discrepancy` per match across `ladder-verify`).*

Run it:

- [x] **[you]** `droid-harness run ... --strategy configs/strategies/training-camp-blind-v1.yaml --matches 5` **without** `--detector`. It navigates menus, recognizes battles and results, reads hand and elixir, and attacks blind (Giant at the bridge, support behind, spells on the tower). `training-camp-v1` cannot attack without a detector: its opponent-elixir estimate never narrows, so the attack rule scores −0.1.
      *Done 2026-11-26: 5/5 then 10 matches (9–1) = **16–1 record**, blind_push firing 40–60×/match, loop ~42 ms, post-match OK ROI fixed (5 min → 2.2 min/match). Grind PAUSED for the ladder pivot; resume for ranker training data.*
- [ ] **[you]** Run `droid-harness analyze --data data/matches --cards ...` and read the warning counts. Fix calibration until `perception_warnings` is near zero.

**Done when:** 20 unattended Training Camp matches complete (menus, battle, result, next match) with correct results in `summary.json` and HUD golden-set accuracy ≥ 99.5% in every card state.

---

## M3 — See the units (plan P2)

Detector foundations (`models/classes.txt`, `models/katacr_alias.json`, `models/katacr_generic.json`, `learn/synth.py`, `learn/train_detector.py`, `tools/fetch_katacr.py`, `tools/extract_backgrounds.py`) are implemented and the first YOLO26-n detector (`18` side-aware + catch-all + effect classes) is trained on `veritas` (epoch 40; `best.pt` epoch 36 `mAP50=0.797`) and running live via `TorchDetector` PyTorch-ROCm (`7.90 ms p50 / 9.83 ms p95`). Open work in M3 is fine-tuning weak/under-represented `*_opp` classes on labeled real frames plus flying-offset/HP-bar-HSV tuning.

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
      *Now 18 classes: 6 starter-deck units × 2 sides + 2 spell effects + `other_ground_{me,opp}` + `other_air_{me,opp}` (no towers, no `minions`: Minions spawn three `minion` units; local `cards.json` uses `unit_classes: [minion]`). The KataCR alias/generic maps are `models/katacr_alias.json` and `models/katacr_generic.json`, keys verified against upstream `models/katacr_label_list.py` (MIT).*
- [x] **[code]** **Synthetic generator** `src/droid_harness/learn/synth.py`: paste sprite cut-outs on empty-arena backgrounds (random position, scale, overlap, HP bars, effects), write YOLO labels. Seed it from KataCR's generative dataset approach.
      *`droid-harness synth`: inference geometry, tile-scaled side-aware KataCR sprites, back-to-front by foot y, real KataCR HP-bar sprites on units only, occlusion/clip-aware labels, Ultralytics layout + `data.yaml` + `label_counts.yaml`. Needs backgrounds: KataCR's empty arenas plus your own crops (only 1 mined so far).*
- [x] **[data]** Pull KataCR's open detection dataset; check the license and which classes it covers (snapshot from May 2024, so newer cards are missing). Map its class names onto yours.
      *Done on `veritas` (2026-11-26): MIT confirmed; converted `data/katacr` via `convert-katacr` (`data/katacr-real`) and merged with synth data on 28 real backgrounds into `data/train.yaml`.*
- [ ] **[data]** Record 30–50 of your own battles (the recorder already saves `stream.h264`). Extract frames, auto-label them with the first model, fix the labels in CVAT/Label Studio, and repeat.
- [x] **[code]** `src/droid_harness/learn/train_detector.py`: Ultralytics YOLO26-n training on PyTorch-ROCm, exported to ONNX with a static shape. Also export `classes.txt`.
- [x] **[you]** Install the ONNX Runtime MIGraphX EP (AMD's ROCm docs). Benchmark `OnnxDetector` on MIGraphX vs CPU vs PyTorch-ROCm at your input size, and keep the fastest. Check `detector.provider` in the logs.
      *Benchmarked 100× on idle RX 7900 XT (`440×707` arena crop → `640×384`): `OnnxDetector` CPU `26.78 ms p50 / 53.40 ms p95` vs new `TorchDetector` PyTorch-ROCm `7.90 ms p50 / 9.83 ms p95` (bit-identical detections; MIGraphX EP not needed since `TorchDetector` meets `≤ 10 ms p95`).*
- [x] **[code]** Crop to the arena: pass only `layout.arena` to the detector and offset the boxes back. This cuts latency and removes false hits on the HUD.
      *Implemented in `perception/pipeline.py::_detect` and proven by `tests/test_detector_m3.py::test_arena_crop_and_flying_offset` (crop coordinates offset back to full-image pixels before ownership sampling and tile mapping).*
- [ ] **[code]** Flying units: the box bottom isn't the ground point for air units. Add a per-class foot offset (from the card DB `flying` flag, tuned on frames).
      *Mechanism exists (`flying_foot_frac` + the `flying` flag, tested at 0.5); tuning the offset on real frames is pending a trained model.*
- [ ] **[code]** Calibrate the HSV ranges in `perception/ownership.py` on real HP bars. Find out whether real troops show bars before they take damage; if not, rely on the spawn-side prior.
- [ ] **[code]** If crowded fights cause ID switches (`tracker.id_switch_suspects`), swap `perception/tracker.py` for ByteTrack (`supervision`) behind the same interface.
- [x] **[code]** Spell plays: map detected spell effects to plays in `SpawnGrouper`, so opponent elixir and cycle tracking see spells. If effects stay unreliable, set `OpponentElixir.miss_widen_per_s` > 0.
- [x] **[code]** Replay regression: a script that re-runs recorded `.h264` matches through perception and diffs the spawn/play/tower events against the previous model.
      *`droid-harness replay --data ... --out run.json [--baseline old.json]` (`domains/clash_royale/replay.py`). The recorder now writes `stream.pts` (device pts per packet) so replays use exact timing; older recordings fall back to 60 fps.*

**Done when:** you meet your per-class recall target on held-out real frames, the detector runs ≤ 10 ms p95 on the 7900 XT, full-loop p99 is under 100 ms in battle (`loop_p99_ms` in summaries — note the current 97 ms was measured *without* detector cost), and the detector refuses to load when its class list disagrees with the ONNX model metadata. *(Current status after `fc94e6b`: `TorchDetector` `9.83 ms p95 <= 10 ms`, live `loop_p99_ms` `4.01–4.36 ms << 100 ms`, and embedded class cross-check are all met; per-class recall on `archer_opp`, `knight_opp`, and the 4 `n=0` KataCR-val `*_opp` classes remains the open gate pending real-frame labeling & fine-tuning.)*

---

## M4 — Play real games (plan P3)

- [x] **[you]** Build the real card DB with `python tools/build_card_db.py ~/src/cr-api-data`. Check the cost and deploy time of every deck and threat card against the current game, and fix stale entries by hand (the source dates from 2023).
      *Built 118-card `configs/domains/clash_royale/cards.json` and patched `arrows` (`damage: 144, radius_tiles: 3.5, ct_damage_pct: 20.8`). Remaining non-deck threat stats still to audit over time.*
- [ ] **[you]** Fill `tier0.spell_tower_damage` in the strategy file from **your** card levels (in-game card info).
- [ ] **[code]** Verify motion units: compare tracker velocities with the DB `speed_tiles_s` (the builder assumes speed/60 = tiles/s). Check spell travel times against recorded video and adjust `projectile_speed_tiles_s`.
- [ ] **[code]** Tune actuation on the device: tap hold/gap in `CRActionMapper` (currently 20/20 ms), tap-tap vs drag, and when pre-selection helps. Watch *Pointer location* and the execution failures in triage.
- [x] **[code]** **Execution retry:** if the card hasn't left the hand within ~300 ms, re-tap once (log it). Right now a missed tap only blocks new actions until `pending_timeout_ms`.
      *`CRPolicy(retry_after_ms=400)`: one re-tap, only if the current frame still shows that card in that slot and it's affordable (a late hand read can't make it tap whatever slid in). Logged with a `retry:` reason; `policy.retries` counts them. Caveat until selected-state templates exist: if the first tap selected the card but the placement missed, the retry's card tap may deselect it.*
- [x] **[code]** Seed the counter table (`configs/strategies/counters/hog-cycle-v1.csv`) for the Training Camp bot's decks. Current values are placeholders.
      *`counters/training-camp-v2.csv` (used by `training-camp-v2.yaml` and the blind strategy): 71 rows of your 8 cards vs your own deck's units and common low-arena threats. v1 pointed at the hog-cycle copy, where nothing countered Giant, Knight, Mini P.E.K.K.A or Princess. Values are hand-set starting points to tune; extend once you know the trainer's cards.*
- [ ] **[code]** Calibrate placement tiles (`placement.building_pull`, `bridge_tiles`, `back_tiles`, `tier0.leak_tiles`) on the real arena.
- [ ] **[you]** Play 50 Training Camp matches. Run `analyze`. Read the triage mix, the lowest-value rules and a few loss stories, then fix the obvious problems.
      *16–1 banked 2026-11-26 (blind). Remaining 34+ when the grind resumes — every win is ranker training data (M8 needs 50+).*

**Done when:** the bot beats Training Camp consistently over 50 matches, and every decision is logged with its candidates, posture, predicted tile and latency.

---

## M5 — Evaluation you can trust (plan P4)

- [x] **[code]** **Experiment runner** `src/droid_harness/flywheel/eval/runner.py`: alternates champion/candidate strategy files per match, tags `experiment_id`/`arm` on every match, checks `game_version`, and stops on an SPRT decision (`flywheel/eval/sprt.py`, `inclusion.py`). Write results to the ledger and the warehouse `experiments` table.
      *Implemented `ExperimentRunner` (alternation, SPRT stop, game-version void, ledger + warehouse writes) with unit tests, plus `droid-harness run-experiment` which plays one Training Camp match per arm with `--experiment-id`/`--arm` until the SPRT decides. Shadow mode / replay-diff CLI / retention / dashboard still open.*
- [x] **[code]** **Shadow mode:** run the candidate policy alongside the champion on the same `GameState` and log its choice (`shadow_strategy_id`, `shadow_choice_json`, already in the schema). Add a report: disagreements × engagement value.
      *Done: `flywheel/shadow.py::ShadowRunner` + `report()` (agreement counting, rule-ranked). NOT yet wired into `run`/`run-experiment` as a live mode — that wiring is still open.*
- [x] **[code]** **Replay-diff CLI:** `droid-harness replay-diff --champion A.yaml --candidate B.yaml --data data/matches --expected-pct N`, wrapping `flywheel/eval/replay_diff.py`.
- [ ] **[code]** **Retention job:** after 7 days, transcode losses and flagged engagements to ~1 Mbps H.265, export anomaly keyframes for labeling, and delete the rest. Raw video is ~60 MB/min.
- [x] **[code]** A dashboard notebook (`notebooks/dashboard.py`, marimo) over DuckDB: win rate with CI by strategy × opponent × game version, play value by rule × posture, triage mix, loop p99 per day.
      *Done as plain-python marimo-compatible `notebooks/dashboard.py` (`__main__` prints text; no marimo import). Reads `data/matches/*/summary.json` + `decisions.jsonl` directly (warehouse hookup still open).*
- [ ] **[you]** Run the first real SPRT: champion vs one small, hand-made change, bounds [0.50, 0.58]. Expect roughly 200 games.
      *EXP-0001 ran on the ladder 2026-11-27 (v4 vs v6, 16 matches): inconclusive at its cap. A decided SPRT is still open.*

**Done when:** one SPRT runs end-to-end unattended, its result is in the ledger, and no match in it spans a game update.

---

## M6 — Coach and tuning (plan P5)

- [ ] **[code]** `droid-harness coach` CLI: analyze → STRATEGY-only loss stories → `coach.context_pack` → `AnthropicClient` → `check_guardrails` → `apply_patch` on a branch → replay-diff gate → queue the experiment.
      *Partial: `flywheel/coach/` has proposal/context/guardrail pieces but no `coach` CLI command and no API client. The L2 advisor (`core/runtime/advisor.py`, stdlib HTTP + `--api-base/--model`) is the closest working API pattern to copy.*
- [ ] **[code]** Wire Optuna's `play_block` (`flywheel/tuning/optuna_study.py`) to the experiment runner. Tuning produces candidates only; they still go through M5.
- [ ] **[you]** Accept or reject at least one coach proposal and one tuned candidate via SPRT.

**Done when:** ≥ 1 coach proposal and ≥ 1 tuned candidate have gone through the full funnel.

---

## Review fixes and Giant beatdown v3 (2026-11-26)

- [x] **[code]** Stale hand: a slot that fails to read keeps its last card (so the play is still detected) but is marked stale in `GameState.hand_fresh`; `affordable_hand`, the `stale_slot` constraint and the tap retry all skip stale slots. Before, a retry could tap whatever card had slid into the slot.
- [x] **[code]** Blind strategy: only a confirmed *spell* uses up the push's spell (a support troop used to cancel it).
- [x] **[code]** `Card.can_hit(flying)`: defense and emergency defense never pick a card that can't hit the threat (Giant never defends; Mini P.E.K.K.A never answers air), whatever the counter table says.
- [x] **[code]** `path.advance` takes tower-alive state: after a princess tower falls, threats are led toward the king.
- [x] **[code]** **Experiments compare against the champion.** `run-experiment --test pairs` (default): randomized blocks of two (one match per arm, random order) tagged with `pair_id`; SPRT on discordant pairs; draws are scored, never dropped; aborted slots are replayed (pairs dropped after `max_slot_retries`); abort-rate imbalance between arms is flagged in the ledger notes. `h2h` (self-play) refuses non-head-to-head games; `fixed` (vs a fixed rate) is documented as never-for-promotion. Old names map: `sprt_h2h` → `fixed`, `sprt_paired` → `h2h`.
      *Not changed, on purpose: `replay_diff` still resets both policies per logged state. Carrying policy state across a match would replay the champion's history into the candidate (pending actions and cooldowns from moves the candidate never made); per-state comparison is the honest counterfactual.*
- [x] **[code]** KataCR converter splits by source sequence (`<video>/<episode>`), with `groups.json` for audit; `val_frac` must be in (0, 1). Before, its `data.yaml` pointed train and val at the same folder.
- [x] **[code]** `droid-harness dataset` combines sources into one `data.yaml` and refuses any image in both train and val; `frames` exports your recordings as arena crops split by match, optionally pre-labeled by the current model.
- [x] **[code]** `droid-harness eval-detector`: per-class recall/precision through `OnnxDetector` (as the bot runs it); fails unless every `*_opp` class with ≥ `--min-support` examples reaches `--min-threat-recall` (an under-sampled threat class fails too).
- [x] **[code]** Catch-all classes `other_ground_*`/`other_air_*` from `models/katacr_generic.json` (`tools/build_katacr_generic.py`; fixes KataCR's ground/air split for balloon, lava hound, …). Generic counter rows in `counters/training-camp-v3.csv`. Synth samples classes evenly so the catch-alls don't drown the 6-sprite classes.
- [x] **[code]** `training-camp-v3.yaml` (E1: Giant is the only win condition; v2 kept for history, marked superseded) and one child per experiment, all off by default in code: `-e3-tank-ahead` (Giant leads counter-pushes), `-e4-cycle-gate` (scale Giant pushes ×0.5 when a known Giant counter may be in their hand, ×1.3 when their deck is known and the counters are in their last four), `-e5-reserve` (keep 3 elixir unless they're low; back-build Giant at 9+ on a quiet board).
- [x] **[code]** A strategy that sets `learned_ranker` refuses to run (the live runner has no loader, so it was silently ignored). Optuna's default space adds `defend_horizon_s`, `spell_value_threshold`, `attack_blocked_by_threat_eta_s` and DEFEND_LEAD defense.
- [x] **[code]** E6: kill-aware spell value (which units die, tower damage, elixir trade) instead of HP-weighted cost.
      *Implemented via `kill_aware_spells` in `policy/rules/spells.py` (`ladder-vision-v4`/`v5`), verified in `tests/test_cr_policy.py`.*
- [ ] **[code]** E7: lane choice from tower HP, their counters' position in cycle, and pressure (today: lower princess HP, then fewer enemies).
- [ ] **[code]** E8: learned ranker as a tie-breaker among legal high-scoring plays: loader + feature schema in the strategy hash, logged scores, fallback; train on warehouse decisions after E1–E5 settle.
- [ ] **[code]** Emergency defense on projected damage (needs unit DPS in the card DB), not only below 35% tower HP.

## Recovery

- [x] **[code]** App watchdog: `BotRunner` checks `pidof com.supercell.clashroyale` every 10 s and relaunches a dead game (aborting the match as `app_died`). It only acts after it has seen the app running once, so a phone without `pidof` can't cause a relaunch loop.

## Studio GUI (done 2026-11-27)

- [x] **[code]** Telemetry tap on the brain loop (`BrainLoop(on_step=...)`, references only) and a rate-capped publisher thread (`ui/telemetry.py`).
- [x] **[code]** One snapshot schema for live and replay (`ui/snapshot.py`): HUD, opponent cycle, units, detections, ranked candidates with `score × weight × posture = total`, act threshold, latency, touches, experiment state.
- [x] **[code]** Browser page with phone + overlays, tactical map, strategy panels, replay elixir tape and transport (`ui/static/studio.html`); stdlib HTTP server with SSE (`ui/server.py`); gzip for match timelines.
- [x] **[code]** `--ui` on `play`, `run`, `experiment`, `run-experiment`, `sim`; `watch` (observe-only: no taps, no BACK, no app restarts); `inspect` (replay, video decoded once at 10 fps when kept); `--native` via pywebview (`[gui]` extra).
- [ ] **[you]** Try `droid-harness play --ui` on `phone-a` and `inspect` over SSH from `veritas`; note anything misdrawn (homography, ROIs) or missing.
- [ ] **[code]** Log detector boxes (sampled) so recorded video can show them too; today replay projects tracked units from their logged tiles.
- [ ] **[code]** Shadow column in the scoreboard: what a second strategy would have chosen at the same state (`flywheel/shadow.py`).

---

## Autopilot and value model (code done 2026-11-27)

- [x] **[code]** `droid-harness autopilot`: farm → diagnose (value model, digest) → propose (`tune-targeted`) → test (paired experiment on a testbed) → promote (`CHAMPION` pointer) → report (`reports/`), `--cycles 0` until `data/autopilot/STOP`, `--ui`.
- [x] **[code]** Notifications (`--notify-url`, ntfy-style POST) and publishing (`--publish-git`: reports, ledger, `CHAMPION`, `auto/` to the `autopilot/reports` branch through a separate worktree).
- [x] **[code]** `tools/systemd/droid-autopilot.service` (waits for the phone, runs `doctor`, restarts on failure).
- [x] **[code]** Weekly coding-agent scheduled task (Saturdays 14:17 Pacific): reads the published reports, adds one opt-in rule with a test and a new strategy YAML, opens a PR, never merges; skips weeks with no new report or an open PR.
- [x] **[code]** Value model (`train-value`, `learn/value_model.py`): validation AUC 0.69 on 56 committed matches; Studio win-chance line; digest swing moments.
- [ ] **[you]** First watched cycle, then the service (priority 4).
- [ ] **[code]** Use the value model as a dense outcome (win-probability gained per pair) once it's validated on 200+ matches; feed its features to the M8 ranker.
- [ ] **[code]** Rotate the Practice gauntlet beyond the first four battle-log entries and record each entry's deck.

**Done when:** the autopilot has run unattended for a week, promoted or rejected at least one candidate on its own, and the weekly job has opened a PR from its reports.

---

## Definition of done: "fully running harness in a real bot game"

- [ ] The bot plays Training Camp matches unattended for hours with automatic recovery (reconnect, popups, app restart).
- [ ] Loop p99 is under 100 ms in battle; HUD accuracy ≥ 99.5%; the detector meets its recall target.
- [ ] Every match is recorded, analyzed, triaged and in the warehouse, with a game version and balance epoch.
- [ ] At least one strategy change has been accepted or rejected by SPRT and recorded in the ledger.

*Current standing (after `fc94e6b`): `loop_p99_ms` (`4.01–4.36 ms`) and `TorchDetector` latency (`9.83 ms p95`) satisfy the latency targets, and 10 unattended `--mode ladder` matches finished with 0 aborts/orphans (`9W–1L–0D`); remaining gates are multi-hour unattended stability + retention, 99.5% HUD golden-set test + weak-class detector recall, full warehouse ingestion, and the first live paired SPRT.*
*2026-11-27: the first live paired SPRT ran (EXP-0001, 16 matches) and ended inconclusive at its cap; recovery and settle now handle the stalls seen in long runs.*

---

## After that (stretch)

- **M7 — Practice gauntlet + self-play.** The `practice`, `friendly_host` and `friendly_guest` flows exist (2026-11-27) but their buttons and screens are uncalibrated. Calibrate them, then run two `BotRunner`s with distinct ports, swapping strategy ↔ (phone, account) per `flywheel/eval/selfplay.py`. Check `seat_effect` for device bias.
- **M8 — Learned policy (plan P6).** CORE DONE 2026-11-26: `learn/label_from_video.py` (card-left-hand + matching spawn → weighted state/action rows from recorded matches), `learn/train_ranker.py` (numpy-MLP behavior cloning, static ONNX export `(32,55)->(32,1)`), live loader in `app.py` (`CRPolicy(ranker=...)` rescores candidates). STILL OPEN: first real training on warehouse decisions (needs 50+ wins of data — the farm), SPRT-gate each step (BC → filtered BC → offline RL), feature-schema in the strategy hash, logged ranker scores with fallback.
- **M9 — L2 advisor (optional).** CORE DONE 2026-11-26: `core/runtime/advisor.py` (separate process, OpenAI-compatible API via `--api-base/--model`, posture-only JSON schema, TTL=2×cadence, never blocks), `advise --dry-run` (offline replay of a recorded match). STILL OPEN: point it at a live key (deepseek/zai), measure loop p99 with it on, keep it only if it wins an SPRT, then distill it.
- **M10 — General Android (plan P7).** Test DroidRun Portal on your phones (`domains/android_os/a11y.py`, with uiautomator fallback). Add a GUI-model fallback with the Set-of-Mark overlay. Measure step latency and cache-hit rate on an everyday task suite.

## Risks to watch

- **Game updates** change UI, cards and balance. Re-run the golden set after each update, bump `balance_epoch`, and void running tests.
- **scrcpy upgrades** can change the protocol. Re-pin the version, re-check the structs, and re-run M1.
- **Thermals:** watch `loop_p99_ms` and loopback drift over long runs.
- **Perception errors** look like strategy errors. Trust triage, and fix perception before tuning strategy.
- **Card levels** change the bot's strength without changing any strategy ID (see R2).

---

# Run log (newest first)

## 2026-11-27 — Plan steps 1–6: trustworthy numbers, levels, testbed, autopilot, cards, value model

All code, no phone runs yet. Verified on the committed match data (56 matches) and in tests.

- **Tower HP (R1).** On the committed full-resolution battle frames the opponent labels sat ~16 px above their fixed regions. Each tower's label is now found by a vertical then horizontal search (~14 ms the first time, down from ~60 ms for the first version), locked per tower and kept across matches, with twin Princess towers sharing the vertical offset and a 2 s → 20 s backoff on misses. Three agreeing higher reads correct an earlier misread (`tower_hp_corrected`). `tower_hp_diff` is now `None` unless all six towers were read; `dense_match_score`, Optuna and analysis handle `None`. `hp-audit` reports coverage on recorded video.
- **Levels (R2, R3).** `configs/accounts/theblackcat99.yaml` (King Level 2 measured; Archers 3; others unrecorded). `account_levels` in `meta.json`; experiments void on a level change. `policy/levels.py`: start levels, `1.1^(level − start)`, King Level from Princess HP. `level_aware` and `level_gap_respect` in kill-aware spells, Tier 0 and utility totals. New `ladder-vision-v7` = v6 + `level_aware: true`, `level_gap_respect: 0.08`.
- **Counter prior (R7).** `policy/counter_prior.py`, opt-in `counter_prior`; CSV rows always win.
- **Policy release check.** The release changed every strategy ID: v4 `639c09fe0c3a` → `61f2df2e1ec1`, v6 `ab2d9564c216` → `434fb3d8d263`. Both make identical decisions to before on all 28,744 logged states. Old IDs → names in `configs/strategies/legacy_ids.yaml`; `CHAMPION` points at v4.
- **Testbed (R4).** `practice`, `friendly_host`, `friendly_guest` flows; per-device `flows:` overrides; `doctor --mode`. `--test dense` (mSPRT on the paired dense margin, `--dense-tau`, `min_dense_pairs`).
- **Speed 1 (R5).** `tune-targeted`. On all committed data, v4's pattern hits: `over_defend_bleed` 34, `opening_throw` 9, `bad_counter_timing` 2, `wincon_starved` 1, `spell_inefficiency` 0. The best 1–2 knob fix for `over_defend_bleed` (`defend_trigger_tiles` 13 → 9.25, plus `ignore_chip_max_threat_value`) removes 25 of 34 hits with 0.17 % collateral, in 12 s on CPU. Not tested live: whether defending later is better is what the paired test measures.
- **Cards (R6).** `models/classes-next.txt` (86 classes), `-next` alias/generic maps, `check-classes`, `eval-unseen`; template harvesting (`--harvest-templates`, `promote-templates`). Skipped: the unknown-spawn cost heuristic (no size signal in the play event; naming cards is the real fix).
- **Value model.** `train-value`: 39 train / 17 validation matches, validation AUC 0.69, under-confident at the low end. Studio win-chance badge and tape line; `digest --value` lists swing moments.
- **Autopilot.** `flywheel/autopilot.py`, `flywheel/eval/live.py` (`run_paired`, shared with `experiment`), `droid-harness autopilot`, `tools/systemd/droid-autopilot.service`, the weekly agent scheduled task. Tests run a full cycle with a fake phone (farm, value model, proposal, promotion, report), the rejected path, the STOP file, notification and publishing into a temp repo.
- Tests 243 passed, 1 skipped; `ruff` clean.

## 2026-11-27 — Studio GUI, docs consolidation, strategy review

- Studio built (see the milestone above). Verified against the simulator (`sim --ui`, live view and replay of the recorded sim match with video) and against the committed ladder matches (replay from `decisions.jsonl` alone). Snapshot build 0.6–0.8 ms on the publisher thread; nothing under `policy/` changed, so v4 is still `639c09fe0c3a` and v6 `ab2d9564c216`.
- `docs/` folded into `README.md` (documentation) and this file (plan and log).
- `tests/test_detector_runtime.py::test_torch_detector_and_load_detector` now skips without torch, so CI (which installs no torch) passes again.
- Review findings are in "Where things stand" and R1–R8 above.

## 2026-11-27 — Stuck-screen recovery and EXP-0001 (v4 vs v6)

Device `phone-a`, account `theblackcat99`, game `160635013`, epoch `2026-11`, strategy IDs `639c09fe0c3a` (v4) and `ab2d9564c216` (v6).

**Recovery (code).** The phone was found on a ladder loss screen that added **Play Again** and shifted **OK** right. The calibrated `ok` ROI (`[0.370, 0.815, 0.630, 0.880]`, centre `(232, 868)` at 464×1024) landed in the 23 px gap, so every tap missed; the screen also hashed `UNKNOWN`, so the UNKNOWN flow cycled other wrong taps. `domains/clash_royale/recovery.py` adds the escalation ladder (calibrated buttons → visual buttons, rightmost first → BACK → force-stop/relaunch after the plan is exhausted or 60 s of liveness findings), keeps clearing each new screen until HOME, and requires the real Battle button for HOME (a Trophy Road scroll that hashes as HOME isn't accepted). Exposed as `droid-harness recover` (`--dry-run`, `--stuck-after`, `--battle-stuck-after`); `doctor --live`'s ADB check was fixed in the same pass. Two more stalls fixed: the purple post-match chest reads as a full elixir bar (30 s battle no-progress guard → `aborted:battle_no_progress`), and a transient HOME animation that hashes `IN_BATTLE` could create a phantom match during settle (menu decisions suspended, `_set_screen` can't start/end a match during settle). Tests then: 209 passed, 1 skipped.

**Live validation.**

| Run | Data | Result |
| --- | --- | --- |
| v4, 1 match | `data/matches-fresh-v4b/971f64eb82c2` | win 3–2, settle `RESULT_WIN → OK → HOME` in ~5 s |
| v4, 1 match | `data/matches-fresh-v4c/c11f767dcfde` | loss 0–3, settle `RESULT_LOSS → OK → HOME` in ~5 s |
| v4, 3-match smoke | `data/matches-fresh-v4` | 2 wins (3–1, 3–1), one 660 s chest stall before the fix (`2b3487f2bfd1`) |

`recover` cleared the stuck loss screen to HOME in ~10 s and rejected a Trophy Road reward frame before accepting the real HOME.

**EXP-0001** (ladder, `--test pairs`, p0 0.5, p1 0.58, `max_games` 16, `--futility-pairs 6`, data `data/matches-exp-v4-v6`): champion v4 2W–6L, candidate v6 2W–5L (+1 abort `left_battle:MATCHMAKING`). Discordant pairs 2–2, 3 tied, dense margin −0.539, LLR −0.05 → continue; hit the 16-game cap → ledger `INCONCLUSIVE`. Mechanism check: v4 played `offense.giant_back_build` at 1.1–1.9 s in six matches (`b5f66a`, `49b06e`, `366789`, `d0d019`, `94b696`, `9b68ac`); v6 never did (earliest plays `opening.safe_first_play` at 4.4–4.7 s and a counter-push Giant at 5.2–5.3 s). Throughput: 16 matches in 41 minutes (2.8 min start to start).

**Speed 1 on this batch.** `tune-offline --champion v6 --data data/matches-exp-v4-v6 --trials 300`: best trial (`defense_reserve 1.62`, `elixir_weight 0.20`) was outside the guardrail (18 bad flips vs 3 good broken) → no promotable v7. Dashboard triage OK 86 %, PERCEPTION 14 %, loop p99 6.1 ms. The digest's 55.9 % PERCEPTION share came from end-of-match frames (clock `0:00`, hand intentionally blank), not mid-match failures. Failure clusters: `opening_throw` 11, `wincon_starved` 12, `over_defend_bleed` 15 (83–98 % of the match below 5 elixir), `spell_inefficiency` 10. Both arms ran ~25–28 % in this pool vs v4's 90 % earlier: trophies had climbed to ≈476 and the pool is much harder.

**Data committed:** compact files only (`meta.json`, `summary.json`, `decisions.jsonl`) for `data/matches-fresh-v4/`, `-v4b/`, `-v4c/` and all 16 EXP-0001 matches; ledger updated. Excluded: orphan phantom `ba1703f05ad9` (no summary, created on the chest before the fix) and the 5.2 s phantom `227876ae59b2` (HOME animation hash).

## 2026-11-25 → 2026-11-26 — Build log (oldest first)

**Where things stand (Nov 2026):** core, perception, policy and flywheel code exist, with 89 passing tests against a fake scrcpy server and a synthetic scene. **Update 2026-11-26:** the harness has now streamed from a real OnePlus N10 (phone-a), phone-a ROIs/screens/card templates/arena geometry are calibrated, and a bootstrap controller driving the real UI through the harness scrcpy client won a Training Camp match **3–1**. M3 detector foundations are committed (`models/classes.txt`, `models/katacr_alias.json`, `learn/synth.py`, `learn/train_detector.py`, `tools/fetch_katacr.py`, `tools/extract_backgrounds.py`), and a read-only detector review found silent-failure bugs to fix *before* the first training run — folded into M3 below, ordered by severity. No detector is trained yet; this host's venv is CPU-only (no torch/ultralytics, ORT CPU-only), so training and the MIGraphX benchmark need a ROCm host. Several real-game pieces are missing (listed per milestone, with file pointers).

**Update 2026-11-26 (PR `m3/phone-free-fixes`):** everything below that didn't need the phone is done: a no-vision strategy that attacks (`training-camp-blind-v1`), CI green on a clean checkout, every detector-review item, side-aware classes, spell-side attribution, tower state from fixed ROIs, async/strided detection, the training-data tools (synth in the inference geometry, KataCR sprites + real-frame converter, trainer), replay regression, execution retry, a real counter table, a stream soak mode and an app watchdog. 120 tests (118 pass, 2 skip without local calibration).

**Update 2026-11-26 (PR `v3/giant-beatdown-and-eval`):** fixes from two outside reviews, checked against the code first: stale hand slots are never played or re-tapped; only a confirmed spell uses up the blind push's spell; a counter-table row can't make a card defend against what it can't hit; path prediction follows destroyed towers. Experiments compare against the champion now (randomized matched pairs; the old test compared the candidate with a fixed 50% and could promote a worse strategy). Detector data: KataCR split by sequence (the converter used to write train == val), `dataset` refuses train/val overlap, `eval-detector` reports per-class recall and gates on threat classes, `frames` exports your matches for labeling split by match, and `other_ground`/`other_air` catch-all classes stop unknown enemy cards from being background. Strategy: `training-camp-v3` (Giant is the only win condition) plus one child per experiment (E3, E4, E5). 138 tests (136 pass, 2 skip without local calibration).

**Update 2026-11-26 (ladder + deck identity):** `--mode ladder` works on the real phone: the ladder flow queues from the lobby's yellow Battle button and never taps the asset-loading screen in between. Getting there needed an arena-independent battle check (`ScreenClassifier.battle_evidence`: the elixir bar exists in every battle, whatever the arena) — the dhash references only describe the Training Camp arena and one opponent nameplate, so a ladder match classified UNKNOWN and the bot sat in a live game with elixir at 10 until it lost. First ladder match under the fix: 8 cards played, recorded 1–1 draw.

The deck screen also settled a naming error: the starter card is **Archers** (Common frame, no crown, avg 3.6; a 0-trophy account cannot own a Legendary), so `cards.json` had been carrying Princess's stats (1 unit, 216 HP, 9 tiles) for a card that spawns two 5-range archers. Card key, unit class, KataCR alias, templates, every strategy deck, the counter tables and the test fixtures now say `archers`; the stats come from cr-api-data. `training-camp-v3` and children inherit the rename, so recorded matches before this date refer to the old key.

Still open for ladder: a trained detector (until then only `training-camp-blind-v1` attacks), 8 template states captured in a ladder arena, and the reviewer's `hand_unreadable` storm (~10k warnings per match on both arenas) — the matcher is failing most slots and retries are carrying the run.

**Update 2026-11-26 (ladder-vision build `00a17c0`, optiplex side):** the vision path for the real deck is built end to end. `ladder-vision-v1.yaml` (+ `counters/ladder-vision-v1.csv`) targets the account's true 8 cards (Knight/Archers/Minions/Arrows/Fireball/Giant/Musketeer/Mini P.E.K.K.A — the deck screen proves it; see deck-identity note below), with tempo tuned from live gaps (threshold 0.08, attack at 5 elixir, leak at 8.0, opening 15 s). `--mode ladder` exists (`ladder` flow HOME→battle, battle button in `rois/phone-a.yaml`, both `run` subcommands). M8 core done: `learn/label_from_video.py` (weighted state/action rows from recorded matches), `learn/train_ranker.py` (numpy-MLP BC, static ONNX export), live ranker loader in `app.py` (was a ValueError). M9 core done: `core/runtime/advisor.py` (API-driven posture intents, TTL=2×cadence, stdlib HTTP) + `advise --dry-run` replays a recorded match offline. M5 gaps closed: `flywheel/shadow.py` (champion-vs-candidate report), `notebooks/dashboard.py` (win rate CI, play value, triage, p99), match-clock reader in `pipeline.py` (1 Hz, drift warning). Tests: **170 passed**. Phone-a calibration additions (gitignored, this machine only): `archers__0.png` rebuilt from pixel-verified deck art (the template-CLI crop was mislabeled), `ok` ROI fixed to [0.370, 0.815, 0.630, 0.880] (was ~140 px low; post-match lag 5 min → 2.2 min/match), `battle` button ROI added. Records: Training Camp 16–1 (blind), ladder 0–0–1 (a 66 s 1–1 draw — likely disconnect, under investigation). The 50-win grind is PAUSED, not done.

**Update 2026-11-26 (deck identity, settled):** the battle deck IS the Battle Deck screen's deck. Template reads of `princess` (30× in one ladder match) are matcher misfires; `archers` reads zero because the template was mislabeled, now rebuilt from verified art. The starter card is **Archers**, not Princess (upstream `0075023` renamed it everywhere: cards, templates, strategies, counters, fixtures). No second deck slot exists in this game version.

**GPU handoff (veritas, nick@192.168.1.199):** venv `~/venvs/droid-det` ready (torch 2.7.1+rocm6.3, `cuda_available: True`, ultralytics, onnx; repo at `~/src/droid-harness`). KataCR fetch running: `tools/fetch_katacr.py --out data/katacr --part2` (log `/home/nick/droid-det-fetch.log`, ~1.3 GB and growing). GPU idle (0%), 400 GB free. Remaining there, in order — see **GPU Runbook** below.

**Update 2026-11-26 (veritas M3 + ladder-vision-v3 verified 9W–1L):**
- **Calibration committed (`phone-a`):** `configs/device_profiles/phone-a.yaml` (`741be240`, `exec_latency_ms: 40`), `rois/phone-a.yaml` (added `dismiss` and `trophy_ok`), rebuilt 18-template deck set in `templates/phone-a/` (removed stale zoomed `archers__0` and redundant duplicates), 10 `alive`/`rubble` tower templates in `towers/phone-a/`, plus `MATCHMAKING__0.png` and `TROPHY_ROAD__0/1.png` (Bone Pit Arena-lock modal) in `screens/phone-a/`. `tiles/phone-a.yaml` and `arena/phone-a.yaml` verified pixel-aligned.
- **M3 detector trained & benchmarked:** YOLO26-n trained on `veritas` (RX 7900 XT, PyTorch-ROCm, stopped at epoch 40; `best.pt` at epoch 36 reached `mAP50=0.797` / `mAP50-95=0.514`, epoch 40 `mAP50=0.793` / `mAP50-95=0.507`; local gitignored weights `models/yolo26n-cr.pt` and static `640×384` `models/yolo26n-cr.onnx`). 100× idle-GPU arena-crop benchmark (`440×707`): `OnnxDetector` CPU `26.78 ms p50 / 53.40 ms p95` vs new `TorchDetector` PyTorch-ROCm `7.90 ms p50 / 9.83 ms p95` (bit-identical detections).
- **Perception, session, menu & strategy fixes:** `CardMatcher` `local_znorm` + bottom-center/border `default_mask`; `TorchDetector` backend in `detector.py`/`app.py`/`cli.py`; early-spend detection, match-start (`frame_i == 0`, `t <= 3.0s`) elixir sync, and low-conf generic tower-skin FP suppression in `pipeline.py`; `read_result_detail()` left+right banner requirement + `battle_min_frac=0.12` in `screens.py`; `battle_evidence` battle-lock & timeout invariants (`unknown_battle_timeout_s=25.0`, `left_battle_timeout_s=5.0`) in `session.py`; `MENU` (`dismiss`) & `TROPHY_ROAD` (`trophy_ok`) navigation + `unknown_settle_s=3.0` in `menus.py`; generic `other_ground`/`other_air` handling & non-spell `leak_guard`; new `ladder-vision-v2` and `ladder-vision-v3` strategies + updated counter tables.
- **Live `--mode ladder` results (`phone-a`, `theblackcat99`):** `ladder-smoke2` **1W–0L** (`3-1`), `ladder-iter-1` **5W–0L** (`v2` + CPU ONNX, 3 orphans before session fixes), `ladder-verify-pre-fix` **4W–2L–1D**, `ladder-cleanup` **1W–0L**, and final clean verification batch `ladder-verify` (`ladder-vision-v3` + epoch-40 `TorchDetector` PyTorch-ROCm + all fixes): **9W–1L–0D (90.0%, 95% CI [0.596, 0.982])** across 10 completed matches (`1-0`, `0-3`, `3-2`, `3-1`, `1-0`, `3-0`, `3-1`, `2-1`, `3-0`, `3-0`), `0` aborts, `0` orphaned dirs, `loop_p50_ms` `2.48–2.77 ms`, `loop_p99_ms` `4.01–4.36 ms`, `elixir_discrepancy` `0–1`/match (paused during Match 11; completed match records saved in `data/matches/`).
- **Tests & lint:** `pytest` **179 passed, 1 skipped**; `ruff check src/ tests/` clean.

**Update 2026-11-26 (31-class detector v2, opponent card/cost/cycle tracking, Arena 2 crown fix, v4 9W–1L vs v5 5W–5L):**
- **31-class YOLO26-n v2 detector:** Expanded `models/classes.txt`, `models/katacr_alias.json`, and `models/katacr_generic.json` from 18 to 31 classes (preserving indices `0..17`, adding 13 Arena 1–3 enemy classes including `goblin_opp`, `spear_goblin_opp`, `goblin_cage_opp`, `goblin_brawler_opp`, `goblin_hut_opp`, `skeleton_opp`, `bomber_opp`, `tombstone_opp`, `valkyrie_opp`, `baby_dragon_opp`, `barbarian_opp`, `cannon_opp`, `prince_opp`), added `bel==0` catch-all fallback in `learn/katacr.py`, and `--synthesize-missing-sides` in `learn/synth.py` for missing starter `*_me` classes (`archer_me`, `mini_pekka_me`, `minion_me`, `giant_me`). Trained `yolo26n-cr-v2` on `veritas` (best epoch 7: `P=0.742, R=0.632, mAP50=0.728, mAP50-95=0.474`; PyTorch-ROCm `TorchDetector` 100× arena-crop GPU benchmark `p50=4.14 ms / mean=4.23 ms`, exported local gitignored `models/yolo26n-cr.{pt,onnx}`).
- **Opponent card/cost/cycle/elixir tracking (`cards.py`, `ownership.py`, `tracker.py`, `opponent_model.py`, `state.py`, `pipeline.py`):** Added bar-color team override, legal opponent spawn zone checks, intro zero-HUD suppression + `hud=7.0` sync, stationary building + physical elixir plausibility gates, 8-card deck lock, cycle deduplication, same-side spatial re-association, and swarm upgrades (`minions`≥4 -> `minion_horde`, `skeletons`≥6 -> `skeleton_army`). Cut offline replay `opp_spawns` noise from `160–348` to `16–44`; 10/10 live matches in both Batch 1 and Batch 2 lock 8-card opponent decks cleanly.
- **Arena 2 golden-sand crown fix (`perception/screens.py`):** Restricted `_large_gold_components` to crown cushion span `_CROWN_X = (0.16, 0.84)` with tightened `HSV >= [12, 160, 160]` and `min_area = 0.0025 * h * w`. Verified 10/10 on Batch 1 tail recordings (`9W–1L–0D`, eliminating 3 false draws and 1 `0-3` misread loss) and 10/10 on Batch 2 (0 false draws, 6/6 audited end screenshots exact).
- **Strategy iterations & live 10-match Arena 2 ladder batches (`phone-a`, `theblackcat99`):**
  - `ladder-vision-v4` (current best champion, Batch 1 `data/ladder-v4-b1` kept local due to pre-fix `summary.json` crowns): **9W–1L–0D (90.0%)** human/script-verified across 10 completed matches (`loop_p50_ms` `2.60–4.76 ms`), using card/cost/rotation tracking (`max_opp_elixir_to_bridge`, `counter_in_hand_mult`/`counter_unaffordable_punish_mult`/`counter_out_mult`, `defense_reserve`, `spell_aware`, `kill_aware_spells`, `save_counters_for_in_hand_threats`).
  - `ladder-vision-v5` (rejected candidate, immutable history; Batch 2 `meta.json`/`summary.json`/`decisions.jsonl` committed in `data/matches/`): **5W–5L–0D (50.0%)** across 10 completed matches (`loop_p50_ms` `2.51–3.15 ms`, `loop_p99_ms` `3.95–8.66 ms`). Regression root cause: lowering `giant_safe_build_elixir` to `7.2`, raising `giant_back_build_score` to `0.72`, and softening `reserve_penalty` to `0.60` caused immediate `t=1.1–1.4s` blind back-Giant openings into `7.5–10.0` opponent elixir, causing three ~95s `0–3` losses (`98b80f1060b2`, `d1485a87825e`, `d88cb32961f4`).
  - **Next recommended step (paused):** Branch `ladder-vision-v6` from `ladder-vision-v4` (never back-build Giant during `OPENING` or while opponent elixir is near full) and run paired SPRT (`run-experiment --test pairs`) vs `ladder-vision-v4`.
- **Tests & lint:** `pytest` **189 passed, 1 skipped**; `ruff check src/ tests/` clean.

**Update 2026-11-26 (Tower HP digits, Triage blackout fix, 3-Speed Flywheel, Pro rules v6, M8 v2 Residual Ranker & One-Command CLI):**
- **Tower HP & Countdown Clock Fixed (`grid.py`, `digits.py`, `rois/phone-a.yaml`, `digits/phone-a/0..9.png`, `pipeline.py`, `engagements.py`, `session.py`):** Separated `tower_hp` ROIs from `towers` rubble ROIs in `rois/phone-a.yaml`, upgraded `digits.py` with 3× bicubic HSV white+outline binarization + merged-pair splitting, calibrated `digits/phone-a/0.png`..`9.png` against real `phone-a` ladder video (`1750`, `1400`, `1335`, `1358`, `1196`, `1170`, `1128`, plus regulation/overtime/red countdown clock), seeded unactivated King Tower max HP (`infer_king_max_hp`), recorded `tower_hp_diff` in `summary.json`, and restored real tower-damage deltas in `Engagement.value`.
- **89% Triage Blackout Fixed (`flywheel/triage.py`):** Ignored transient `hand_unreadable`/`affordability_mismatch` slot-transition warnings around confirmed plays while keeping `elixir_discrepancy`/`tower_hp_increase` strict; on the 39 recorded matches in `data/matches/`, `PERCEPTION` dropped from `88.9%` to `18.0%` (mostly legacy pre-calibration matches) and scored rule decisions jumped from `1` to `550+`.
- **Intelligent Weight Caching & Persistent-Session `experiment` (`detector.py`, `train_ranker.py`, `app.py`, `cli.py`, `runner.py`):** Cached `TorchDetector`/`OnnxDetector` and `OnnxRanker` in-memory by file signature (`warmup_once` runs once), refactored `BotRunner.set_strategy` + `cmd_experiment` so one persistent `scrcpy` + GPU session runs all paired matches without reloading weights or reconnecting sockets, and added Stage-A dense-margin futility stopping (`--futility-pairs 12`).
- **M8 v2 Board-Aware Residual Ranker (`train_ranker.py`, `label_from_video.py`, `utility.py`):** Added `droid-harness-ranker/2` (+10 per-lane board value, threat ETA, and opponent-cycle state features), bounded residual rescoring (`rule_score + w * (prob - 0.5)` so safety/counter gates are never overwritten), and engagement-credit Advantage-Weighted / Filtered BC in `label_from_video`.
- **Pro Engine Mechanics & `ladder-vision-v6` (`offense.py`, `defend.py`, `placement.py`, `ladder-vision-v6.yaml`):** Landed `OPENING`/`x1` guard on `GiantBackBuild`, `opp_min_cycle_elixir_to_return` cycle-lock punish boost, healthy-tower chip filter (`ignore_chip_tower_hp_frac: 0.60`), `OppositeLanePunish` rule, anti-spell ranged defender offset (`3.2` tiles), cross-lane melee kite, and exact Lv1/Lv3 `spell_tower_damage` (`lethal_margin_hp: 0`, `fireball: 75`, `arrows: 30`).
- **One-Command CLI & 3-Speed Flywheel (`cli.py`, `offline_study.py`, `digest.py`, `retention.py`):** Added `--device phone-a` auto-bundle resolution, short strategy aliases (`v6`), `droid-harness doctor`, `play`, `experiment`, `tune-offline` (Speed-1 CPU counterfactual parameter search), `shadow`, `retain`, and `digest` (`< 30 KB` Speed-3 Agent Handoff Artifact).

**Update 2026-11-26 (stuck-screen HOME recovery, verified live):**
- **Root cause found on `phone-a`:** a ladder loss with a reward chest adds a **Play Again** button and shifts **OK** right. The calibrated `ok` ROI (`[0.370, 0.815, 0.630, 0.880]`, centre `(232,868)` at 464×1024) landed in the 23 px gap between the buttons, so every result-screen tap missed. The screen also hashes `UNKNOWN` (new nameplate + reward layout), so the UNKNOWN flow cycled other wrong taps. The phone sat on the loss screen indefinitely.
- **Deterministic recovery (`recovery.py`, `session.py`, `app.py`, `cli.py`):** after 8 s on an unchanged result screen (15 s for any other non-battle menu), the session arms an escalation ladder: calibrated buttons → wide saturated buttons found **visually** in the lower frame (rightmost first, which finds the real OK) → BACK through scrcpy → force-stop/relaunch after the plan is exhausted or 60 s of liveness findings. Once an episode starts it keeps clearing each new screen until HOME; HOME disarms it. `home_visible()` also requires the Battle button so a Trophy Road scroll that hashes as HOME is not accepted.
- **Standalone command:** `droid-harness recover [--dry-run] [--stuck-after 3]` walks a phone that was left on any screen back to HOME without the harness running. `doctor --live`'s ADB check was fixed in the same pass (`.run()` returns a string, not a process).
- **Post-match chest stall (found during the live v4 batch):** the purple chest/reward animation is 100% magenta in the elixir ROI, so `battle_evidence` read it as a full elixir bar and the session stayed `IN_BATTLE` for ~6 minutes after the match. Added a **30 s battle no-progress guard**: if the clock (`match_t_s`), any fresh hand slot and elixir all stop changing, the match is recorded as `aborted:battle_no_progress` and the same recovery ladder runs. A static-screen variant still relies on the liveness reconnect, but the animated chest is now handled.
- **Batch-end settle:** a finished run (or a graceful `stop_when`) walks back to HOME before disconnecting, using only the recovery ladder. Menu decisions are suspended and `_set_screen` cannot start/end a match, so a transient HOME animation that hashes `IN_BATTLE` cannot create a phantom match; the earlier settle did exactly that once (a 5.2 s aborted match) before this guard. `walk_to_home` also flags a frozen clock+elixir pair as `UNKNOWN` so the standalone `recover` command can clear the purple chest even when the classifier calls it a battle.
- **Verified live:** `recover` took the stuck `RESULT_LOSS` (0–3, −15 trophies) to HOME in ~10 s, then correctly rejected a Trophy Road reward screen and stopped on the real HOME. Tests: **208 passed, 1 skipped**; `ruff check src/ tests/` clean.

## 2026-11-26 — Refinement-loop measurements (why the 3-speed loop was built)

Measured on ladder batches 1 and 2 (10 matches each):

1. **Noise in 10-game batches.** Wilson 95 % intervals span 40–53 points (v4 9W–1L = [0.596, 0.982]; v5 5W–5L = [0.237, 0.763]); a real +6–8 point gain needs 50–100+ games or paired blocks.
2. **Triage blackout.** Every card play animates its slot for ~300 ms (15–18 frames) of transient `hand_unreadable`, so any engagement touching a play crossed the 2-frame threshold: triage mix `{PERCEPTION: 0.904, OK: 0.089, STRATEGY: 0.007}` and `{0.889, 0.077, 0.034}`, leaving one scored decision in ten games. Fixed (Fix A): ignore slot-transition warnings around confirmed plays, require sustained corruption; PERCEPTION fell to ~18 % on 39 matches and scored rule decisions rose from 1 to 550+.
3. **Disk.** ~180 MB per 3-minute match of video, 9–18 GB per 50–100 games; compact files ~1.5 MB per match. Fixed (Fix B): `retain` and `--prune-win-video`.
4. **Ladder drift between sequential batches.** v4's 9–1 climbed trophies and pushed v5 into Lv4–5 opponents: sequential unpaired batches mix the strategy effect with the climb. Answer: randomized pairs (`--test pairs`).
5. **No failure aggregator.** v5's regression (three ~95 s 0–3 losses from 1.1–1.4 s back Giants at 7.2–7.5 elixir: `98b80f1060b2`, `d1485a87825e`, `d88cb32961f4`) took a custom script to find. Answer: `droid-harness digest`.

## 2026-11-26 — Overnight plan snapshot (post-`fc94e6b`)

Goal at the time: leave the phone running overnight, record and analyze every match, improve through deterministic experiments with no LLM in the control loop. Phase 0 (trustworthy runner: result settling, result-screen recovery, popups, tower-HP digits, watchdog), Phase 1 (detector) and most of Phase 2 (batch runner, analysis, experiment runner writing to the ledger and DuckDB) are done; Phase 3 (learned decision model) is M8. Known issues then: weak `archer_opp` (R 0.27) / `knight_opp` (R 0.43) and four `*_opp` classes with no KataCR validation examples; transition-window hand warnings (since fixed in triage); loopback, display-off and thermals unverified overnight; no live SPRT yet (since run: EXP-0001). Definition of done for the overnight loop: 50+ unattended matches without intervention, every match with summary, video, decisions and a correct result, low `perception_warnings` in DuckDB, at least one SPRT decided and in the ledger, win rate with a confidence interval.

## 2026-11-26 — `phone-a` bring-up (first real-device run)

OnePlus N10 (BE2028), Android 11, 1080×2400, 90 Hz, serial `741be240`; Clash Royale (`com.supercell.clashroyale`), account `theblackcat99` (0 trophies, no clan). Stream `BE2028 464x1024 h264`, 867 frames in 15 s (57.8 fps while moving), decode p50 3.8 ms / p99 15 ms on `scrcpy-server-v2.4`. Screen classification only works on the **top 18 %** of the frame (the bottom half changes with the hand). Card matching scored 1.00 on every occupied slot of the saved frames. First real win: a bootstrap controller (`tools/bootstrap_win_bot.py`: harness frames, elixir from the bar, every affordable card at the bridges via `adb shell input tap`) beat Trainer Jonas **3–1** in ~90 s with 12 plays, validating capture, HUD, hand rotation and the touch path before the detector existed. Rough edges found then (all since fixed): the result screen ignoring injected taps (BACK opened "Return to the Main Menu?"), a top-only dhash that couldn't tell battle from result (ended a match at 56.9 s and mislabeled a loss as a win; fixed by banner/crown verification and 3 stable observations), untuned placement tiles, and no detector/digits/clock.
