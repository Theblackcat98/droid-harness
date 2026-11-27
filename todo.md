# todo.md — from this repo to a bot that plays real matches

**Target:** the bot plays Clash Royale **Training Camp** matches on a real phone, unattended. Every match is logged, analyzed and triaged, and a strategy change has been accepted or rejected by a real SPRT. Stretch goals follow after that (Practice gauntlet, two-phone self-play, coach, learned policy, general Android).

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

### Locked-in Roadmap & Priority Order (updated 2026-11-26)

1. **[you/live]** **Account & Ladder Progression toward Hog EQ (`theblackcat99`):**
   - Upgrade **1 cheap card right now** (`theblackcat99` is `8/9` XP to King Level 2 — +112 Princess HP, +168 King HP, +8% tower DPS, unlocks Arena 2+).
   - **Gold & Wildcard Rule (Bridge to 2.6 Hog EQ):** Funnel gold/wildcards ONLY into permanent Hog EQ / cycle staples (`Musketeer`, `Knight`, `Fireball` until `Earthquake` in Arena 9, `Skeletons` [Arena 2], `Cannon` [Arena 3], `Fire Spirit` / `Ice Spirit` [Arena 4/8], `Hog Rider` [Arena 5], `The Log` / `Barbarian Barrel`). Keep `Giant`, `Mini P.E.K.K.A`, `Archers`, `Minions`, `Arrows` only at minimal survival floors until replaced. See [`docs/pro-strategy-research.md`](docs/pro-strategy-research.md).
2. **[you/live]** **Speed-2 Live Paired Experiment (`v4` vs `v6` on Ladder):**
   - Run pre-flight: `.venv/bin/droid-harness doctor --live`
   - Run persistent paired evaluation: `.venv/bin/droid-harness experiment --champion v4 --candidate v6 --mode ladder --prune-win-video --title "v6 pro mechanics + digits"`
3. **[data/you]** **Real-Frame Detector Fine-Tuning (`4.3`):**
   - Export crops from existing `data/ladder-*` matches (`droid-harness frames`), clean up weak classes (`archer_opp`, `knight_opp`) + any newly unlocked Arena 2–5 cards (`skeleton`, `cannon`, `hog_rider`), fine-tune 15–20 epochs on `veritas`, and gate with `eval-detector`.
4. **[code/agent]** **3-Speed Autonomous Refinement Loop (`3.1–3.4`):**
   - **Speed 1 (CPU seconds, 0 LLM):** Run `droid-harness tune-offline --champion v6 --data data/matches --trials 100 --out configs/strategies/ladder-vision-v7-tuned.yaml` + `shadow` + `replay-diff` to pick the single best numerical candidate offline.
   - **Speed 2 (Phone hours, 0 LLM):** Validate the single offline winner via `droid-harness experiment` (Stage-A dense-margin futility at 12 pairs → Stage-B paired SPRT promotion into `ledger/STRATEGY_LEDGER.md`).
   - **Speed 3 (Agent Harness only on structural plateau):** Generate `droid-harness digest --data data/matches --out data/agent_digest.md` (`< 30 KB` dossier) for the coding agent to add 1 structural rule + unit test.
   - **M8 v2 Residual Ranker:** Once 50–100+ post-digits ladder matches are banked, run `python -m droid_harness.learn.label_from_video --version 2 --cards configs/domains/clash_royale/cards.json --min-credit -0.5` → `train_ranker` → SPRT-gate `learned_ranker` in residual mode (`ranker_mode: residual`).
5. **[deferred — GUI]** **Live Annotated GUI & Post-Match Replay Inspector (`droid-harness watch` / `--ui` & `inspect`):**
   - *Deferred per operator instruction — see full specification and mandatory alignment question under **Deferred: Live GUI & Replay Studio** below.*

Tags: **[you]** needs your hands, phone or account · **[code]** pure code work · **[data]** collecting or labeling.
Each milestone ends with **Done when** — don't start the next one until it's true.

> Account rule: the repo says throwaway-only, never ladder. **Operator override 2026-11-26:** ladder IS approved on `theblackcat99` (0 trophies, bot-tier opponents, nothing to lose). Revisit if the account gains real trophies or Supercell flags anything.

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
      *Verified on `phone-a`: streamed `464×1024 h264` at ~57.8 fps moving (decode p50 ≈ 3.8 ms) and across 30+ live matches (see `docs/phone-a-bringup.md`).*
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

## GPU Runbook (veritas — initial epoch-40 run done; steps for fine-tuning)

Venv `~/venvs/droid-det` (`/home/nick/venvs/droid-det`, symlinked as `.venv` in this repo) is ready: Python 3.13, `torch 2.7.1+rocm6.3` (`cuda_available: True` on RX 7900 XT 20 GB), `torchvision`, `ultralytics`, `onnx`.

Initial M3 training run (2026-11-26) completed Steps 1–7 (`data/katacr` fetched, 28 arena backgrounds extracted, `data/train.yaml` built with 10,483 train + 1,332 val across 18 classes, `yolo26n.pt` trained to epoch 40 with `best.pt` epoch 36 `mAP50=0.797` / `mAP50-95=0.514`, exported to local gitignored `models/yolo26n-cr.pt` and static `640×384` `models/yolo26n-cr.onnx`).

To fine-tune on newly labeled real frames from `data/ladder-verify` / `data/matches`:

1. **Export own frames & pre-label:** `.venv/bin/droid-harness frames --data data/ladder-verify --layout configs/domains/clash_royale/rois/phone-a.yaml --detector models/yolo26n-cr.pt --out data/own` (then fix weak/missing labels).
2. **Merge datasets (refuses overlap):** `.venv/bin/droid-harness dataset --train data/synth/data.yaml:train data/katacr-real/data.yaml:train data/own/data.yaml:train --val data/katacr-real/data.yaml:val data/own/data.yaml:val --out data/train.yaml`.
3. **Train / fine-tune:** `.venv/bin/python -m droid_harness.learn.train_detector --data data/train.yaml --weights models/yolo26n-cr.pt --epochs 20 --batch 32 --imgsz 640,384` *(note: invoke `learn.train_detector` directly because `cli.py train-detector` outer parser intercepts `--data` before `REMAINDER`; use shorter `--epochs`/faster settings when iterating on added cards)*.
4. **Gate with `eval-detector`:** `.venv/bin/droid-harness eval-detector --model models/yolo26n-cr.onnx --data data/own/data.yaml --min-threat-recall <target>`.
5. **Run live on `veritas` with `TorchDetector` (PyTorch-ROCm):** pass `--detector models/yolo26n-cr.pt --classes models/classes.txt --async-detect` (`7.90 ms p50 / 9.83 ms p95` vs CPU `OnnxDetector` `26.78 ms p50 / 53.40 ms p95`).

## Definition of done: "fully running harness in a real bot game"

- [ ] The bot plays Training Camp matches unattended for hours with automatic recovery (reconnect, popups, app restart).
- [ ] Loop p99 is under 100 ms in battle; HUD accuracy ≥ 99.5%; the detector meets its recall target.
- [ ] Every match is recorded, analyzed, triaged and in the warehouse, with a game version and balance epoch.
- [ ] At least one strategy change has been accepted or rejected by SPRT and recorded in the ledger.

*Current standing (after `fc94e6b`): `loop_p99_ms` (`4.01–4.36 ms`) and `TorchDetector` latency (`9.83 ms p95`) satisfy the latency targets, and 10 unattended `--mode ladder` matches finished with 0 aborts/orphans (`9W–1L–0D`); remaining gates are multi-hour unattended stability + retention, 99.5% HUD golden-set test + weak-class detector recall, full warehouse ingestion, and the first live paired SPRT.*

---

## Deferred: Live GUI & Replay Studio (`--ui` / `watch` / `inspect`)

> **MANDATORY ALIGNMENT QUESTION — ASK THE OPERATOR BEFORE IMPLEMENTING THE GUI:**
> *"Before building the GUI: do you want **(A) the zero-build Local Browser Web Studio (`--ui` on `localhost:8765`)** — live annotated phone `<canvas>` on the left + live strategy/cycle scoreboard & offline `.h264` match scrubber on the right (works seamlessly with `opencv-python-headless` and over SSH to `veritas`), or **(B) a Native Desktop Window** (e.g., PySide6 / DearPyGui / non-headless OpenCV)?"*

### Preserved GUI Architecture & Feature Spec (ready to build once aligned)

1. **Zero-Latency Telemetry Tap (already pre-wired in `perception/pipeline.py`):**
   - `CRPerception.last_detections` already records raw pixel boxes `(bx1, by1, bx2, by2)`, `cls`, `side`, `conf`, `tile`, and `hp` per frame at zero copy cost.
   - Publish `(frame, state, last_detections, decision, experiment_state)` into a lock-free latest-snapshot slot sampled at `15–20 Hz` by a background UI publisher thread so `BrainLoop` stays at `~3–5 ms` (`0.0 ms` added latency).
2. **Left Pane — Live Phone Video + Toggleable Overlays:**
   - Live `scrcpy` frame with crisp vector overlays: `_me` (blue) / `_opp` (red) / spell bounding boxes, class + confidence + HP%, velocity vectors, `18×32` tile grid + river/bridges, `tower_hp` & `hand` ROIs, predicted intercept target vector, and animated touch/tap ripples when `Executor` dispatches touches.
3. **Right Pane — Live Strategy Numbers & Opponent Model:**
   - Dual Elixir bars (`My Elixir` vs `Opp Elixir [low, high]` + leaked elixir counter) and Tower HP readout (`me_*` vs `opp_*` + `tower_hp_diff`).
   - **Opponent 8-Card Cycle Widget:** Live 4-card hand vs 4-card queue (`cards_until_back` + `min_cycle_elixir_to_return` + affordability badges).
   - **Hand & Template Health:** Slots `0..3` + `next_card`, `fresh` vs `stale` flags, greyed check.
   - **Live Candidate Scoreboard:** Ranked real-time table of *all* proposed candidates (`rule_id`, `card @ tile`, raw `score`, `weight × posture_mult = total`, `counter_factor`, `reserve_factor`, `ranker` residual delta, `act_threshold` cutoff line, and green highlight on the chosen action).
   - Experiment & Latency strip: W–L–D record, active arm (`champion` vs `candidate`), paired dense margin, live SPRT LLR bar, and `decode / perceive / decide / p99 ms`.
4. **Offline Match Replay Scrubber (`droid-harness inspect <match_dir>`):**
   - Reuse the exact same two-pane UI to scrub frame-by-frame through any saved `stream.h264` + `decisions.jsonl` to visually inspect losses and counterfactual `shadow` choices.

---

## After that (stretch)

- **M7 — Practice gauntlet + self-play.** Add a Practice Mode flow (battle log → opponent → Training) and a friendly-battle invite/accept flow (`menus.py`; the `friendly` flow is a stub). Run two `BotRunner`s with distinct ports, swapping strategy ↔ (phone, account) per `flywheel/eval/selfplay.py`. Check `seat_effect` for device bias.
- **M8 — Learned policy (plan P6).** CORE DONE 2026-11-26: `learn/label_from_video.py` (card-left-hand + matching spawn → weighted state/action rows from recorded matches), `learn/train_ranker.py` (numpy-MLP behavior cloning, static ONNX export `(32,55)->(32,1)`), live loader in `app.py` (`CRPolicy(ranker=...)` rescores candidates). STILL OPEN: first real training on warehouse decisions (needs 50+ wins of data — the farm), SPRT-gate each step (BC → filtered BC → offline RL), feature-schema in the strategy hash, logged ranker scores with fallback.
- **M9 — L2 advisor (optional).** CORE DONE 2026-11-26: `core/runtime/advisor.py` (separate process, OpenAI-compatible API via `--api-base/--model`, posture-only JSON schema, TTL=2×cadence, never blocks), `advise --dry-run` (offline replay of a recorded match). STILL OPEN: point it at a live key (deepseek/zai), measure loop p99 with it on, keep it only if it wins an SPRT, then distill it.
- **M10 — General Android (plan P7).** Test DroidRun Portal on your phones (`domains/android_os/a11y.py`, with uiautomator fallback). Add a GUI-model fallback with the Set-of-Mark overlay. Measure step latency and cache-hit rate on an everyday task suite.

## Risks to watch

- **Game updates** change UI, cards and balance. Re-run the golden set after each update, bump `balance_epoch`, and void running tests.
- **scrcpy upgrades** can change the protocol. Re-pin the version, re-check the structs, and re-run M1.
- **Thermals:** watch `loop_p99_ms` and loopback drift over long runs.
- **Perception errors** look like strategy errors. Trust triage, and fix perception before tuning strategy.
