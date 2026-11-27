# droid-harness

A deterministic, low-latency Android automation harness, with Clash Royale as the reference game. It runs **without any LLM in the loop**: scrcpy H.264 stream → computer-vision perception → rule-based utility policy → timed scrcpy touches.

Every strategy parameter, threshold, counter table and rule lives in versioned YAML/CSV files under [`configs/strategies/`](configs/strategies/). The analytics flywheel (`analyze`, `replay-diff`, `experiment`, `sprt`, `ledger`) measures, tunes and promotes strategies statistically with local Python tools only. The Studio (`--ui`, `watch`, `inspect`) shows what the bot sees and why it chose each play, live or after the fact.

> **Account scope.** Supercell's terms treat bots as cheating and can ban accounts. Use a throwaway account. The plan was Training Camp, Practice and two-phone friendlies only; **operator override 2026-11-26:** ladder is approved on `theblackcat99` (throwaway, nothing to lose). The general Android core carries none of this risk.

**Two documents.** This README is the documentation: how to run, how it works, the design and the strategy playbook. [`todo.md`](todo.md) is the plan: current status, priorities, open tasks per milestone and the run log. Nothing else needs to be kept in sync. ([Where the old `docs/` files went](#where-the-old-docs-went).)

## Contents

1. [Status](#1-status)
2. [Install and smoke-test](#2-install-and-smoke-test-no-phone-needed)
3. [Play matches](#3-play-matches-on-a-calibrated-phone)
4. [Studio: live view and replay inspector](#4-studio-live-view-and-replay-inspector)
5. [Improvement loop](#5-improvement-loop)
6. [Strategies: files, rules, promotion](#6-strategies-files-rules-promotion)
7. [Strategy playbook](#7-strategy-playbook)
8. [Analyze matches](#8-analyze-matches)
9. [Set up a new phone](#9-set-up-a-new-phone)
10. [Detector: retrain or add cards](#10-detector-retrain-or-add-cards)
11. [Architecture and design](#11-architecture-and-design)
12. [Evaluation and statistics](#12-evaluation-and-statistics)
13. [LLM roles, learned policy, general Android](#13-llm-roles-learned-policy-general-android)
14. [Design history](#14-design-history)
15. [Troubleshooting and operational limits](#15-troubleshooting-and-operational-limits)
16. [Sources](#16-sources)

---

## 1. Status

Measured on `phone-a` (OnePlus N10, `741be240`), account `theblackcat99`. The latest numbers and the open work are in [`todo.md`](todo.md).

| Component | Current measured state |
| --- | --- |
| Transport and loop | `scrcpy-server-v2.4` over USB + PyAV H.264 decode; battle loop `2.5–4.8 ms` p50 / `4–12 ms` p99 with `--async-detect` |
| Detector | 31-class YOLO26-n v2 ([`models/classes.txt`](models/classes.txt)): mAP50 0.728, mAP50-95 0.474; `TorchDetector` (PyTorch-ROCm) 4.1 ms p50 on the RX 7900 XT |
| Opponent tracking | Card, cost and cycle tracker locks 8-card opponent decks (10/10 matches in recent batches) |
| HUD | Tower-HP digits with a per-tower label search, locked offsets and misread correction; countdown clock; both sides' King Level inferred from Princess Tower HP. `tower_hp_diff` is only written when all six towers were read |
| Results | `ladder-vision-v4` 9W–1L near 0 trophies; at ~476 trophies EXP-0001 (v4 vs v6, 16 matches) went 2–6 vs 2–5, inconclusive. Opponents there were King Level 4–5 against your 2 |
| Strategies | Champion `ladder-vision-v4` (`61f2df2e1ec1`); candidates v6 (`434fb3d8d263`) and level-aware v7 (`f42458bd6f37`). The 2026-11-27 policy release changed every ID; old IDs resolve through [`legacy_ids.yaml`](configs/strategies/legacy_ids.yaml) |
| Automation | `autopilot` runs farm → diagnose → propose → test → promote → report unattended ([§5](#autopilot-unattended-loop)); a weekly coding-agent job turns its reports into one structural-rule pull request |
| Studio | Live view (`play --ui`, `watch`), replay inspector (`inspect`) with a win-chance line, no-phone demo (`sim --ui`) |
| Tests | `pytest` 243 passed, 1 skipped without torch; `ruff check src tests tools` clean |

---

## 2. Install and smoke-test (no phone needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

pytest -ra
ruff check src/ tests/ tools/

# Synthetic closed loop (~3 ms loop latency, no phone); add --ui to watch it in the Studio
droid-harness sim --seconds 10 --out data/sim
droid-harness analyze --data data/sim --cards tests/fixtures/cards_min.json

# SPRT sample-size table for win-rate tests
droid-harness sprt
```

Optional extras: `[detector]` (ONNX Runtime), `[tuning]` (Optuna), `[gui]` (pywebview for a desktop Studio window), `[coach]` (offline LLM coach, rarely needed). PyTorch-ROCm for `TorchDetector` and training is installed separately on the GPU host (see [§10](#10-detector-retrain-or-add-cards)).

---

## 3. Play matches on a calibrated phone

`phone-a` has its calibration checked in under [`configs/`](configs/). Check `adb devices`, unlock the phone onto Clash Royale's home screen, then:

```bash
# Pre-flight: calibration, 10/10 digits, 8/8 deck templates, GPU detector, ADB
.venv/bin/droid-harness doctor --live

# Stuck on a loss/result screen or a menu? Walk it back to HOME (safe while the harness is stopped)
.venv/bin/droid-harness recover --dry-run   # print the plan
.venv/bin/droid-harness recover

# Play 10 ladder matches with v6, deleting the video of clean wins; --ui serves the Studio
.venv/bin/droid-harness play --strategy v6 --mode ladder --matches 10 --prune-win-video --ui

# Paired evaluation, v4 vs v6, in one persistent session (GPU weights and scrcpy load once)
.venv/bin/droid-harness experiment --champion v4 --candidate v6 --mode ladder --prune-win-video --ui \
  --title "v6 pro rules + tower HP digits" --hypothesis "Fewer opening throws; cycle-lock/chip/punish"
```

`--device phone-a` is the default preset. It resolves `profile`, `rois`, `tiles`, `arena`, `templates`, `screens`, `towers`, `digits`, `cards.json`, `models/yolo26n-cr.pt`, `models/classes.txt` and turns on `--async-detect`. Short strategy names resolve too (`v6` → `configs/strategies/ladder-vision-v6.yaml`, `champion` → whatever [`configs/strategies/CHAMPION`](configs/strategies/CHAMPION) points at).

**Modes** (`--mode`):

| Mode | Opponent | Use |
| --- | --- | --- |
| `ladder` | Trophy Road matchmaking | Farming and confirming a result; opponents drift with trophies |
| `training_camp` | The Trainer bot | Cheap, fixed opponent for tactical A/Bs |
| `practice` | A bot with a real opponent's deck and levels, from your battle log | Fixed-deck gauntlet; in an experiment both arms of a pair face the same battle-log entry (the pair number picks the entry, rotating through the first four; recorded as `opponent_id: practice:slotN`) |
| `friendly_host` / `friendly_guest` | Your second phone | Two-phone self-play (`--test h2h`, `selfplay-schedule`) |

The `practice` and `friendly_*` flows use buttons and screens that aren't calibrated on `phone-a` yet; `doctor --mode practice` lists what's missing. A device can override any flow under `flows:` in its ROI file instead of editing code.

Explicit paths still work for custom setups:

```bash
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
  --detector models/yolo26n-cr.pt --classes models/classes.txt --async-detect \
  --mode ladder --matches 10 --balance-epoch 2026-11 --data data/matches-batch
```

### Key runtime flags

| Flag | Purpose |
| --- | --- |
| `--detector <path>` | `.pt` runs `TorchDetector` (PyTorch GPU/ROCm); `.onnx` runs `OnnxDetector` (MIGraphX EP if built, else CPU). Omit only for a blind strategy such as [`training-camp-blind-v1.yaml`](configs/strategies/training-camp-blind-v1.yaml). |
| `--classes models/classes.txt` | Cross-checks the model's embedded class names; refuses to load on mismatch. |
| `--async-detect` | Detection on its own thread, so the 60 Hz HUD/policy/touch loop never waits for inference. |
| `--detect-every N` | Detector every Nth frame (default 1); units are extrapolated in between. |
| `--det-conf` / `--det-iou` | Confidence threshold (0.35) and NMS IoU (0.5). |
| `--matches N` | Completed matches to play unattended (menus, queue, result screen, popups). `Ctrl+C` stops cleanly. |
| `--data <dir>` | Output. Each match writes `<dir>/<match_id>/` with `meta.json`, `summary.json`, `decisions.jsonl`, `frames.jsonl`, `stream.h264` + `stream.pts`. |
| `--ui` | Serve the Studio while playing (also on `run`, `experiment`, `run-experiment`, `autopilot`, `sim`). See [§4](#4-studio-live-view-and-replay-inspector). |
| `--harvest-templates <dir>` | Save hand crops the template matcher can't read (a newly unlocked card). Review them, then `promote-templates` ([§10](#new-cards-without-hand-capturing-templates)). |

### Stuck-screen recovery (automatic)

A result screen that doesn't change for **8 s**, or any other non-battle screen for **15 s**, arms a deterministic walk to HOME inside the live loop:

1. the calibrated buttons for that screen (the normal flow, again);
2. wide saturated buttons found **visually** in the lower frame, rightmost first. This is what recovers a mis-calibrated `ok`: a ladder loss with a reward chest adds "Play Again" and shifts "OK" right, so the calibrated centre lands between them;
3. BACK through scrcpy;
4. force-stop and relaunch once the plan is exhausted (or after 60 s of liveness findings).

A battle whose clock, hand and elixir all stop changing for **30 s** is also treated as over. The purple post-match chest reads as a full elixir bar, so the session would otherwise stay "in battle"; it's recorded as `aborted:battle_no_progress` and handed to the same ladder. Recovery keeps acting on each new screen until HOME. A finished batch walks itself back to HOME before disconnecting, with the normal menu flow suspended so the settle can't queue a new game. The standalone `droid-harness recover` uses the same ladder (`--dry-run`, `--stuck-after`, `--battle-stuck-after`).

---

## 4. Studio: live view and replay inspector

One browser page, served by the harness itself (standard library only, no build step). It shows the phone with what perception found drawn on top, a top-down tactical map in arena tiles, and the numbers the policy decided from.

```bash
droid-harness play --strategy v6 --ui            # play and watch: http://localhost:8765
droid-harness experiment --champion v4 --candidate v6 --ui   # adds the SPRT strip
droid-harness watch                              # you play; the bot shows what it would do, never taps
droid-harness inspect                            # replay every match under data/
droid-harness inspect data/matches-exp-v4-v6 --match matches-exp-v4-v6/32dc3e473260
droid-harness sim --ui --seconds 60              # try the live view with no phone

# From another machine (e.g. the GPU host runs the bot):
ssh -L 8765:localhost:8765 nick@192.168.1.199    # then open http://localhost:8765
```

`--native` opens the same page in a desktop window instead (`pip install -e ".[gui]"`, which installs pywebview); without pywebview it falls back to the browser. `--open` opens the default browser, `--ui-port` and `--ui-host` change the address (it binds to localhost by default), `--ui-fps` caps the live snapshot rate (15).

**What's on screen**

- **Phone** (left): the live frame (or the decoded recording) with toggleable overlays: detector boxes with class, confidence and HP; tracked units with velocity vectors; the 18×32 tile grid and river; HUD regions; the top candidates as numbered rings; the chosen play as a gold ring with a dashed line to where its target is predicted to be; and fading rings where touches landed.
- **Tactical map**: the same moment in tile space: towers with HP, units by side (air units hollow), candidates and the chosen play. It needs no video, so it's the default for recorded matches whose video was pruned.
- **Right column**: clock, elixir phase and posture; your elixir and the opponent's estimated range; the hand with stale (unread) slots outlined; the decision table (every candidate's `score × weight × posture multiplier = total`, the act threshold as a line, the chosen row in gold, and each rule's reason); the opponent's 8-card cycle (what could be in hand, what's cycling back and how much elixir it takes to return); towers; the experiment strip (record per arm, pairs, dense margin, LLR between its bounds); latency per stage.
- **Replay** adds the elixir tape along the bottom: your elixir, the opponent's range, your plays, their plays and fallen towers over the whole match, with a playhead you can drag. Space plays or pauses, arrow keys step through logged states, Shift+arrows jump between chosen plays.
- **Win chance** (replay, when `models/value.json` exists): a badge with the value model's win probability at the playhead and a line on the tape, smoothed over 2 s. Sharp drops mark the moments worth scrubbing to. Train the model with `droid-harness train-value` ([§8](#8-analyze-matches)); `inspect --value <path>` uses another one.

**What it costs the bot.** The brain thread hands the Studio references it already holds (frame, observation, decision, timings, detector boxes): one lock and one tuple assignment per frame. A separate publisher thread samples that at `--ui-fps`, builds the snapshot JSON and the JPEG, and serves them; building took 0.6–0.8 ms per snapshot in the simulator. The Studio lives outside `policy/`, so it doesn't change strategy IDs.

**Replay data.** `inspect` reads `decisions.jsonl` (logged every 5th frame and on every play), `summary.json` and `meta.json`. If `stream.h264` survived retention, it's decoded once in the background at 10 fps. Detector boxes aren't logged, so recorded video shows tracked units projected from their logged tiles. Strategy IDs from older policy code resolve to a name through [`legacy_ids.yaml`](configs/strategies/legacy_ids.yaml); one that still doesn't resolve shows `× weight` derived from `total ÷ score`.

Code: [`src/droid_harness/ui/`](src/droid_harness/ui/) (`telemetry.py` tap and publisher, `snapshot.py` shared schema, `library.py` recorded matches, `server.py` routes, `static/studio.html` the page).

---

## 5. Improvement loop

Three speeds: seconds on CPU, hours on the phone, and a coding agent only for changes a parameter can't make. No LLM runs inside the harness. `autopilot` chains Speeds 1 and 2 unattended; the weekly agent job is Speed 3.

```bash
# 0. Pre-flight before every session (add --mode practice to check that flow's buttons)
.venv/bin/droid-harness doctor --live

# The whole loop, unattended (see "Autopilot" below)
.venv/bin/droid-harness autopilot --testbed training_camp --test dense --cycles 0 --ui

# Or step by step:
# 1. Collect games with the champion
.venv/bin/droid-harness play --strategy champion --mode ladder --matches 30 --prune-win-video

# 2. Speed 1: offline, between phone runs (seconds, no phone, no LLM)
.venv/bin/droid-harness tune-targeted --champion v4 --data data --scan          # pattern hits
.venv/bin/droid-harness tune-targeted --champion v4 --data data \
  --out configs/strategies/ladder-vision-v4-t1.yaml                             # 1-2 knob fix
.venv/bin/droid-harness shadow --champion v4 --candidate configs/strategies/ladder-vision-v4-t1.yaml

# 3. Speed 2: paired live test of candidate vs champion. Randomized pairs cancel opponent drift;
#    clearly worse candidates stop early (dense-margin futility); the result goes to ledger/
.venv/bin/droid-harness experiment --champion v4 --candidate configs/strategies/ladder-vision-v4-t1.yaml \
  --mode training_camp --test dense --title "narrower defend trigger" --hypothesis "less over-defending"
.venv/bin/droid-harness ledger        # renders ledger/STRATEGY_LEDGER.md

# 4. Speed 3: a <30 KB digest for a coding agent (the weekly job reads the autopilot's copy)
.venv/bin/droid-harness digest --data data/matches --out data/agent_digest.md

# 5. Housekeeping
.venv/bin/droid-harness retain --data data/matches   # prune win videos, keep loss keyframes
```

Notes:
- Use a new `--balance-epoch` after a game update; experiments are void across updates. They're also void across an upgrade (the `account_levels` recorded on every match changed).
- `tune-offline` (the older value-weighted search, 2–5 knobs) is still there, but it rewards changing decisions inside losing engagements without knowing whether the new decision is better. Prefer `tune-targeted`.

### The three speeds in detail

1. **Speed 1, targeted counterfactual search** (`tune-targeted`; CPU seconds, no phone, no LLM). Each digest failure pattern is a predicate over logged states ([`targeted.py`](src/droid_harness/flywheel/tuning/targeted.py)): `opening_throw` (a play in the first 15 s that leaves you under 3.2 elixir while they may hold ≥ 6.5), `bad_counter_timing` (win condition played while a hard counter they've shown is out of their last four plays, so in hand, and affordable), `over_defend_bleed` (defending a push worth under 3 elixir down to under 2 elixir), `wincon_starved` (≥ 8.5 elixir, a win condition ready in hand, an empty board, and something else played), `spell_inefficiency` (a spell whose own value estimate is negative). The search picks the champion's most frequent pattern (or `--pattern`), tries 1–2 knobs from that pattern's list, and keeps a candidate only when it removes at least `--min-removed` of the hits (default 50 %) while changing at most `--max-collateral` % (default 3 %) of the other decisions. One or two knobs keep each experiment one hypothesis (R8). It's a filter: whether the new decision is better is what Speed 2 measures.
2. **Speed 2, persistent paired live gate** (`experiment`; phone hours, no LLM). One `BotRunner` (detector weights and scrcpy loaded once, `CRPolicy` hot-swapped per match) plays randomized two-game blocks. **Stage A** scores every pair by crowns + tower-HP difference + speed and stops a candidate early when the upper 95 % bound of the mean pair margin is below zero (`--futility-pairs`). **Stage B** decides: `--test pairs` runs the SPRT on discordant pairs (win/loss only), `--test dense` runs an always-valid test on the paired margin itself, which uses every pair rather than only the discordant ones, so it usually needs fewer games ([§12](#12-evaluation-and-statistics)). The result goes to the [ledger](ledger/STRATEGY_LEDGER.md).
3. **Speed 3, structural rule** (a coding agent, weekly). The digest (failure clusters, loss stories, matchup matrix and, with a value model, the moments the win chance fell hardest) goes to an agent that adds **one** rule as an opt-in parameter with a unit test, in a new strategy file, as a pull request you review. The scheduled job does this from the autopilot's published reports; numeric tuning of the new rule then goes back to Speed 1.
4. **Tie-breaker, M8 v2 residual ranker** (`learn/label_from_video.py --version 2` + `train_ranker.py`). Per-lane board value, threat ETA and opponent-cycle features (`droid-harness-ranker/2`), engagement-credit advantage-weighted / filtered BC, blended as a bounded residual (`ranker_mode: residual`, `rule_score + w·(p − 0.5)`) so safety and counter gates are never overwritten.

### Autopilot (unattended loop)

`droid-harness autopilot` ([`flywheel/autopilot.py`](src/droid_harness/flywheel/autopilot.py)) runs cycles of:

1. **Farm** `--farm-matches` (20) with the current champion in `--farm-mode` (ladder), then prune clean-win video.
2. **Diagnose.** Retrain the value model on everything farmed (once there are enough matches) and write the digest for this batch.
3. **Propose.** `tune-targeted` on the champion's most frequent failure pattern; if nothing passes, the next one. The candidate is a new file, `configs/strategies/auto/<champion>-<pattern>-<stamp>.yaml`.
4. **Test** it against the champion with a paired experiment on `--testbed` (`training_camp` or `practice` for fixed opponents; `ladder` to confirm), `--test pairs|dense`, capped at `--max-games` with Stage-A futility after `--futility-pairs`.
5. **Promote** on acceptance: [`configs/strategies/CHAMPION`](configs/strategies/CHAMPION) moves to the candidate. Strategy files are never edited, and rejections stay in the ledger so the same idea isn't retried.
6. **Report** to `reports/<stamp>/` (`report.md`, `report.json`, `digest.md`) and `reports/latest.md`. With `--notify-url` it POSTs a one-line summary (an [ntfy.sh](https://ntfy.sh) topic URL puts it on your phone). With `--publish-git` it commits the reports, the ledger, `CHAMPION` and `auto/` to the `autopilot/reports` branch through a separate worktree, so your checkout is never touched, and pushes it.

`--cycles 0` runs until `data/autopilot/STOP` exists (`touch` it to stop after the current cycle). `--ui` serves the Studio for the whole run. The autopilot never writes code: new rules arrive as pull requests from the weekly job, and you merge them.

**Run it as a service** on the bot machine with [`tools/systemd/droid-autopilot.service`](tools/systemd/droid-autopilot.service) (setup steps in the file): it waits for the phone, runs `doctor` first, restarts on failure after 5 minutes, and `loginctl enable-linger` keeps it running after you log out.

**Weekly coding-agent job** (a scheduled task in Claude, Saturdays 14:17 Pacific). It reads `reports/latest.md` and `latest-digest.md` from `autopilot/reports` and does nothing when there's no report from the past 8 days or its previous PR is still open. Otherwise it picks one structural weakness the reports show and parameters didn't fix, implements it as an opt-in parameter (default off, so existing strategies don't change behaviour) with a unit test and a new strategy YAML copied from the champion, checks it with `pytest`, `ruff`, `replay-diff`, `shadow` and a pattern scan, and opens a PR against `main`. It never merges. Merging and letting the autopilot test it is your call.

### What the digest contains

`droid-harness digest --data <dir> --out data/agent_digest.md` writes six deterministic sections (never raw `decisions.jsonl`):

1. **Batch health and drift slices** by chronological quartile: W–L–D with Wilson 95 % CI, crown distribution, median duration, aborts, loop p50/p99, retry rate, 8-card deck-lock rate, triage mix.
2. **Opponent card matchup matrix**: win rate and crown differential when facing each card seen in `opp_deck_seen`.
3. **Rule and posture scorecard**: plays, share, mean and total engagement value, mean `(my_elixir, opp_elixir)` when triggered.
4. **Six failure-pattern detectors**: `opening_throw` (early plays leaving < 3 elixir while they hold ≥ 6.5, then tower damage before 60 s), `wincon_starved` (win condition played ≤ 1 time in a ≥ 120 s match), `over_defend_bleed` (> 75 % of frames below 5 elixir), `spell_inefficiency`, `bad_counter_timing` (win condition into an in-hand, affordable counter vs when it was ≥ 2 cards away), card utilization skew.
5. **Top 8–12 strategy loss stories** with elixir context and the runner-up candidates.
6. **Optional visual pack** (≤ 5 keyframes, < 250 KB) when a perception blind spot is suspected.

### Things only you can do

- **Card levels.** Upgrade cards that survive into the Hog EQ deck (Musketeer, Knight, Fireball); keep Giant, Mini P.E.K.K.A, Archers, Minions and Arrows at minimum. After every upgrade session, update [`configs/accounts/theblackcat99.yaml`](configs/accounts/theblackcat99.yaml) ([§6](#account-levels)). `doctor` tells you when a strategy's `tier0.spell_tower_damage` no longer matches your spell levels.
- **New cards after an arena unlock.** Play a few matches with `--harvest-templates data/harvest`, then `promote-templates` (or capture by hand with `droid-harness template`; `doctor` flags missing ones), add counter rows or rely on the counter prior, then fine-tune the detector ([§10](#10-detector-retrain-or-add-cards)).
- **Calibrate the Practice and friendly flows** (battle-log Practice buttons, social/friendly screens) before using them as a testbed: `doctor --mode practice` lists what's missing.
- **Autopilot plumbing:** pick an ntfy topic, install the systemd unit, and decide when to merge the weekly agent's PRs.
- **Hog Rider (Arena 5).** Switch to the Hog cycle blueprint in [§7](#7-strategy-playbook).
- **M8 ranker** after ~50+ post-digits ladder matches: `python -m droid_harness.learn.label_from_video --version 2 --cards configs/domains/clash_royale/cards.json --min-credit -0.5`, then `train_ranker`, then test it with `experiment` like any candidate.
- Ladder botting can get the account banned; that risk is yours.

---

## 6. Strategies: files, rules, promotion

**Immutability rule.** Never modify a strategy YAML or counter CSV once it has logged matches. Copy it to a new version (`ladder-vision-v6.yaml` → `ladder-vision-v7.yaml`), update `strategy:` and `parent:`, and point `counters:` at its own CSV. Every strategy gets a 12-character `strategy_id` hashing the YAML, its counter CSV, any ranker model, **and every file under [`policy/`](src/droid_harness/domains/clash_royale/policy/)**: editing policy code changes every strategy's ID, which voids running experiments. New policy features therefore ship as **opt-in parameters, default off**, batched into one release, with a check that existing strategies make identical decisions on logged states.

**Current strategies.**

| File | ID | Notes |
| --- | --- | --- |
| [`ladder-vision-v4.yaml`](configs/strategies/ladder-vision-v4.yaml) | `61f2df2e1ec1` | Champion ([`CHAMPION`](configs/strategies/CHAMPION)) |
| [`ladder-vision-v6.yaml`](configs/strategies/ladder-vision-v6.yaml) | `434fb3d8d263` | Pro-rules candidate (EXP-0001 inconclusive) |
| [`ladder-vision-v7.yaml`](configs/strategies/ladder-vision-v7.yaml) | `f42458bd6f37` | v6 + `level_aware: true`, `level_gap_respect: 0.08` |

The 2026-11-27 policy release (levels, counter prior) changed every ID: v4 was `639c09fe0c3a` and v6 `ab2d9564c216`. Both make the same decisions as before on all 28,744 logged states. [`legacy_ids.yaml`](configs/strategies/legacy_ids.yaml) maps old IDs to names so the Studio, ledger and digest still label old matches.

**Champion pointer.** [`configs/strategies/CHAMPION`](configs/strategies/CHAMPION) holds the path of the current champion, relative to `configs/strategies/`. `--strategy champion` resolves it; the autopilot reads it and moves it when a candidate is accepted.

### Account levels

Card levels change the bot's strength without changing any strategy ID, so they're data: [`configs/accounts/<account>.yaml`](configs/accounts/theblackcat99.yaml) (the account comes from the device profile) holds `king_level` and each deck card's level on the game's unified 1–15 scale; `null` means not recorded yet. Every match records them in `meta.json` as `account_levels`, and an experiment is voided if they change partway.

Perception reads each side's King Level from its Princess Tower max HP (1400 → 1, 1512 → 2, 1624 → 3, 1750 → 4, 1890 → 5, …), so every match also logs the level gap. `doctor` compares the file with what it measured and with each strategy's `tier0.spell_tower_damage`.

With `params.level_aware: true` ([`policy/levels.py`](src/droid_harness/domains/clash_royale/policy/levels.py)):
- card HP and damage scale by `1.1^(level − start level)` (start levels: common 1, rare 3, epic 6, legendary 9, champion 11). Your cards use the file, falling back to King Level; the opponent's use their inferred King Level;
- kill-aware spells use scaled damage against scaled HP. Tier 0's lethal check derives a spell's tower damage from your levels when `tier0.spell_tower_damage` doesn't list that spell (a listed value still wins, so leave it out of new strategies and let the levels decide);
- `level_gap_respect: r` multiplies defense by `1 + r·gap` and divides offense by it when you're under-leveled (`gap` = their King Level − yours).

### Counter prior

A missing row in a counters CSV used to mean "this card never defends against that threat". With `params.counter_prior: true` ([`policy/counter_prior.py`](src/droid_harness/domains/clash_royale/policy/counter_prior.py)), a missing pair gets a coarse effectiveness from card attributes instead (the card DB has HP, range, unit count, targets and cost, but no DPS or splash): buildings pull building-targeters, swarms and heavy single troops answer tanks, ranged troops answer most things a little, spells only answer what they kill (more for swarms), and catch-all `other_*` threats get a middling value. `Card.can_hit` still gates it, and `counter_prior_scale` scales it. A CSV row always wins, even `0`. This is what lets a new deck, or an opponent card you've never written a row for, get sensible defense on day one.

### Strategy file structure ([`policy/strategy.py`](src/droid_harness/domains/clash_royale/policy/strategy.py))

| Section | Key fields | Meaning |
| --- | --- | --- |
| Identity | `strategy`, `parent`, `deck`, `win_conditions` | Name, lineage, the exact 8 card keys from [`cards.json`](configs/domains/clash_royale/), win-condition card(s). |
| Tempo and defense | `act_threshold`, `elixir_weight`, `defend_trigger_tiles`, `defend_horizon_s`, `opening_ms`, `min_action_gap_ms` | Minimum utility to play, elixir penalty, defense radius and ETA horizon. |
| Bridge gating | `min_elixir_to_attack`, `press_opp_elixir_max`, `max_opp_elixir_to_bridge`, `min_elixir_advantage_to_bridge`, `attack_blocked_by_threat_eta_s` | Elixir floors and enemy-elixir ceilings before playing a win condition at the bridge. |
| Counter-push and back build | `counterpush_min_value`, `counterpush_tank_ahead`, `counterpush_tank_bonus`, `giant_back_build`, `giant_safe_build_elixir`, `giant_back_build_score`, `giant_build_block_tiles`, `defense_reserve`, `reserve_penalty`, `giant_back_build_allow_in_opening`, `giant_x1_min_elixir` | Snowball surviving defenders behind a tank; back-build Giant only above a safe elixir level while keeping reserve. |
| Card, cycle and spell IQ | `counter_gate`, `wc_counters`, `counter_in_hand_mult`, `counter_out_mult`, `counter_unaffordable_punish_mult`, `back_build_counter_in_hand_mult`, `cycle_punish_partial_deck`, `use_min_cycle_elixir`, `spell_aware`, `kill_aware_spells`, `save_counters_for_in_hand_threats`, `spell_value_threshold`, `spell_cycle_max_casts` | Scales pushes by the opponent's deck, cycle and affordability; kill-aware spell value; protects tank killers. |
| Pro extensions (v6) | `ignore_chip_tower_hp_frac`, `ignore_chip_max_threat_value`, `opposite_lane_punish`, `punish_min_enemy_cost`, `punish_enemy_back_y`, `punish_min_my_elixir`, `punish_score`, `punish_cards`, `placement.anti_spell_tower_offset` | Tower HP as a resource, opposite-lane punish, anti-spell spacing. |
| Tier 0 reflexes (`tier0:`) | `lethal_margin_hp`, `spell_tower_damage`, `leak_threshold`, `leak_ms`, `leak_card_preference`, `leak_tiles`, `emergency_eta_s`, `emergency_hp_frac` | Unconditional tower-finishing spells, emergency defense below 35 % HP, anti-leak dumps. Set `spell_tower_damage` to **your** card levels. |
| Placement (`placement:`) | `building_pull`, `ranged_standoff_tiles`, `bridge_tiles`, `back_tiles` | Anchor tiles and standoff distance on the 18×32 grid. |
| Weights and postures | `weights`, `posture_multipliers` | Per-rule multipliers and scaling for `OPENING`, `BALANCED`, `DEFEND_LEAD`, `PRESS`, `SPELL_CYCLE`. |
| Counters CSV (`counters:`) | `card,threat,effectiveness` (0..1) | Defense matchup matrix, always subject to `Card.can_hit(flying)`. |
| Levels and counters (2026-11-27) | `level_aware`, `level_gap_respect`, `counter_prior`, `counter_prior_scale` | Level-scaled numbers and the counter prior, below. All off by default. |
| Ranker and advisor | `learned_ranker`, `ranker_mode`, `ranker_residual_weight`, `intent_advisor`, `advisor_may_set` | Optional M8 ranker and M9 advisor (off by default). |

A candidate's final value is `score × weights[rule] × posture_multipliers[posture][family]`; the best one is played if it clears `act_threshold`. The Studio's decision table shows exactly this breakdown.

### Promotion funnel (deterministic gates)

1. **Unit tests and lint:** `pytest -ra && ruff check src/ tests/ tools/`.
2. **Offline counterfactual diff:** how often the candidate changes logged decisions, before any match.
   ```bash
   droid-harness replay-diff --champion v4 --candidate v6 --data data/matches --expected-pct 5.0 --tolerance 4.0
   ```
3. **Shadow scoring and offline search (optional):** `droid-harness shadow` (disagreements × engagement value), `tune-offline`, or Optuna (`flywheel/tuning/optuna_study.py`, `pip install -e ".[tuning]"`).
4. **Paired live test:** `droid-harness experiment` (ladder defaults) or `run-experiment` (Training Camp defaults). Randomized blocks of two (`--test pairs` or `dense`) cancel matchmaking drift; every result goes to [`ledger/experiments.jsonl`](ledger/experiments.jsonl) and [`ledger/STRATEGY_LEDGER.md`](ledger/STRATEGY_LEDGER.md). Never use `--test fixed` to promote; use `pairs`/`dense`, or `h2h` for two-phone self-play (`selfplay-schedule`).
5. **Promotion:** move [`CHAMPION`](configs/strategies/CHAMPION) to the accepted file (the autopilot does this itself).

---

## 7. Strategy playbook

Synthesized from web research (2026-11-26) on top-ladder macro and micro play and the Arena 1–5 card progression, translated into engine capabilities and valid strategy YAML.

### Archetypes

High-win-rate ladder decks exploit elixir and cycle rather than spamming one lane: control and bridge-spam hybrids (force elixir splits across both lanes, turn clean defense into immediate counter-pressure), anchored control beatdown (one tank or ranged anchor plus high-value defenders, two spells and cheap cycle), and 2.6–3.0 cycle or spell bait (out-rotate the opponent's one counter).

| Archetype | Win conditions | How it wins | Edge against | Weak against |
| --- | --- | --- | --- | --- |
| Beatdown | Giant, Golem, Lava Hound, Electro Giant | Survive single elixir, back-plant the tank at +2..+4 or in double elixir, stack DPS behind it | Control, siege | Fast cycle and bridge-spam punishes; hard tank killers |
| Cycle | Hog Rider, Wall Breakers, Miner | 2.6–3.0 average; out-cycle the counter building or spell, defend positively | Beatdown | Overlapping counters, heavy splash |
| Control | Graveyard, Miner + Poison, Royal Giant + Fisherman | +2..+4 trades, full-HP defenders become counter-pushes | Bridge spam, aggro | Multi-wave beatdown in double/triple elixir |
| Spell bait | Goblin Barrel, Goblin Drill, Skeleton Barrel | Bait the small spell, punish during its 3-card cooldown | Single-spell decks | Two small spells or fast cycle |
| Bridge spam | Battle Ram, Ram Rider, Bandit, Prince | Punish the opposite lane whenever the opponent invests ≥ 5 in the back | Back-plant beatdown, siege | Cheap cycle with disciplined reserves |
| Siege | X-Bow, Mortar | Protect a building on your side; spell-cycle the rest | Control without a big tank | Tanks that soak building fire |

### Upgrade path: starter deck → 2.6 Hog EQ (`theblackcat99`)

Ladder climbing raises opponents' card levels, so **gold and wildcards go only into cards that survive into the Hog EQ cycle deck** (Hog Rider, Earthquake [Fireball until Arena 9], The Log / Barbarian Barrel [Arrows until unlocked], Musketeer / Firecracker, Cannon, Knight / Ice Golem, Skeletons, Ice / Fire Spirit):

1. **Now (Arena 1 → King Level 2):** one cheap upgrade reaches King Level 2 (+112 Princess HP, +168 King HP, +8 % tower DPS) and unlocks Bone Pit. Musketeer, Knight and Fireball can be upgraded freely: all three carry through Arena 9.
2. **Arena 2 (Bone Pit, 400+):** unlock and slot Skeletons. Optional bridge card: Valkyrie or Bomber, without spending gold on Giant, Minions or Archers.
3. **Arena 3 (Barbarian Bowl, 800+):** Cannon (the Hog deck's building) and Barbarian Barrel (replaces Arrows).
4. **Arena 4 (Spell Valley, 1400+):** Fire Spirit or Electro Spirit.
5. **Arena 5 (Builder's Workshop, 1600+):** Hog Rider: switch from Giant beatdown to the Hog cycle blueprint (`[hog_rider, knight, cannon, musketeer, skeletons, fire_spirit, fireball, barbarian_barrel]`, average 2.8).
6. **Arena 8–9:** Ice Spirit / Ice Golem, then Earthquake: full Hog EQ.

> **Before a new card goes into `deck:`** capture its `affordable` and `greyed` templates, add counter rows to its CSV, and add `<unit>_me` / `<unit>_opp` to `models/classes.txt` if the detector lacks it (then retrain, [§10](#10-detector-retrain-or-add-cards)).

### Eight mechanics the engine models

1. **Elixir gap and phase pacing.** Regeneration: 1 per 2.8 s in single, 1.4 s in double, ~0.9 s in triple overtime. Spending down to < 3 while the opponent holds 7.5–10 loses to opposite-lane rushes (v5's three ~95 s 0–3 losses opened with 1.1 s back Giants at 7.2–7.5 elixir). Never back-plant a Giant in single elixir unless at ≥ 8.5 (keeping ≥ 3.5 in reserve) or after a positive defense.
2. **Four-card cycle and the elixir cost to return a counter.** A played card returns only after four other plays. If `opp_elixir + regen < min_cycle_elixir_to_return(counter)`, the win condition has an open window (`CycleTracker.min_cycle_elixir_to_return`).
3. **Opposite-lane punish vs same-lane build.** When they spend ≥ 5 behind their King Tower in single elixir, punish the other lane if even or ahead; always keep one anti-punish card affordable when back-building.
4. **Counter-push snowballing.** Defend on your side at a +2..+4 profit, then plant the tank in front of the survivors before they cross.
5. **Tower HP as a resource.** On a healthy Princess Tower (> 60 %), accept 200–350 chip from low-value pokes rather than spending 3 elixir every 5 s; respect high-DPS threats and swarms.
6. **Tile geometry** ([`arena.py`](src/droid_harness/domains/clash_royale/arena.py)). Center building pull at `x 7.5–10.5, y ≈ 10.5` drags building-targeters through both towers; cross-lane melee kite near `(7.5–10.5, 11–13)`; ranged defenders 3–4 tiles behind the contact tile; units are tracked by their foot (shadow) tile.
7. **Spell value discipline.** No Fireball on a lone tank; spread backline defenders ≥ 3.5 tiles from your own towers when they hold Fireball or Poison.
8. **Overtime spell cycle.** With ≤ 2–3 spells needed to finish a low tower in overtime, defend and rotate spells onto it (`SPELL_CYCLE`).

### Capability audit

| Mechanic | State | Mechanism |
| --- | --- | --- |
| Elixir-gated bridge pushes and Giant back-build | YAML | `max_opp_elixir_to_bridge`, `min_elixir_advantage_to_bridge`, `giant_safe_build_elixir`, `giant_build_block_tiles`, `defense_reserve`, `reserve_penalty` |
| Counter availability and affordability | YAML | `counter_gate`, `wc_counters`, `counter_in_hand_mult`, `counter_out_mult`, `counter_unaffordable_punish_mult`, `cycle_punish_partial_deck`, `back_build_counter_in_hand_mult` |
| Kill-aware spells, Minions vs Arrows | YAML | `spell_aware`, `kill_aware_spells`, `spell_value_threshold`, `tier0.spell_tower_damage` |
| Save tank killers while cycling | YAML | `save_counters_for_in_hand_threats` ([`rules/cycle.py`](src/droid_harness/domains/clash_royale/policy/rules/cycle.py)) |
| Counter-push tank ahead | YAML | `counterpush_tank_ahead`, `counterpush_tank_bonus`, `counterpush_min_value` |
| Overtime spell cycle | YAML | `posture.py::SPELL_CYCLE`, `spell_cycle_max_casts`, `tier0.lethal_spell` |
| Opening / single-elixir guard on Giant back-build | Code, v6 | `giant_back_build_allow_in_opening: false`, `giant_x1_min_elixir` |
| `min_cycle_elixir_to_return` in scoring | Code, v6 | `use_min_cycle_elixir: true` |
| Tower-HP-as-resource chip filter | Code, v6 | `ignore_chip_tower_hp_frac`, `ignore_chip_max_threat_value` |
| `OppositeLanePunish` rule | Code, v6 | `opposite_lane_punish` and `punish_*` |
| Cross-lane kite, anti-spell offset | Code, v6 | `placement.anti_spell_tower_offset` (3.2 tiles) |
| Predictive pre-hover Arrows on a bridge Giant push | Open | Preselect Arrows ahead of the Giant when Skeleton Army or Minion Horde can be played |

### Blueprints

Each uses only keys accepted by `StrategySpec`. **Blueprint 1** is checked in as [`ladder-vision-v6.yaml`](configs/strategies/ladder-vision-v6.yaml) (+ [`counters/ladder-vision-v6.csv`](configs/strategies/counters/ladder-vision-v6.csv)): it fixes v5's opening throws with `defense_reserve: 2.5` and `giant_safe_build_elixir: 8.6`, narrower hard-counter `wc_counters`, moderate multipliers (0.55 / 1.45) and the v6 extensions above. The others need templates and detector classes for their cards before use. They ship with `counters: null`, so **add `counter_prior: true` and `level_aware: true` to their `params:`** when you instantiate one: without the prior a card with no CSV row never defends, and with `level_aware` you can leave `tier0.spell_tower_damage` empty and let your account levels supply it ([§6](#account-levels)). Write CSV rows only where the prior is wrong.

<details><summary>Blueprint 2: 2.6 Hog cycle (<code>pro-hog-cycle-blueprint</code>)</summary>

Out-rotates building counters (Cannon, Tesla, Tombstone, Inferno Tower) with high bridge tempo.

```yaml
strategy: pro-hog-cycle-blueprint
parent: null
deck: [hog_rider, ice_golem, cannon, musketeer, skeletons, ice_spirit, fireball, the_log]
win_conditions: [hog_rider]
params:
  act_threshold: 0.08
  elixir_weight: 0.25
  defend_trigger_tiles: 11.5
  defend_horizon_s: 7.5
  min_elixir_to_attack: 4.5
  press_opp_elixir_max: 4.5
  max_opp_elixir_to_bridge: 6.5
  min_elixir_advantage_to_bridge: 1.0
  spell_value_threshold: 3.5
  spell_cycle_max_casts: 4
  opening_ms: 12000
  min_action_gap_ms: 350
  counter_push_bonus: 0.45
  counterpush_min_value: 2.0
  counterpush_tank_ahead: true
  counterpush_tank_bonus: 0.20
  giant_back_build: false
  attack_blocked_by_threat_eta_s: 5.5
  counter_gate: true
  wc_counters: [cannon, tesla, tombstone, bomb_tower, inferno_tower, mini_pekka, pekka, skeleton_army, tornado, fisherman]
  counter_in_hand_mult: 0.40
  counter_out_mult: 1.60
  counter_unaffordable_punish_mult: 1.35
  cycle_punish_partial_deck: true
  defense_reserve: 1.5
  reserve_penalty: 0.65
  spell_aware: true
  kill_aware_spells: true
  save_counters_for_in_hand_threats: false
tier0:
  lethal_margin_hp: 20
  spell_tower_damage: {fireball: 207, the_log: 58}  # update to your card levels
  leak_threshold: 9.2
  leak_ms: 600
  leak_card_preference: [skeletons, ice_spirit, ice_golem, the_log, musketeer]
  emergency_eta_s: 1.6
  emergency_hp_frac: 0.35
placement:
  building_pull: {left: [8.0, 10.5], right: [10.0, 10.5]}
  ranged_standoff_tiles: 3.5
  bridge_tiles: {left: [2.5, 14.0], right: [15.5, 14.0]}  # edge hog pig-push alignment
weights:
  defend.lane_threat: 1.05
  defend.air_threat: 1.05
  offense.win_condition_when_opp_low: 1.15
  offense.counter_push: 0.85
  spell.value: 0.90
  spell.tower_chip: 0.85
  opening.safe_first_play: 1.0
  cycle.cheap_cycle: 1.20
posture_multipliers:
  OPENING:     {offense: 0.7, cycle: 1.4}
  DEFEND_LEAD: {offense: 0.8, defend: 1.35}
  PRESS:       {offense: 1.5, cycle: 1.2}
  SPELL_CYCLE: {spell: 2.2, offense: 0.7}
counters: null  # add a counters CSV covering hog/tank/swarm defense
```
</details>

<details><summary>Blueprint 3: control / counter-push (<code>pro-control-blueprint</code>)</summary>

Positive defensive trades first, high counter-push reward, strict anti-punish reserve.

```yaml
strategy: pro-control-blueprint
parent: null
deck: [royal_giant, knight, baby_dragon, mini_pekka, skeletons, bomber, fireball, arrows]
win_conditions: [royal_giant]
params:
  act_threshold: 0.10
  elixir_weight: 0.45
  defend_trigger_tiles: 10.5
  defend_horizon_s: 7.5
  min_elixir_to_attack: 6.5
  press_opp_elixir_max: 3.5
  max_opp_elixir_to_bridge: 5.0
  min_elixir_advantage_to_bridge: 2.5
  spell_value_threshold: 4.0
  spell_cycle_max_casts: 3
  opening_ms: 20000
  min_action_gap_ms: 350
  counter_push_bonus: 0.85
  counterpush_min_value: 2.5
  counterpush_tank_ahead: true
  counterpush_tank_bonus: 0.30
  giant_back_build: false
  attack_blocked_by_threat_eta_s: 6.5
  counter_gate: true
  wc_counters: [inferno_tower, mini_pekka, pekka, barbarians, goblin_cage, cannon, tombstone]
  counter_in_hand_mult: 0.45
  counter_out_mult: 1.50
  counter_unaffordable_punish_mult: 1.35
  cycle_punish_partial_deck: true
  defense_reserve: 3.0
  reserve_penalty: 0.45
  spell_aware: true
  kill_aware_spells: true
  save_counters_for_in_hand_threats: true
tier0:
  lethal_margin_hp: 25
  spell_tower_damage: {fireball: 207, arrows: 93}
  leak_threshold: 9.4
  leak_ms: 900
  leak_card_preference: [skeletons, bomber, knight, baby_dragon]
  emergency_eta_s: 1.8
  emergency_hp_frac: 0.35
weights:
  defend.lane_threat: 1.15
  defend.air_threat: 1.15
  offense.win_condition_when_opp_low: 0.90
  offense.counter_push: 1.15
  spell.value: 0.95
  spell.tower_chip: 0.65
  opening.safe_first_play: 1.0
  cycle.cheap_cycle: 1.0
posture_multipliers:
  OPENING:     {offense: 0.4, defend: 1.1, cycle: 1.2}
  DEFEND_LEAD: {offense: 0.6, defend: 1.4}
  PRESS:       {offense: 1.4}
  SPELL_CYCLE: {spell: 2.0, defend: 1.2, offense: 0.5}
counters: null
```
</details>

<details><summary>Blueprint 4: dual-lane bridge spam (<code>pro-bridgespam-blueprint</code>)</summary>

Punishes opponent elixir deficits immediately while holding P.E.K.K.A and Musketeer for heavy defense.

```yaml
strategy: pro-bridgespam-blueprint
parent: null
deck: [battle_ram, bandit, pekka, prince, musketeer, minions, fireball, arrows]
win_conditions: [battle_ram, prince]
params:
  act_threshold: 0.08
  elixir_weight: 0.30
  defend_trigger_tiles: 12.0
  defend_horizon_s: 8.0
  min_elixir_to_attack: 4.5
  press_opp_elixir_max: 5.0
  max_opp_elixir_to_bridge: 6.0
  min_elixir_advantage_to_bridge: 1.0
  spell_value_threshold: 3.5
  spell_cycle_max_casts: 2
  opening_ms: 12000
  min_action_gap_ms: 350
  counter_push_bonus: 0.70
  counterpush_min_value: 2.5
  counterpush_tank_ahead: true
  counterpush_tank_bonus: 0.30
  giant_back_build: false
  attack_blocked_by_threat_eta_s: 5.0
  counter_gate: true
  wc_counters: [tombstone, goblin_cage, cannon, barbarians, skeleton_army, valkyrie, mega_knight]
  counter_in_hand_mult: 0.60
  counter_out_mult: 1.45
  counter_unaffordable_punish_mult: 1.40
  cycle_punish_partial_deck: true
  defense_reserve: 2.0
  reserve_penalty: 0.55
  spell_aware: true
  kill_aware_spells: true
  save_counters_for_in_hand_threats: true
  opposite_lane_punish: true
tier0:
  lethal_margin_hp: 20
  spell_tower_damage: {fireball: 207, arrows: 93}
  leak_threshold: 9.0
  leak_ms: 700
  leak_card_preference: [bandit, minions, battle_ram, knight]
  emergency_eta_s: 1.6
  emergency_hp_frac: 0.35
weights:
  defend.lane_threat: 1.0
  defend.air_threat: 1.0
  offense.win_condition_when_opp_low: 1.25
  offense.counter_push: 1.0
  spell.value: 0.85
  spell.tower_chip: 0.5
  opening.safe_first_play: 0.9
  cycle.cheap_cycle: 1.0
posture_multipliers:
  OPENING:     {offense: 0.8, cycle: 1.1}
  DEFEND_LEAD: {offense: 0.7, defend: 1.3}
  PRESS:       {offense: 1.6}
  SPELL_CYCLE: {spell: 1.8, offense: 0.7}
counters: null
```
</details>

<details><summary>Blueprint 5: spell bait (<code>pro-spellbait-blueprint</code>)</summary>

Holds Goblin Barrel while their small spell is in hand and affordable, spikes as soon as it leaves the cycle, finishes with Rocket in overtime.

```yaml
strategy: pro-spellbait-blueprint
parent: null
deck: [goblin_barrel, princess, goblin_gang, knight, valkyrie, inferno_tower, rocket, the_log]
win_conditions: [goblin_barrel]
params:
  act_threshold: 0.08
  elixir_weight: 0.35
  defend_trigger_tiles: 12.0
  defend_horizon_s: 8.5
  min_elixir_to_attack: 4.0
  press_opp_elixir_max: 5.0
  max_opp_elixir_to_bridge: 7.5
  min_elixir_advantage_to_bridge: 0.5
  spell_value_threshold: 4.5
  spell_cycle_max_casts: 4
  opening_ms: 15000
  min_action_gap_ms: 350
  counter_push_bonus: 0.40
  counterpush_min_value: 2.0
  counterpush_tank_ahead: true
  counterpush_tank_bonus: 0.25
  giant_back_build: false
  attack_blocked_by_threat_eta_s: 5.5
  counter_gate: true
  wc_counters: [the_log, arrows, zap, barbarian_barrel, giant_snowball]
  counter_in_hand_mult: 0.35
  counter_out_mult: 1.70
  counter_unaffordable_punish_mult: 1.45
  cycle_punish_partial_deck: true
  defense_reserve: 2.5
  reserve_penalty: 0.60
  spell_aware: true
  kill_aware_spells: true
  save_counters_for_in_hand_threats: false
tier0:
  lethal_margin_hp: 25
  spell_tower_damage: {rocket: 371, the_log: 58}
  leak_threshold: 9.2
  leak_ms: 700
  leak_card_preference: [princess, the_log, knight, goblin_gang]
  emergency_eta_s: 1.7
  emergency_hp_frac: 0.35
weights:
  defend.lane_threat: 1.1
  defend.air_threat: 1.1
  offense.win_condition_when_opp_low: 1.25
  offense.counter_push: 0.70
  spell.value: 0.90
  spell.tower_chip: 1.10
  opening.safe_first_play: 1.0
  cycle.cheap_cycle: 1.25
posture_multipliers:
  OPENING:     {offense: 0.85, cycle: 1.3}
  DEFEND_LEAD: {offense: 0.70, defend: 1.35}
  PRESS:       {offense: 1.55}
  SPELL_CYCLE: {spell: 2.4, defend: 1.3, offense: 0.6}
counters: null
# Goblin Barrel casts on tower coordinates would benefit from a dedicated spell-wincon placement rule.
```
</details>

---

## 8. Analyze matches

All analytics run on recorded directories, with no API calls:

```bash
# Lane engagements, triage tags (PERCEPTION/EXECUTION/LATENCY/STRATEGY/OK), DuckDB ingest, report
droid-harness analyze --data data/matches-batch --db data/warehouse.duckdb

# Plain-text dashboard: win rate with CI by strategy and game version, rule/posture value, triage mix, loop p99
python notebooks/dashboard.py data/matches-batch

# Scrub any match visually
droid-harness inspect data/matches-batch

# Win-probability model from finished matches (Studio win-chance line, digest swing moments)
droid-harness train-value data --out models/value.json

# Tower-HP label coverage on recorded video: fixed regions vs the label search, per tower
droid-harness hp-audit --data data/matches-exp-v4-v6 --crops data/hp-crops
```

**Value model** ([`learn/value_model.py`](src/droid_harness/learn/value_model.py)). Logistic regression (numpy, no extra dependencies) from a logged state to P(win), over 23 features: time and phase, your elixir and the opponent's range, all six tower HP fractions, crown difference, unit counts and HP on each half, the King Level gap, how much of their deck is known, and hand freshness. It's split by match, so the validation score is honest. On the 56 committed matches (39 train / 17 validation) it reaches validation AUC 0.69 and is under-confident at the low end, so read it as a ranking of moments, not a calibrated probability. It finds the 2–3 seconds where a match turned (`digest` lists them as swings) and is the first piece of the M8 learned ranker. Retrain whenever data grows; the autopilot does it every cycle.

**Tower HP.** Opponent tower labels move with the arena art and the King Tower's bar appears only once it's activated, so fixed regions miss many reads. Each tower's label is now found by a small vertical and horizontal search around its region (~14 ms the first time), the offset is locked and reused across matches, and failed searches back off (2 s doubling to 20 s). A reading that drops and then three agreeing reads go back up is treated as a misread and corrected (`tower_hp_corrected` event). `summary.json` carries `tower_hp_reads` per tower and leaves `tower_hp_diff` empty unless all six towers were read, so no analysis gets a one-sided difference.

Each match folder holds:
- `meta.json`: device, account, `strategy_id`, `strategy_name`, `game_version`, `balance_epoch`, experiment tags, and `account_levels` (your recorded King and card levels).
- `summary.json`: `result`, `my_crowns`, `opp_crowns`, `tower_hp_diff` (empty unless all six towers were read), `tower_hp_reads`, `my_king_level_seen` / `opp_king_level_seen`, `strategy_id`, `status`, `duration_s`, `elixir_leaked`, `loop_p50_ms`, `loop_p99_ms`, `perception_warnings`, `warning_counts`, `opp_deck_seen`, `events` (your plays, spawn plays, fallen towers).
- `decisions.jsonl`: every 5th frame and every play: `t_ms`, `posture`, `chosen_rule`, `chosen_card`, `tile`, `predicted_tile`, `score`, `candidates` (with totals), the `state` snapshot (elixir, opponent range, hand and freshness, units, towers, opponent cycle), and stage latency.
- `frames.jsonl`, `stream.h264`, `stream.pts`: per-frame timings and the raw video with exact device timestamps.

Compact files are ~1.5 MB per match; video is ~60 MB per minute (see retention below).

### Re-run perception on saved video (`replay`)

Test a new detector checkpoint or perception change against past recordings without the phone:

```bash
droid-harness replay --data data/matches-batch \
  --detector models/yolo26n-cr.pt --classes models/classes.txt \
  --out data/replay/run-new.json --baseline data/replay/run-prev.json
```

---

## 9. Set up a new phone

1. **Pin `scrcpy-server-v2.4`** into `vendor/` and create `configs/device_profiles/<phone>.yaml` from [`phone-a.example.yaml`](configs/device_profiles/phone-a.example.yaml) (`serial`, `local_port`, `scrcpy_server_jar`). Each concurrent phone needs its own `local_port`. The client implements the v2.4 wire format (12-byte codec meta, no session packet); v5.0.1 adds a 12-byte session packet after the codec id, so don't point a profile at v5 without updating `scrcpy_bridge.py`.
2. **Stream and latency:**
   ```bash
   droid-harness stream --profile configs/device_profiles/<phone>.yaml --seconds 10 --save-frame data/frame.png
   droid-harness loopback --profile configs/device_profiles/<phone>.yaml --x 30 --y 400
   droid-harness stream --profile configs/device_profiles/<phone>.yaml --duration 1800   # 30-min soak
   ```
   Put the loopback p50 into `exec_latency_ms`. Static screens send no frames; tap or animate while testing.
3. **Card DB** (only when stats change or from scratch):
   ```bash
   git clone --depth 1 https://github.com/RoyaleAPI/cr-api-data ~/src/cr-api-data
   python tools/build_card_db.py ~/src/cr-api-data --out configs/domains/clash_royale/cards.json
   ```
   The source dates from 2023: check costs and deploy times of your deck and common threats, and patch stale entries.
4. **ROIs and tile homography.** Copy [`rois/phone-a.yaml`](configs/domains/clash_royale/rois/phone-a.yaml) and measure fractions `[x0, y0, x1, y1]` for hand slots, `next_card`, `elixir_bar`, `clock`, tower HP fields, `arena` and buttons (`battle`, `ok`, `dismiss`, `trophy_ok`, …). Then fit pixel → 18×32 tiles (max landmark error < 0.5 tiles):
   ```bash
   droid-harness calibrate-tiles --landmarks <landmarks.yaml> --out configs/domains/clash_royale/tiles/<phone>.yaml
   ```
   Use `droid-harness inspect` or the live Studio with *HUD regions* and *Tile grid* on to check the calibration visually.
5. **Templates** from saved frames:
   ```bash
   droid-harness template --frame <battle.png> --layout <rois.yaml> --slot 0 --card knight --state affordable --out configs/domains/clash_royale/templates/<phone>
   droid-harness template --frame <low_elixir.png> --layout <rois.yaml> --slot 0 --card knight --state greyed --out configs/domains/clash_royale/templates/<phone>
   droid-harness tower-template --frame <battle.png> --layout <rois.yaml> --tower all --state alive --out configs/domains/clash_royale/towers/<phone>
   droid-harness screen --frame <home.png> --name HOME --out configs/domains/clash_royale/screens/<phone>
   ```

### Reference: `phone-a` calibration

| Item | Value |
| --- | --- |
| Phone | OnePlus N10 (BE2028), Android 11, 1080×2400, 90 Hz; serial `741be240` |
| Stream | `464×1024` H.264, ~58 fps while the screen moves; decode p50 3.8 ms, p99 15 ms |
| Profile | `configs/device_profiles/phone-a.yaml` (`exec_latency_ms: 40`: OxygenOS hides synthetic touch dots, so loopback can't measure it) |
| ROIs | `rois/phone-a.yaml`: hand ×4, next, elixir (HSV `[135,80,120]–[175,255,255]`), clock (countdown), tower HP fields, arena, buttons `menu`, `training_camp`, `confirm`, `ok`, `battle`, `close`, `dismiss`, `trophy_ok` |
| Screen classifier | The **top 18 %** of the frame only (clock, opponent name): the bottom half changes with the hand |
| Tiles / arena | Homography max landmark error 0.31 tiles; river y 15.0–16.5, bridges x 3.6 / 14.4 |
| Templates | 18 hand templates for the 8-card starter deck (affordable/greyed), 10 tower alive/rubble, digits 0–9 |
| Screens | HOME, MENU, TRAINING_CONFIRM, LOADING, MATCHMAKING, TROPHY_ROAD (incl. the Bone Pit arena-lock modal near 275–299 trophies, dismissed by `trophy_ok`), IN_BATTLE, RESULT_WIN, RESULT_LOSS |
| Menu path (Training Camp) | HOME → hamburger `menu` → `training_camp` → `confirm` → battle |
| Deck | Knight, Archers, Minions, Arrows, Fireball, Giant, Musketeer, Mini P.E.K.K.A (the starter card is **Archers**, not Princess) |

---

## 10. Detector: retrain or add cards

When you unlock cards or promote a frequent enemy out of `other_ground_opp` / `other_air_opp`:

1. Add `<unit>_me` / `<unit>_opp` to [`models/classes.txt`](models/classes.txt) and the alias to [`models/katacr_alias.json`](models/katacr_alias.json) (keep existing indices stable or re-export cleanly).
2. Build datasets and verify zero train/val overlap:
   ```bash
   python tools/fetch_katacr.py --out data/katacr --part2
   python tools/extract_backgrounds.py --data data/matches-batch --out data/backgrounds
   droid-harness synth --backgrounds data/backgrounds data/katacr/backgrounds \
     --sprites data/katacr/segment --bars data/katacr/bars --classes models/classes.txt \
     --synthesize-missing-sides --out data/synth --n 4500 --imgsz 640 384
   droid-harness convert-katacr --part2 data/katacr/_repo/images/part2 --classes models/classes.txt --out data/katacr-real
   droid-harness frames --data data/matches-batch --layout configs/domains/clash_royale/rois/phone-a.yaml \
     --detector models/yolo26n-cr.pt --classes models/classes.txt --out data/own
   droid-harness dataset \
     --train data/synth/data.yaml:train data/katacr-real/data.yaml:train data/own/data.yaml:train \
     --val data/katacr-real/data.yaml:val data/own/data.yaml:val --out data/train.yaml
   ```
3. Train, export, and gate on per-class threat recall:
   ```bash
   python -m droid_harness.learn.train_detector --data data/train.yaml --model models/yolo26n-cr.pt \
     --epochs 30 --batch 64 --cache ram --workers 8 --patience 8 \
     --imgsz 640 --export-imgsz 640 384 --name yolo26n-cr-next --out models/yolo26n-cr-next.onnx
   droid-harness eval-detector --model models/yolo26n-cr-next.onnx --classes models/classes.txt \
     --data data/katacr-real/data.yaml --min-threat-recall 0.70 --min-support 20
   ```
   Invoke `learn.train_detector` directly: the CLI's `train-detector` parser intercepts `--data`. Copy `models/runs/yolo26n-cr-next/weights/best.pt` → `models/yolo26n-cr.pt` (back up the old `.pt`/`.onnx`) once `eval-detector` and `replay` pass.

### Cards the detector has no name for

The 31-class model names only the starter deck and the most common early threats; everything else falls into `other_ground_*` / `other_air_*`. Three tools prepare for the climb:

- **The next class list.** [`models/classes-next.txt`](models/classes-next.txt) keeps the current 31 classes at their indices and adds 55: `_me` classes for the Hog EQ cards you'll play (Skeletons, Cannon, Hog Rider, Ice Golem, the spirits), `_opp` classes for the cards you'll meet on the Arena 1–9 path, and five spell effects (Zap, The Log, Poison, Earthquake, Rocket) so the opponent's elixir and cycle stay right when they cast them. [`katacr_alias-next.json`](models/katacr_alias-next.json) and [`katacr_generic-next.json`](models/katacr_generic-next.json) map KataCR sprites to it. Train the next detector with these three files in place of the current ones (same commands as above).
- **`droid-harness check-classes [--classes models/classes-next.txt]`** lists detector classes with no card in `cards.json` (a spawn the opponent tracker can't cost).
- **`droid-harness eval-unseen --model <model> --part2 data/katacr/_repo/images/part2`** measures what a model does with units it has no name for: how often they're caught by the right catch-all (ground/air, right side), mislabeled as a named card, or missed. Run it before and after a retrain.

An unknown spawn is still costed at a flat estimate. Guessing its cost from unit count and box size was considered and skipped: the play event carries no size signal, and naming the cards is the real fix.

### New cards without hand-capturing templates

`--harvest-templates data/harvest` on `play`/`run`/`experiment` saves hand-slot crops the matcher can't read, labelled by the card that must be there: the "next" card refills the slot you just played from, so if slot *i* stays unreadable after the refill, it holds the card the next slot showed. It never changes what perception reports. After a look through the folder, `droid-harness promote-templates --from data/harvest` turns groups with at least `--min-crops` (3) crops that agree (`--min-agreement` 0.6) into `affordable`/`greyed` templates in the device bundle (`--dry-run` first).

**GPU host (`veritas`, nick@192.168.1.199).** Venv `~/venvs/droid-det` (Python 3.13, `torch 2.7.1+rocm6.3` on the RX 7900 XT, `ultralytics`, `onnx`), symlinked as `.venv` in `~/src/droid-harness`. To fine-tune on new real frames: export with `frames … --detector models/yolo26n-cr.pt --out data/own`, fix the labels, merge with `dataset`, train ~15–20 epochs from `models/yolo26n-cr.pt`, gate with `eval-detector`, then run live with `--detector models/yolo26n-cr.pt --classes models/classes.txt --async-detect` (`TorchDetector` 7.9 ms p50 / 9.8 ms p95 for the 18-class model vs CPU ONNX 26.8 / 53.4 ms).

---

## 11. Architecture and design

### Decisions at a glance

1. **No LLM in the reflex loop.** Capture → perceive → decide → touch runs on deterministic code plus small models, targeting ≤ 60 ms p50 / ≤ 100 ms p99 (measured: under 5 ms p50 for the loop after decode). LLMs only appear in slower loops: an optional non-blocking advisor and an offline coach/agent.
2. **One persistent scrcpy session per phone is the device bus.** Video and touches ride `scrcpy-server`'s sockets through a small Python client with a per-device port. No `screencap`, no `adb shell input` in the hot path.
3. **Perception = fixed-region HUD parsing + one YOLO26 detector + a tracker.** Card matching works on lightness with masks so greyed and selected cards still classify.
4. **Decisions in layers:** Tier 0 reflex invariants → posture → utility-scored rules (→ optional ranker), with lead targeting by deploy time, spell travel and measured action latency.
5. **A strategy is versioned code + YAML + optional weights, content-hashed.** Every decision is logged with the candidates it beat; every lane engagement gets a value and a root-cause tag.
6. **Win rate is the gate, not the guide.** Cheap filters first (replay diff, shadow, engagement value), then a paired SPRT with an outcome-blind inclusion rule and game-version pinning.
7. **Only strategy problems reach the coach.** Perception, execution and latency problems go to their own queues.
8. **The same harness drives any Android app** (accessibility tree first, then templates/OCR, then a GUI model), with successful action sequences cached.

| Goal | Measured by |
| --- | --- |
| Fastest decision-to-action | Per-stage timestamps on every frame; loopback round-trip test |
| Highest win rate | Win rate vs a fixed opponent pool under a pinned game version (§12) |
| Calculated, iterable strategies | Every change is a hypothesis with an ID, a test and a recorded result |
| Reusable for general Android | Core harness + domain plugins; same logging and experiment tables |

Non-goals: reading game memory, sniffing traffic, modifying the APK, emulators, champion-heavy decks, 2v2.

### Runtime

```
 PHONE                                        HOST (Linux, RX 7900 XT)
┌────────────────────────────┐  USB   ┌───────────────────────────────────────────────────────────┐
│ scrcpy-server (app_process) │       │ T1 video:   socket → tee(T4) → H.264 decode → [latest slot]│
│  ├ MediaCodec H.264 encoder ├─video▶│                                          │                 │
│  ├ InputManager injection   │◀ctrl──┤ T3 executor ◀── timed touches ◀── T2 brain:                │
│  └ display power control    │       │   (TCP_NODELAY)        screen → HUD → detector → tracker    │
└────────────────────────────┘       │                        → Tier 0 → posture → rules → placement│
                                      │ T4 recorder: raw .h264 (config + pre-roll) + JSONL, drops if full│
                                      │ T5 liveness: socket/process/ADB ping, stuck-screen detector     │
                                      │ UI publisher (optional): snapshot + JPEG at --ui-fps → Studio   │
                                      │ P2 advisor (optional, separate process): LLM → posture intent   │
                                      └───────────────────────────────────────────────────────────┘
 OFFLINE: matches → engagements → triage → { STRATEGY → digest/coach → candidate → funnel → SPRT }
                                           { PERCEPTION → labeling/calibration queue           }
                                           { EXECUTION / LATENCY → bug & profiling queue        }
```

| Loop | Cadence | What runs | May block the reflex loop? |
| --- | --- | --- | --- |
| L1 reflex | every frame (30–60 Hz), event-driven decisions | capture, HUD, detector, tracker, Tier 0, posture, rules, executor | — |
| L2 intent (optional) | every 3–5 s, async | LLM reads a compact text state, may suggest a posture | Never: stale intent expires (TTL = 2× cadence) |
| L3 learning | between batches | engagements, triage, digest, tuning, model training, SPRT | n/a |

A `threading.Condition` guards the latest-frame slot. If GIL contention ever shows up, the brain moves to its own process with shared memory.

### Latency budget and speed rules

| Stage | Target p50 |
| --- | --- |
| Game frame → encoder output | 10–25 ms (device-dependent, the largest term) |
| USB transfer | < 2 ms |
| Decode (~460×1024 H.264, 1 thread) | 2–5 ms (measured 3.8 ms) |
| HUD parse | < 1 ms |
| Arena detector | 3–8 ms (measured 4.1 ms, async) |
| Tracker + state + Tier 0 + posture + rules | < 2 ms |
| Control message → injected | 1–3 ms |
| Game picks up the touch | 0–17 ms (next frame at 60 Hz) |
| **Frame out → touch injected** | **≈ 35–60 ms** |

1. Drop decoded frames, never packets (H.264 frames reference earlier ones).
2. No bitstream parser: feed whole scrcpy packets, config prepended.
3. Single-threaded, low-delay decoder (frame threading adds frames of delay).
4. `TCP_NODELAY` on the control socket.
5. Static ONNX input shapes plus warm-up.
6. Adaptive detector cadence with tracker extrapolation (`--detect-every`, `--async-detect`).
7. Event-driven decisions plus a heartbeat.
8. Timed actions (`execute_at_ms`): deciding fast and placing at the right moment are separate.
9. Resolution is a sweep judged by small-unit recall and latency, not a guess (720 px long edge is only ~324 px wide on a portrait phone).
10. Phone hygiene: USB, cooling, fixed refresh rate, stay awake, display off for unattended runs.

Once p99 is under ~100 ms the bot is several times faster than human reaction (~200–250 ms); after that, perception accuracy and decision quality bound the win rate.

### Stack

| Layer | Pick | Avoid (why) |
| --- | --- | --- |
| Transport | ADB over USB | Wi-Fi in the hot path (jitter) |
| Capture + input | scrcpy-server v2.4 + own Python client | `screencap` (hundreds of ms), `adb shell input` (a process per tap) |
| Decode | PyAV software, 1 thread, low-delay | A parser in front; frame threading |
| Detector | Ultralytics YOLO26-n → `.pt` (PyTorch-ROCm) or static ONNX | A VLM per frame |
| GPU inference | PyTorch-ROCm `TorchDetector` (MIGraphX EP optional) | ONNX Runtime ROCm EP (removed upstream); TensorRT/NVDEC (NVIDIA-only) |
| Policy | Tier 0 + postures + utility rules + optional ranker | String-eval rules in YAML |
| Logging / warehouse | JSONL + raw `.h264`; DuckDB | Synchronous DB writes in the loop |
| Tuning / statistics | Offline search, Optuna, own SPRT | Eyeballing win rates under ~200 games |
| Studio | stdlib HTTP server + one HTML page (optional pywebview) | Anything that blocks the brain |
| Game data | RoyaleAPI `cr-api-data` | Hard-coded stats |

License: Ultralytics YOLO is AGPL-3.0 (or a paid license). Fine for a private project; switch to an Apache-2.0 detector such as RF-DETR before publishing or hosting.

### Device I/O

Per-device profile (`configs/device_profiles/<phone>.yaml`): serial, unique `local_port`, account, pinned `scrcpy_server_version` + jar, `max_size`, `max_fps`, `video_bit_rate`, encoder, display-power byte, `exec_latency_ms`. Server options: `video_codec=h264`, low-latency codec hints, `audio=false`, `control=true`, `tunnel_forward=true`, frame meta on, `stay_awake=true`. The server refuses a client of another version, so pin one and test the structs ([`scrcpy_protocol.py`](src/droid_harness/core/transport/scrcpy_protocol.py), [`tests/test_scrcpy_protocol.py`](tests/test_scrcpy_protocol.py)).

- **Card play is tap-tap:** tap the slot, then the tile (20 ms hold / 20 ms gap, [`gestures.py`](src/droid_harness/core/actuate/gestures.py)); pre-selection makes placement a single tap at `execute_at`. Drag stays available.
- **Execution check:** the card must leave the hand within ~300 ms (one re-tap at 400 ms if the slot still shows the same card, fresh and affordable) and a matching spawn must appear near the target.
- **Liveness, not frame gaps:** scrcpy sends frames only when the screen changes, so a gap outside battle is normal. Socket errors and server exit reconnect; ADB ping failures reconnect ADB; a gap > 250 ms in battle is a latency anomaly; a stuck screen arms recovery ([`liveness.py`](src/droid_harness/core/runtime/liveness.py), [`recovery.py`](src/droid_harness/domains/clash_royale/recovery.py)). An app watchdog relaunches a dead game.
- **Recorder:** tees every packet before decode; a match file starts with the latest SPS/PPS and the packets since the last key frame, so it decodes from its first byte; `stream.pts` keeps exact device timestamps for replay.
- **Display off** (`turn_display_off`) cools the phone during long runs; on some LCDs only the backlight turns off.

### Perception (Clash Royale)

- **Layout:** normalized ROIs per phone; arena pixels ↔ 18×32 tiles through a homography fitted from ≥ 4 landmarks.
- **HUD every frame (< 1 ms):** hand and next card by masked normalized cross-correlation on lightly blurred luma (`local_znorm`, bottom-center/border mask), affordability from slot saturation (cross-checked against elixir), elixir by HSV fill, clock (1 Hz) and tower HP (2 Hz) by digit templates, King HP seeded from Princess max HP. A slot that fails to read keeps its last card for play detection but is **stale**: the policy never plays it.
- **Detector:** YOLO26-n on the arena crop (letterboxed to 640×384), side-aware classes (`knight_me` / `knight_opp`), `other_ground_*` / `other_air_*` catch-alls for unknown cards, spell effects attributed by your own recent casts (they land on the other half), class-offset NMS.
- **Tracking:** side-keyed association, velocity, extrapolation between detections (capped at 500 ms), flying-unit foot offset; spawn grouping turns tracks into play events, which drive opponent elixir and cycle tracking; swarms upgrade (≥ 4 minions → Minion Horde).
- **Screens:** perceptual hash of the top HUD band + battle evidence from the elixir bar (works in any arena); results are verified from the banners and crowns and must settle before a match ends.
- **QA:** replay regression over recorded matches, `eval-detector` per-class recall, `perception_warnings` per match.

### Game state and opponent model

`GameState` ([`state.py`](src/droid_harness/domains/clash_royale/state.py)) holds time and phase, your elixir (HUD fused with a model), the opponent's elixir **interval**, hand with affordability and freshness, next card, units, towers, crowns, the opponent's seen deck and last four plays, pocket state and posture.

- **Phases:** single elixir for 2 minutes, double in the third; overtime double then triple. About 1 elixir per 2.8 s in single, cap 10.
- **Opponent elixir** = generation − costs of detected plays, as an interval. A full bar collapses it (10 is known); uncertainty comes from unknown costs and missed plays.
- **Opponent cycle:** 8 cards, 4 in hand; a played card returns after four more plays. Once their deck is known, their hand = deck − last four.
- **Card DB:** cost, type, targets, range, speed, deploy time, spell travel speed, crown-tower damage % (level 1 stats). Levels come from [`configs/accounts/`](configs/accounts/) for your cards and from the inferred King Level for theirs, applied only by strategies with `level_aware: true` ([§6](#account-levels)).
- **King Levels:** `my_king_level` / `opp_king_level` in `GameState`, from each side's Princess Tower max HP.

### Decision layer

| Layer | What |
| --- | --- |
| Hard constraints | Legal deploy zone (own half; pocket once a tower falls; spells anywhere), elixir at `execute_at`, one play in flight, minimum spacing, no input outside battle. Never learned, never LLM-editable. |
| Tier 0 reflexes | `t0.lethal_spell` (an affordable spell that finishes a tower), `t0.leak_guard` (elixir above a threshold too long), `t0.emergency_defend` (threat ETA to a low tower). |
| Posture | `OPENING`, `BALANCED`, `DEFEND_LEAD`, `PRESS`, `SPELL_CYCLE` from phase, crowns, time, opponent elixir and spells-to-finish; an advisor may set only allowlisted postures. |
| Utility rules | `defend.lane_threat`, `defend.air_threat`, `offense.win_condition_when_opp_low`, `offense.counter_push`, `offense.giant_back_build`, `offense.opposite_lane_punish`, `spell.value`, `spell.tower_chip`, `cycle.cheap_cycle`, `opening.safe_first_play`, … ([`policy/rules/`](src/droid_harness/domains/clash_royale/policy/rules/)) |
| Ranker (optional) | Bounded residual on top of rule scores. |

**Lead targeting:** a placed card takes effect after action latency + deploy time (+ travel for spells), so targets are advanced along their lane path ([`path.py`](src/droid_harness/domains/clash_royale/path.py)) by that much; predictions beyond ~1.5 s get unreliable. `predicted_tile` is logged for every play.

### Identity, logging and credit

- `strategy_id = sha256(policy source + YAML + counters + model)[:12]`.
- **Engagements:** each lane is segmented into engagements that open when a unit enters contested ground (or a card is placed there) and close when no unit is past the river in that lane for 2 s (cap 25 s). Spells join the open engagement.
- **Value with shared credit:** board value = Σ (card cost ÷ units per card) × `hp_pct`. `value = (net_me − net_opp) + λ·(tower damage dealt − taken)`, with λ = 1 elixir per 100 HP. Each of your decisions gets `value × its elixir ÷ elixir spent`. Value ranks rules and finds problems; it doesn't accept changes.
- **Root-cause triage:**

| Tag | Detection (any) | Routed to |
| --- | --- | --- |
| `PERCEPTION` | Elixir disagreement or tower HP rising in ≥ 2 frames; unreadable hand or affordability contradictions in ≥ 12 frames and ≥ 45 % of the engagement, outside the card-slide window around your plays | Labeling/calibration |
| `EXECUTION` | Card didn't leave the hand, spawn far from target, rejected placement | Executor bugs |
| `LATENCY` | Loop p99 > 100 ms or frame gap > 250 ms during the engagement | Profiling |
| `STRATEGY` | None of the above and value below threshold | Digest / coach |

- **Storage:** raw video ~60 MB/min. `retain` deletes video of clean wins and keeps loss keyframes; compact files are kept forever.

---

## 12. Evaluation and statistics

**Opponent pools.** Training Camp (baseline); the ladder (approved on the throwaway account; trophy-based matchmaking keeps moving the pool, so only paired comparisons are meaningful); a Practice gauntlet of 10–20 archetypes from your battle log (keep opponents where the champion wins 15–85 %); two-phone self-play with strategy ↔ (phone, account) swapped every game so device and card-level bias cancel.

**Inclusion rule (fixed before results):** count every completed match. Exclude only infrastructure aborts and compare abort rates between arms (an arm that aborts more is failing). Never exclude by triage tags. Every match records `game_version` and `balance_epoch`; an experiment that spans an update is void.

**Sample size.** ~3–5 min per match plus menus → ~10–15 matches/hour per phone. 95 % Wilson interval at 50 %: ±13 pts at 50 games, ±10 at 100, ±7 at 200, ±5 at 400.

| True candidate edge (H1) | ≈ Elo | Avg decisive comparisons to decide (H0 = 50 %, α = β = 0.05) | Hours at 12/h, one phone |
| --- | --- | --- | --- |
| 55 % | +35 | ~530 | ~44 |
| 58 % | +56 | ~205 | ~17 |
| 60 % | +70 | ~130 | ~11 |
| 65 % | +108 | ~57 | ~5 |

In the paired test only **discordant pairs** count, so the games needed are this number divided by the share of discordant pairs (~40–50 % at a 25–50 % win rate): several hundred games for a +8-point edge. Filter candidates offline and test changes worth having.

**Dense test** (`--test dense`). Each pair contributes the difference of `dense_match_score` (result, crowns, tower-HP difference when all six towers were read, speed), so win-win and loss-loss pairs count too. An always-valid mixture SPRT (mSPRT; Johari et al.) on the mean difference x̄ over n pairs, with plug-in variance σ² and mixture scale τ (`--dense-tau`, the margin difference you care about, default 0.5):

  Λₙ = √(σ² / (σ² + nτ²)) · exp(n²τ²x̄² / (2σ²(σ² + nτ²)))

It decides when Λₙ ≥ 1/α, in the direction of x̄, and can be checked after every pair without inflating the error rate. It measures margin, not win rate directly: a candidate that wins bigger and loses smaller is "better" even before it wins more often. Confirm an accepted dense result on the ladder with `pairs` when the decision matters.

**SPRT** ([`sprt.py`](src/droid_harness/flywheel/eval/sprt.py)): `LLR = wins·ln(p1/p0) + losses·ln((1−p1)/(1−p0))`, accept at `ln((1−β)/α)`, reject at `ln(β/(1−α))`. Non-regression changes use bounds like [0.47, 0.50]. `pairs` (default) tests whether the candidate wins discordant pairs > 50 %; draws are the middle score; aborted slots are replayed. Stage A futility uses the dense pair margin.

**Optuna** (`flywheel/tuning/optuna_study.py`): search over `params`, `tier0` and posture multipliers; one trial = a block of ~8 matches; objective = crown + tower-HP differential; ≤ 30 trials per study. The best trial is a candidate that still goes through the funnel.

---

## 13. LLM roles, learned policy, general Android

**Offline coach / agent.** After a batch: digest → **one** proposal (title, hypothesis, evidence, change type, patch, target metric, expected decision-change %) → replay-diff gate → paired SPRT → ledger. Guardrails: patches touch only `configs/strategies/**`, counters and `policy/rules/**`; a rule change ships with a unit test; a replay diff much larger than claimed is rejected; one hypothesis per proposal; the ledger blocks near-duplicates of failed ideas unless the context changed.

**In-match advisor (M9, off by default)** ([`advisor.py`](src/droid_harness/core/runtime/advisor.py)): every 3–5 s an OpenAI-compatible model reads ~300 tokens of state and may set an allowlisted posture; 1.5 s timeout, ignored after 2× cadence, separate process, never blocks. Enabled only as a strategy variant that wins an SPRT; if it wins, distill it into a small classifier. `droid-harness advise --dry-run` replays a recorded match offline.

**Learned policy (M8).** Data: the bot's logs (`label_from_video` keeps a chosen play only when a `my_play` event proves it happened) plus, later, auto-labeled human demos: `droid-harness watch` records your own matches with your plays as `my_play` events next to what the bot would have done, but the labeler doesn't turn those into demonstrations yet. Model: small network over units, hand, elixir, clock, posture and opponent cycle; numpy MLP behavior cloning → filtered BC → offline RL; ONNX, ~1–2 ms. Deployment: shadow → residual score inside the utility engine → standalone, each step SPRT-gated; the model checksum is part of `strategy_id`.

**General Android (M10).** Same core and tables (task success instead of wins). Observation ladder, cheapest first: the accessibility tree via DroidRun Portal (uiautomator2 behind the same interface; some Android skins restrict it), then templates / small detectors / OCR (RapidOCR) for what the tree can't see (game engines, Flutter canvases, custom WebViews), then a GUI-agent model with a Set-of-Mark overlay. Successful action sequences are cached by `(package, screen signature, goal)` with pre/post-condition checks. Settled = no new frame for `quiet_ms` (scrcpy only sends frames on change) or N near-identical frames, and an unchanged tree hash ([`settle.py`](src/droid_harness/domains/android_os/settle.py)).

**Risks.**

| Risk | Mitigation |
| --- | --- |
| Account ban | Throwaway accounts |
| Game updates, balance patches | `game_version` + `balance_epoch`; void tests across updates; re-check templates |
| scrcpy protocol changes | Pinned version; protocol tests |
| Thermals, burn-in | Display off, cooling, watch `loop_p99_ms` drift |
| Perception errors steering strategy | Triage before the coach; replay regression |
| Biased win-rate tests | Outcome-blind inclusion, abort-rate comparison, paired blocks |
| Too few matches | Offline filters, dense secondary metrics, meaningful SPRT bounds |
| Disk growth | `retain`, `--prune-win-video` |
| Coach churn | Path allowlist, replay-diff gate, one hypothesis per proposal, ledger dedupe |

### Repository layout

```
configs/device_profiles/            per-phone profiles (phone-a.yaml tracked)
configs/domains/clash_royale/       rois/, tiles/, arena/, templates/, screens/, towers/, digits/, cards.json (local)
configs/strategies/                 strategy YAMLs + counters/*.csv (immutable once played), CHAMPION pointer,
                                    legacy_ids.yaml, auto/ (autopilot candidates)
configs/accounts/                   card and King levels per account
reports/                            autopilot reports (gitignored; published to the autopilot/reports branch)
src/droid_harness/core/             transport (scrcpy, decoder, frame slot), actuate, runtime (brain, liveness,
                                    watchdog, advisor), telemetry (recorder, latency, loopback, soak)
src/droid_harness/domains/clash_royale/   perception/ (incl. template harvesting), policy/ (hashed into strategy IDs),
                                    state, opponent model, account levels, session, menus, recovery, replay, hp_audit
src/droid_harness/domains/android_os/     a11y, settle, Set-of-Mark, skills cache
src/droid_harness/flywheel/         engagements, triage, report, digest, retention, shadow, eval/ (runner, SPRT,
                                    mSPRT, replay diff, live paired runs), tuning/ (targeted, offline, Optuna),
                                    autopilot, coach/, ledger, warehouse
src/droid_harness/learn/            synth, KataCR conversion, datasets, detector/ranker training, labeling,
                                    value model, unseen-card eval
src/droid_harness/ui/               the Studio (live view + replay inspector)
src/droid_harness/sim/              synthetic scene + fake scrcpy server (CI, `sim`)
ledger/                             experiments.jsonl + STRATEGY_LEDGER.md
notebooks/dashboard.py              plain-text / marimo dashboard
tools/                              card DB builder, KataCR fetch, background extraction, bootstrap bot,
                                    systemd/ (autopilot service)
data/                               recordings (gitignored; compact match files are committed selectively)
```

---

## 14. Design history

The design merged a first plan, an outside plan ("Aegis-Droid") and a critique of the first plan. Kept from the first plan: the whole backbone (stack, loops, wire-level details, data strategy with KataCR synthetic sprites, rule engine, logging, evaluation funnel, coach guardrails, macro cache, explicit account scope). Adopted from Aegis: tap-tap + pre-selection, HP-bar strip for ownership and `hp_pct`, an "avoid" column, the AGPL license note, Set-of-Mark/OCR fallbacks for general apps, core + domain plugins with per-phone profiles, display-off runs, deploy-delay lead targeting, discrete postures with reflex invariants, and root-cause triage. Rejected: an NVIDIA-only stack, an LLM in the tactical tier, a 720 px default, an "atomic index" buffer in Python, sparse-keyframe-only recording, and an all-at-once learning stack.

Flaws found and fixed in the first plan: a frame-gap watchdog (scrcpy sends no frames on static screens) → liveness checks; settle detection that waited for frames → quiescence; a hard-coded port → per-device ports; per-unit kill credit (not observable) → engagement value with shared credit; a coach fed perception errors → triage first; BGR templates failing on greyed cards → masked lightness matching; no deploy/travel model → lead targeting; recordings without SPS/PPS → config + pre-roll; excluding "unhealthy" matches from tests (biased) → outcome-blind inclusion; self-play device/account bias → seat swapping; no game-version tracking → `game_version` + `balance_epoch`; unbudgeted storage → retention.

Build notes since: the elixir cap *collapses* the opponent interval (a full bar is known); offense stands down while an unanswered threat is inbound (`attack_blocked_by_threat_eta_s`); a lane stays live while a unit is still advancing toward the river; triage ignores the ~300 ms card-slide window around confirmed plays (it had tagged 89 % of engagements `PERCEPTION`).

### Where the old docs went

| Old file | Now |
| --- | --- |
| `docs/plan.md` | [§11](#11-architecture-and-design)–[§13](#13-llm-roles-learned-policy-general-android) (code skeletons dropped: the real code is in `src/`) |
| `docs/review.md` | [§14](#14-design-history) |
| `docs/pro-strategy-research.md` | [§7](#7-strategy-playbook) (comments in `policy/strategy.py` and `ladder-vision-v6.yaml` still cite it; those files are hashed into strategy IDs, so they weren't edited) |
| `docs/refinement-loop-plan.md` | [§5](#5-improvement-loop); its measured bottlenecks are in the run log in `todo.md` |
| `docs/phone-a-bringup.md` | [§9](#9-set-up-a-new-phone) (reference table) and the run log in `todo.md` |
| `docs/overnight-plan.md` | [§15](#15-troubleshooting-and-operational-limits) (guardrails) and the run log in `todo.md` |
| `docs/run-2026-11-27-recovery-v4-v6.md` | The run log in `todo.md` |

---

## 15. Troubleshooting and operational limits

- **Phone sleeps or drops ADB:** enable *Stay awake* while charging; check `adb devices`. If `scrcpy-server` is busy after a crash, wait ~3 s for the old socket on `local_port` or kill stray `adb` / `droid-harness` processes. Only one scrcpy client per phone/port.
- **Stuck on a result screen or popup:** `droid-harness recover` (or wait 8–15 s for the live loop to do the same). If the calibrated `ok` keeps missing, re-capture the result screen or recalibrate `buttons:`; recovery finds the real button visually either way.
- **Deck changed in the game:** capture `affordable` and `greyed` templates into a clean directory, update `deck:` in a **new** strategy file, re-run `replay` / `pytest`. Stale templates show up as `hand_unreadable` warnings and dashed slots in the Studio.
- **Disk:** raw video is ~60 MB/min (2–4 GB per 10 matches). Use `--prune-win-video` and `droid-harness retain`; keep `meta.json`, `summary.json`, `decisions.jsonl` (~1.5 MB/match).
- **AMD GPU:** ONNX Runtime removed its ROCm EP upstream. Pass `--detector models/yolo26n-cr.pt` for PyTorch-ROCm; `.onnx` runs on MIGraphX if built, else CPU (~27 ms p50).
- **Unattended runs:** use the systemd user unit in [`tools/systemd/`](tools/systemd/droid-autopilot.service) (or `tmux`); keep the phone plugged in with sleep disabled; test `turn_display_off: true` and thermals before a long run; bump `--balance-epoch` after a game update. `touch data/autopilot/STOP` ends the autopilot after its current cycle.
- **Autopilot didn't notify or publish:** both are best-effort and never stop a cycle; the outcome is in the local `reports/<stamp>/report.json` (`notified`, `published` with git's error) and in the service log. Publishing needs a push-capable remote on the bot machine (`git push` must work non-interactively).
- **Studio:** binds to localhost; use an SSH tunnel rather than `--ui-host 0.0.0.0` on untrusted networks. A live page that says "Waiting for the first frame" on a menu is normal: static screens send no frames.

---

## 16. Sources

- scrcpy: https://github.com/Genymobile/scrcpy (frames only on change: `doc/develop.md`, `doc/video.md`; display power: `doc/device.md`)
- KataCR: https://arxiv.org/abs/2504.04783 · https://github.com/wty-yy/katacr · https://github.com/wty-yy/Clash-Royale-Detection-Dataset · https://github.com/wty-yy/Clash-Royale-Replay-Dataset
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- ONNX Runtime ROCm EP removal: https://github.com/microsoft/onnxruntime/pull/25181 · MIGraphX EP: https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html
- DroidRun Portal: https://github.com/droidrun/droidrun-portal · GUI-Owl-1.5: https://arxiv.org/abs/2602.16855 · MobileWorld: https://arxiv.org/abs/2512.19432
- Supercell: https://support.clashroyale.com/hc/en-us/articles/49484901806747-Don-t-Be-a-Cheat · https://supercell.com/en/safe-and-fair-play/ · Elixir: https://support.clashroyale.com/hc/en-us/articles/49484936933403-Elixir
- Card data: https://developer.clashroyale.com · https://github.com/RoyaleAPI/cr-api-data · Meta tracking: https://royaleapi.com/decks/meta
- SPRT: https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html · https://dannyhammer.github.io/engine-testing-guide/sprt.html
