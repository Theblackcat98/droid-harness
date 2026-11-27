# Run report — 2026-11-27: stuck-screen recovery + v4 vs v6 paired batch

Device `phone-a` (OnePlus N10 `741be240`), account `theblackcat99`, game
`160635013`, epoch `2026-11`, strategy IDs `639c09fe0c3a` (v4) and
`ab2d9564c216` (v6).

## 1. Stuck-screen recovery (code)

The live phone was found on a ladder loss screen that added a **Play Again** button and
shifted **OK** right. The calibrated `ok` ROI (`[0.370, 0.815, 0.630, 0.880]`, centre
`(232,868)` at 464×1024) landed in the 23 px gap between the two buttons, so every
result-screen tap missed. The same screen also hashes `UNKNOWN` (new nameplate + reward
layout), so the UNKNOWN flow cycled other wrong taps and the farm stopped.

`domains/clash_royale/recovery.py` adds an escalation ladder driven by screen age:

1. calibrated buttons for the screen (normal flow, again),
2. wide saturated buttons found **visually** in the lower frame, rightmost first
   (this is what finds the real OK),
3. BACK through scrcpy,
4. force-stop + relaunch after the plan is exhausted (or 60 s of liveness findings).

Once an episode starts it keeps clearing each new screen until HOME. `home_visible()`
requires the actual Battle button, so a Trophy Road scroll that hashes as HOME is not
accepted. The same ladder is exposed as `droid-harness recover` (with `--dry-run`,
`--stuck-after`, `--battle-stuck-after`).

Two extra stuck states were found and fixed during the run:

- **Purple post-match chest** — the chest is 100 % magenta in the elixir ROI, so
  `battle_evidence` read it as a full elixir bar and the session stayed `IN_BATTLE`.
  A 30 s **battle no-progress** guard aborts when the clock (`match_t_s`), a fresh hand
  slot and elixir all stop changing, records `aborted:battle_no_progress`, and hands the
  screen to recovery.
- **Batch-end settle** — `BotRunner` now walks back to HOME after a finished batch using
  only the recovery ladder. Menu decisions are suspended and `_set_screen` cannot
  start/end a match, so a transient HOME animation that hashes `IN_BATTLE` cannot create
  a phantom match.

Tests: **209 passed, 1 skipped**; `ruff check src/ tests/` clean.

## 2. Live validation

| Run | Data | Result |
| --- | --- | --- |
| v4 1 match | `data/matches-fresh-v4b/971f64eb82c2` | **win 3–2**, settle `RESULT_WIN → OK → HOME` in ~5 s |
| v4 1 match | `data/matches-fresh-v4c/c11f767dcfde` | **loss 0–3**, settle `RESULT_LOSS → OK → HOME` in ~5 s |
| v4 3-match smoke | `data/matches-fresh-v4` | 2 wins (3–1, 3–1), one 660 s chest stall before the fix (`2b3487f2bfd1`) |

`droid-harness recover` cleared the live stuck loss screen to HOME in ~10 s and rejected
a Trophy Road reward frame before accepting the real HOME.

## 3. Paired experiment EXP-0001 (v4 vs v6)

Config: ladder, `--test pairs`, `p0=0.5`, `p1=0.58`, `max_games=16`,
`--futility-pairs 6`, data `data/matches-exp-v4-v6`.

| Arm | Record |
| --- | --- |
| champion v4 | 2W–6L |
| candidate v6 | 2W–5L (+1 abort `left_battle:MATCHMAKING`) |

Discordant pairs **2–2**, 3 tied, dense margin **−0.539**, LLR **−0.05** → SPRT
**continue**; the run hit its 16-game cap. Ledger: `EXP-0001 · INCONCLUSIVE · 16
matches · candidate 2–5–0`.

**v6 mechanism check:** v4 played `offense.giant_back_build` at **1.1–1.9 s** in six
matches (`b5f66a`, `49b06e`, `366789`, `d0d019`, `94b696`, `9b68ac`); v6 played **none**
(all six are champion matches). v6's earliest cards are `opening.safe_first_play` at
4.4–4.7 s and a counter-push Giant at 5.2–5.3 s. The opening guard works; eight pairs
are simply too few to decide the win rate.

## 4. Speed-1 findings

- `tune-offline --champion v6 --data data/matches-exp-v4-v6 --trials 300`: best trial
  (`defense_reserve 1.62`, `elixir_weight 0.20`) has `within_guardrail: false`
  (18 bad flips vs 3 good broken) → **no promotable v7**.
- Dashboard triage: **OK 86.0 %, PERCEPTION 14.0 %**, loop p99 **6.1 ms**. The digest's
  55.9 % PERCEPTION share is dominated by end-of-match frames: a decoded frame at
  t=176.2 s shows the clock at `0:00` with all four hand slots intentionally blank. The
  mid-match hand is readable (3–4 slots fresh); the raw `hand_unreadable` count is not a
  mid-match perception failure.
- Failure clusters: `opening_throw` 11, `wincon_starved` 12, `over_defend_bleed` 15
  (83–98 % of the match below 5 elixir), `spell_inefficiency` 10.
- Both arms at ~25–28 % in this pool, versus v4's 90 % (9–1) earlier: the trophy level
  moved up (≈476) and the current opponent pool is much harder. The paired design
  cancels that drift for the v4/v6 comparison, but the absolute rate is not comparable
  with the old batch.

## 5. Data committed

Compact per-match files (`meta.json`, `summary.json`, `decisions.jsonl`) only; videos and
`frames.jsonl` stay local.

- `data/matches-fresh-v4/` — 2 wins + the 660 s chest-stall abort (`2b3487f2bfd1`).
- `data/matches-fresh-v4b/` — settled win `971f64eb82c2`.
- `data/matches-fresh-v4c/` — 0–3 loss `c11f767dcfde`.
- `data/matches-exp-v4-v6/` — all 16 EXP-0001 matches.
- `ledger/experiments.jsonl`, `ledger/STRATEGY_LEDGER.md` — EXP-0001 record.

Excluded: the orphan phantom `ba1703f05ad9` (no summary; created on the chest before the
fix) and the 5.2 s phantom `227876ae59b2` (HOME animation hash; fixed by the settle
guard).

## 6. Next step

The paired test is undecided. The documented next move is another bounded Stage-B block
(v6 vs v4) rather than stacking a v7 on an undecided candidate. If live games are
paused, the recovery work is complete and the data above is the current baseline.
