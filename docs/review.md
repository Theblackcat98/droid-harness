# Review — Plan v1 vs "Aegis-Droid", and an evaluation of the attached critique

**About the inputs.** The uploaded Aegis-Droid plan ends at line 218, partway through its §5.1 (the hand-card classifier row). I evaluated its §1–§5.1 directly. For its later sections (macro postures, root-cause triage, deploy-delay lead targeting, its labeling approach, its System 3 tooling), I relied on how the critique describes them, and say so where it matters.

Naming below: **v1** = my first `plan.md`. **Aegis** = the attached plan. **Critique** = the attached review, which calls Aegis "Plan A" and v1 "Plan B".

---

## Part 1 — v1 vs Aegis-Droid

### Where Aegis is stronger (adopted into v2)

| # | Aegis idea | Why it's good | v2 section |
|---|---|---|---|
| A1 | **Tap-tap + pre-selection** spelled out as the default actuation, with drag as the slow path | Concrete and testable. v1 only said "test both" | §5.3 |
| A2 | **HP-bar strip above each box** for ownership *and* `hp_pct` | Gives a per-unit health fraction, which makes "elixir value on board" computable (see C3) | §6.4 |
| A3 | **"Explicitly avoid (and why)" column** in the stack table | Cheap guardrail against well-known traps | §4 |
| A4 | **License note:** Ultralytics is AGPL-3.0; RF-DETR is Apache-2.0 | v1 missed it. Irrelevant for a private project, important if you ever publish or host it | §4 |
| A5 | **General-app fallbacks:** Set-of-Mark overlay before a VLM call; RapidOCR (ONNX) for text; the warning that UI-tree dumps are blind to game engines, Flutter canvases and custom WebViews | All sound. v1's observation ladder lacked the SoM and OCR steps | §13 |
| A6 | **Core harness + domain plugins**, and per-phone `device_profiles/` | Cleaner than v1's split once two phones and general apps exist | §14 |
| A7 | **Screen power off during long runs** | Real thermal and OLED burn-in benefit; scrcpy supports it natively | §5.2 |
| A8 | *(per critique)* **Deploy-delay lead targeting** | v1 had `placement.timing()` but no motion model | §8.5 |
| A9 | *(per critique)* **Discrete macro postures** + deterministic "reflex invariants" (e.g., lethal spell, elixir cap) | Clash Royale has sharp regime switches; postures express them without an LLM | §8.2–8.3 |
| A10 | *(per critique)* **Root-cause triage** before strategy changes | Stops the optimizer from "fixing" sensor bugs with strategy | §9.5 |

### Where Aegis is weaker (rejected or modified)

| # | Issue | Why |
|---|---|---|
| B1 | **NVIDIA-only stack** (TensorRT, NVDEC, `h264_cuvid`, SGLang FP8) | None of it runs on an RX 7900 XT. The ROCm/MIGraphX path is the real work on your machine |
| B2 | **LLM in the tactical tier** ("Mode B", 85–235 ms, local 8B or a cloud low-latency LLM API) | On one GPU it competes with YOLO and spikes p99. LLMs are weak at precise tile placement. Cloud calls add network jitter. And nothing shows it beats a utility ranker. v2 allows an LLM only as a tested strategy variant in the slow loop |
| B3 | **`max_size=720`, 4 Mbps** as the default, with a ">60%" latency claim | On a portrait phone, a 720 px long edge is only ~324 px wide. Upscaling the arena to 640 for YOLO adds no information, so small units (skeletons, bats, spirits) lose recall. The 60% figure isn't sourced. v2 treats resolution as a sweep judged by detector recall *and* latency |
| B4 | **Latency figures presented as facts** (e.g., capture 25–35 ms, UI tree 15–40 ms) | Plausible, but they're targets. v2 keeps v1's rule: measure with the loopback test |
| B5 | **"Double-buffered SharedMemory slot toggled via an atomic index"** | Python has no atomics. With threads, a condition variable is simpler and safe. SharedMemory only matters if the brain moves to its own process |
| B6 | **Sparse keyframes only** for replays | You can't replay full matches through new perception models or auto-label your own play without full video. v2 keeps the raw stream (but fixes v1's storage blind spot, D8) |
| B7 | *(per critique)* System 3 scope (DSPy, GEPA, Reflexion, bandits, LoRA SFT/DPO/GRPO all at once) | Too much for one person; each tool needs the evaluation harness to exist first |
| B8 | *(per critique)* Thin statistics (no match-throughput math, "50–100 games") | At ~12 matches/hour, this is the constraint that shapes the whole design |
| B9 | Missing in the visible part: KataCR's datasets, the Practice Mode gauntlet, the menu FSM, the account-risk scope | All present in v1 |

### Where v1 is stronger (kept)
- Hardware fit: ROCm/MIGraphX vs PyTorch-ROCm, a CPU decode default with "measure VA-API".
- Wire-level craft: whole-packet decode (no parser), single-thread low-delay decoder, `TCP_NODELAY`, `execute_at_ms` timed actions, a software loopback RTT test.
- Data strategy: KataCR-style synthetic sprites + their open datasets + active learning; closed-set 8-card matching.
- Evaluation: throughput math, SPRT, and the Replay Diff → Shadow → Screening → SPRT funnel with a Strategy Ledger.
- LLM placement: out of L1. Coach proposals as structured diffs with a path allowlist.
- General Android: the compiled-skill macro cache and settle detection.
- Explicit account-risk scope (throwaway account; Training Camp, Practice, own friendlies only).

### Where v1 is weaker — self-audit
The critique found the first eight. I found the last six while re-checking.

| # | v1 flaw | Severity | Fix in v2 |
|---|---|---|---|
| D1 | Watchdog treats "no frame for 500 ms" as failure — but scrcpy only produces frames when the screen changes | **Bug** | Liveness = socket/process/ping; a frame gap outside battle means "static" (§5.5) |
| D2 | Settle detection waits for "N frames differing < ε", which never arrive on a static screen; v1's timeout makes it always slow rather than hung | Bug | Quiescence = no new frame for T ms, *or* low diff for N frames, *and* stable UI tree (§13) |
| D3 | Hard-coded port 27183 breaks two-phone runs | Bug | Port and scid per device (§5.2) |
| D4 | `play_value` uses "kills by this action's units" — not observable from pixels | Design flaw | Engagement-level value with shared credit (§9.4) |
| D5 | Its own loss-story example feeds a HUD misread (`elixir read 8.9`) to the coach as a strategy problem | Design flaw | Root-cause triage before the coach (§9.5) |
| D6 | BGR template matching breaks when cards are greyed out (unaffordable) or lifted (selected) | Bug | Masked lightness matching or a tiny CNN, trained on all card states (§6.2) |
| D7 | No deploy-time or spell-travel model in placement | Gap | Lead targeting (§8.5) |
| D8 | Recorder may start a file without SPS/PPS | Bug | Tee packets before decode; write the latest config first (§5.2) |
| D9 | **Excluding "health failure" matches from strategy stats** can bias the SPRT, because data-quality problems can correlate with outcomes | Statistical flaw | Gate on all completed matches; exclude only pre-defined infrastructure aborts and compare abort rates per arm (§10.2) |
| D10 | **Two-phone self-play doesn't cancel device/account bias** (different phones, different card levels) | Statistical flaw | Swap strategy ↔ (phone, account) every game (§10.1) |
| D11 | **No game-version / balance-patch tracking** — a mid-test update silently mixes two games | Gap | `game_version` + `balance_epoch` on every match; tests void across updates (§10.2) |
| D12 | **Storage:** raw H.264 at 8 Mbps is ~60 MB/min, so ~0.2–0.3 GB per match and tens of GB per day | Gap | Retention policy plus nightly transcode (§9.7) |
| D13 | "Latest frame wins" wasn't stated precisely: H.264 packets can't be dropped before decode (later frames reference earlier ones) | Clarity | Drop only decoded frames; the decoder must keep up with the stream (§3) |
| D14 | No license note | Gap | From A4 (§4) |

### Keep / change summary
- **Keep from v1:** the whole backbone (stack, three loops, wire craft, perception data strategy, rule engine, logging schema, evaluation funnel, coach guardrails, macro cache).
- **Bring in from Aegis:** A1–A10.
- **Fix in v1:** D1–D14.
- **Reject from Aegis:** B1, B2, B3-as-default, B5, B6-as-only-recording, B7.

---

## Part 2 — Evaluating the critique

**Verdict:** accurate and genuinely useful. Its headline conclusion (use v1 as the backbone, merge five Aegis ideas, fix four bugs) is right. I checked its biggest technical claims and they hold. Its weaknesses are a formula that double-counts credit, a vague watchdog suggestion, some inflated certainty, and blind spots on statistics and operations.

### Claim-by-claim

| Critique claim | Verdict | Notes |
|---|---|---|
| scrcpy emits no frames on a static screen, so v1's watchdog misfires | **Correct — verified** | scrcpy's developer docs say new frames are produced only when the surface changes. The `KEY_REPEAT_PREVIOUS_FRAME_AFTER` flag only handles the first and last frame; it doesn't keep a stream going |
| v1's settle detection "will hang" | Mostly correct | v1 had a timeout, so it would always hit the timeout (slow) rather than hang. Same fix either way |
| Hard-coded port breaks two-phone self-play | Correct | |
| Per-unit kill attribution isn't computable from vision | Correct | |
| Coach contamination, shown by v1's own example | **Correct, sharp catch** | |
| Greyed-out cards break BGR templates | Correct | Greying is also a free affordability signal; v2 uses it as a cross-check |
| Missing deploy delay / spell travel | Correct, but "~1.0 s" needs nuance | Most troops deploy in 1 s, but it varies by card (e.g., a 2025 balance change cut Spear Goblin's deploy delay to 0.5 s), and multi-unit cards deploy their units 0.1 s apart. Read it from the card DB |
| `SET_SCREEN_POWER_MODE` off, "type=10, mode=0" | Correct, with a version caveat | Type 10 is the display-power message (renamed `SET_DISPLAY_POWER` in newer protocol docs). The payload changed between versions; `0` means off in both. Caveats: on some LCDs only the backlight goes off, and in-display fingerprint readers stop working |
| Tee SPS/PPS + packets to the recorder before decode | Correct | |
| Test DroidRun Portal on your phone's Android skin; keep uiautomator2 behind the same interface | Reasonable | v1 already listed uiautomator2; adding an explicit Phase 0 check is right |
| Lane-window formula `Δ(NetBoardElixir) + λ·ΔTowerHP` per decision over t → t+6 s | **Right direction, flawed as written** | Overlapping windows count the same swing for every decision inside them (double counting), and opponent plays land in the window too. v2 scores whole lane *engagements* and splits credit across the decisions in them. "Board elixir" also needs per-unit HP, which is only as good as HP-bar reading |
| Watchdog fix: "enable scrcpy's capture-side frame repetition only during battle if supported" | Vague | I know of no such per-phase option. The robust fix is liveness checks (socket, server process, periodic ADB ping), plus "screen unchanged too long" detection from content hashes |
| `SPELL_CYCLE` when the tower is "within 2× Fireball + Log range" | Imprecise | It means *damage*, and spells deal reduced damage to crown towers. Use crown-tower damage values from the card DB |
| "Single-thread CPU decode beats VA-API" (attributed to v1) | Overstated | v1 said measure both; a GPU→RAM copy-back is a likely cost, not a proven loss |
| "Skipping the parser saves an entire 16.6 ms frame"; "TCP_NODELAY avoids 40 ms Nagle stalls" | Directionally right, stated as exact | Fine as rules of thumb; the loopback test is the arbiter |
| Plan A "manual labeling of 1.5–3k frames", "string-eval YAML rules" | Can't verify | Those parts of Aegis weren't in the upload |

### What the critique missed
1. **SPRT inclusion bias** (D9): v1's exclusion of "health failure" matches can bias acceptance tests. The critique praised v1's statistics without catching this.
2. **Device/account bias in two-phone self-play** (D10).
3. **Balance patches and game updates** (D11): the biggest real-world confounder for any win-rate experiment longer than a day.
4. **Storage cost** of v1's raw-stream recording (D12) — the one real advantage of Aegis's sparse keyframes.
5. **Aegis's own good ideas it didn't credit:** the license note (A4), SoM/RapidOCR fallbacks (A5), the HP-bar strip (A2).
6. **Aegis's resolution default** (B3) and its "atomic index" buffer (B5) went unchallenged.

**Score:** about 85–90% of its specific claims hold. Every one of its merge recommendations survives into v2 — the lane-window score and the watchdog fix in modified form.

---

## Part 3 — What changed in plan v2

| Area | Change | Source |
|---|---|---|
| Device I/O | Per-device port and scid; display-power control; recorder tee with config packets; liveness watchdog instead of frame-gap watchdog | Critique + Aegis + D1/D3/D8 |
| Capture | Resolution sweep (720/1024/1280) judged by recall *and* latency; explicit "drop only decoded frames" rule | B3, D13 |
| Perception | Masked lightness card matching / tiny CNN over all card states; HP-bar strip for ownership and `hp_pct` | Critique + Aegis |
| Decision | Tier 0 reflex invariants; deterministic postures with YAML multipliers; lead targeting from per-card deploy time + spell travel + measured action latency | Aegis + critique |
| Credit | Engagement-level value with shared credit (replaces per-unit kills and per-decision windows) | D4, refined critique |
| Triage | PERCEPTION / EXECUTION / LATENCY / STRATEGY tags; only STRATEGY reaches the coach | Aegis + critique, extended |
| Statistics | Outcome-blind inclusion rule; strategy↔(phone, account) swapping; `game_version` / `balance_epoch` | New (D9–D11) |
| Storage | Retention + nightly transcode; keyframe export for labeling | New (D12) + Aegis |
| General Android | Settle = quiescence; SoM overlay; OCR fallback; OEM check for Portal | Critique + Aegis |
| Stack | License note; "avoid" column | Aegis |
