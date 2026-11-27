# Pro Clash Royale Strategy Research & Engine Translation

Web research synthesized for `droid-harness` (top ladder macro/micro meta + Arena 1–5 card progression) and translated into concrete engine capabilities and valid strategy YAML configurations ([`configs/strategies/`](../configs/strategies/)).

---

## 1. Research queries & sources

Queries executed during this synthesis (`search_web` on 2026-11-26):
1. `"Clash Royale" top ladder meta decks 2025 2026 highest win rate site:royaleapi.com` — RoyaleAPI Ultimate Champion / Grand Challenge archetype tracking ([RoyaleAPI Meta](https://royaleapi.com/decks/meta), [RoyaleAPI Popular](https://royaleapi.com/decks/popular?time=7d&sort=rating&size=20&mode=ged&global_tourney=0)).
2. `"Clash Royale" pro strategy "card counting" "elixir counting" "cycle tracking" beatdown control macro guide` — Gap-based elixir counting, 4-card cycle rotation, single vs. double vs. triple elixir pacing, opposite-lane punishing.
3. `"Clash Royale" "Giant" beatdown pro strategy guide "opposite lane" "counterpush" "elixir advantage" "spell cycle"` — Giant slow push behind King Tower, anti-punish holding, surviving-defender snowballing, dual-value spell cycling.
4. `"Clash Royale" pro tips "tower HP as a resource" OR "give spell value" OR "outcycle" OR "kiting" placement tiles guide` — Accepting minor chip to bank elixir, anti-spell 3–4 tile spacing, center pulls (`2–3` tiles from river), cross-lane melee kiting, shadow alignment for air units, King Tower activation.
5. `"Clash Royale" "archetypes" beatdown cycle control bait bridge spam siege win conditions matchup macro guide` — The 6 core archetypes and matchup rock-paper-scissors triangle.
6. `"Clash Royale" "Arena 2" OR "Arena 3" OR "Arena 4" OR "starter deck" Giant Musketeer Mini Pekka Fireball Arrows strategy tips` & `"Clash Royale" "Arena 2" "Arena 3" "Arena 4" "Arena 5" cards unlock list wiki OR royaleapi` — Arena 2–5 unlock sequence and starter-deck macro roles.

---

## 2. Competitive meta landscape & card progression

### Top-ladder archetype picture

High win-rate competitive archetypes center on flexible Elixir/Cycle exploitation rather than linear single-lane spamming:
- **Control / Bridge Spam Hybrids** (e.g., *Evo Battle Ram + Giant Skeleton Control*, *Royal Recruits + Wall Breakers Split-Lane Spam*): win by forcing asymmetric elixir splits across both lanes and converting clean defense into immediate bridge counter-pressure.
- **Anchored Control Beatdown** (e.g., *Royal Giant / Monk Control*): pairs one ranged/tank anchor with high-value defensive specialists (`Fisherman`, `Hunter`), spells (`Fireball` + `The Log`), and cheap cycle (`Electro Spirit`, swarms) so opponent buildings and tank killers get neutralized cleanly.
- **Low-Cost Cycle & Spell Bait** (e.g., *Hog Rider / Earthquake*, *Goblin Barrel / Log Bait*): runs `2.6–3.0` average cost to out-rotate the opponent's single counter building or small spell using the 4-card queue.

### The 6 enduring archetypes & matchup triangle

| Archetype | Core win conditions | How it wins | Natural edge vs. | Vulnerable to |
| --- | --- | --- | --- | --- |
| **Beatdown** | `giant`, `golem`, `lava_hound`, `electro_giant` | Survive `x1` passively/positively, back-plant tank at `+2..+4` or in `x2`, stack DPS behind tank | **Control**, **Siege** (overwhelms single-target DPS & buildings) | Fast **Cycle** / **Bridge Spam** opposite-lane punishes; hard tank killers (`mini_pekka`, `pekka`, `inferno_tower`) |
| **Cycle** | `hog_rider`, `wall_breakers`, `miner` | `2.6–3.0` avg cost; out-cycle opponent's counter building/spell and kit-defend positively | **Beatdown** (punishes back-row tanks opposite lane, rotates faster) | Multiple overlapping counters / heavy splash control |
| **Control** | `graveyard`, `miner` + `poison`, `royal_giant` + `fisherman` | Defend with positive elixir trades (`+2..+4`), turn full-HP defenders into counter-pushes | **Bridge Spam**, aggro decks | Heavy multi-wave **Beatdown** in double/triple elixir |
| **Spell Bait** | `goblin_barrel`, `goblin_drill`, `skeleton_barrel` | Force small spell (`arrows`/`the_log`) on `princess`/`goblin_gang`, punish during 3-card cooldown | Slow / single-spell decks | Double small spell or cheap cycle fast enough to match bait rotation |
| **Bridge Spam** | `battle_ram`, `ram_rider`, `bandit`, `prince` | Instant opposite-lane bridge punishment whenever opponent invests `>= 5` in the back | Heavy back-plant **Beatdown**, **Siege** | Cheap cycle & disciplined elixir reserves |
| **Siege** | `x_bow`, `mortar` | Protect offensive building in your territory (`y ≈ 13.5`); spell-cycle remaining HP if blocked | **Control** without big tank answer | Heavy **Beatdown** tanks soaking building fire |

### Starter deck → Arena 2–9 Directed Upgrade Path to **2.6 Hog EQ** (`theblackcat99`)

Because ladder optimization climbs trophies into higher-level cards, **gold and wildcards must only go into cards that survive into the target Hog EQ cycle deck** (`hog_rider`, `earthquake` [Fireball until Arena 9], `the_log` / `barbarian_barrel` [Arrows until unlocked], `musketeer` / `firecracker`, `cannon`, `knight` / `ice_golem`, `skeletons`, `ice_spirit` / `fire_spirit`):

1. **Immediate (Arena 1 -> King Lv2):** Perform 1 cheap upgrade (`8/9` XP on `theblackcat99`) to reach **King Level 2** (+112 Princess HP, +168 King HP, +8% tower DPS) and unlock Bone Pit. Upgrade **Musketeer**, **Knight**, and **Fireball** freely — all three carry directly into Hog Cycle/Control through Arena 9.
2. **Arena 2 (Bone Pit — 400+ trophies):** Unlock & slot **`skeletons`** (1-elixir cycle/distraction staple of Hog EQ). Optional bridge card: `valkyrie` or `bomber` if swarm pressure spikes, without over-investing gold in `giant`/`minions`/`archers`.
3. **Arena 3 (Barbarian Bowl — 800+ trophies):** Unlock & slot **`cannon`** (core 3-elixir defensive building of 2.6 Hog / Hog EQ) + `barbarian_barrel` (small-spell upgrade over `arrows`).
4. **Arena 4 (Spell Valley — 1400+ trophies):** Unlock & slot **`fire_spirit`** / **`electro_spirit`** (1-elixir cycle slot).
5. **Arena 5 (Builder's Workshop — 1600+ trophies):** Unlock **`hog_rider`** — pivot strategy from `ladder-vision-v6` (Giant Beatdown) to **Blueprint 2 (`pro-hog-cycle-blueprint`)** (`[hog_rider, knight, cannon, musketeer, skeletons, fire_spirit, fireball, barbarian_barrel]`, avg cost `2.8`!).
6. **Arena 8–9 (Frozen Peak / Jungle Arena):** Swap `ice_spirit` / `ice_golem` (Arena 8) and **`earthquake`** (Arena 9) to complete full **Hog EQ**!

> **Operational prerequisite for swapping cards:** Before putting a newly unlocked card into `deck:`, capture both `affordable` and `greyed` templates via `droid-harness template`, add counter rows to `counters/*.csv`, and add `<unit>_me` / `<unit>_opp` labels to `models/classes.txt` if not already in the 31-class detector.

---

## 3. The 8 core pro mechanics (real-time decision principles)

1. **Track the Elixir Gap & Phase Pacing:**
   - Generation rates: `1` every `2.8 s` in `x1`, `1.4 s` in `x2`, `~0.9 s` in `ot_x3`; leaking past `10` is wasted resource, but **proactively spending down to `< 3` elixir while opponent sits at `7.5–10`** loses instantly to opposite-lane rushes (proven directly by Batch 2's 3 ~95 s `0–3` losses when `v5` opened with `t=1.1s` back-Giants at `7.2–7.5` elixir).
   - **Rule of thumb:** Never plant a back-row Giant in `x1` unless `my_elixir >= 8.5` (keeping `>= 3.5` defense reserve) OR after an already-completed positive defense generating `+2..+4` gap. Go aggressive in `x2` / `ot_x2`.
2. **4-Card Cycle & Minimum Cycle Cost to Return:**
   - Opponent decks have 8 cards (4 in hand, 4 in queue); a played card requires playing **4 other cards** (`3` queued transitions after play) to return.
   - Tracking *whether* a counter (`mini_pekka`, `goblin_cage`, `tombstone`, `cannon`, `barbarians`, `prince`) is in hand vs. out of cycle is half the equation — pros also estimate the **minimum elixir required to rotate that counter back** (`CycleTracker.min_cycle_elixir_to_return`). If `opp_elixir + regen < min_cycle_elixir_to_return`, our win condition has a guaranteed unobstructed window!
3. **Opposite-Lane Punish vs. Same-Lane Tank-Build:**
   - When an opponent spends `>= 5` elixir behind their King Tower (`y >= 25`) in `x1`, either drop a fast/punish card in the **opposite lane** if our elixir is even/ahead (forcing them to split support away from their tank) OR build on defense only if holding hard counter. Conversely, always keep **1 anti-punish answer** (`knight`, `minions`, or `mini_pekka`) affordable when back-building our own Giant.
4. **Counter-Push Snowballing:**
   - Never throw defensive troops away just because a threat crossed the bridge. Defend on your side with `musketeer`/`archers`/`mini_pekka`/`knight`, let both Princess Tower and defenders kill the push at a `+2..+4` elixir profit, then **plant Giant in front of survivors** before they cross the river.
5. **Tower HP as a Resource:**
   - On a healthy Princess Tower (`> 60%` HP), accepting `200–350` chip damage from solo low-value pokes (`spear_goblin`, `skeleton`, lone low-HP `archer`) lets elixir climb from `4 → 8` instead of burning `3` elixir every 5 seconds and starving win conditions. Only respect urgent or high-DPS threats (`giant`, `mini_pekka`, `prince`, `valkyrie`, `balloon`, swarms), or when tower HP enters opponent spell range.
6. **Exact Tile Geometry on our `18×32` Arena Grid ([`arena.py`](../src/droid_harness/domains/clash_royale/arena.py)):**
   - **Center Building Pull:** Placing a defensive building at `x ∈ [7.5, 10.5], y ≈ 10.5` (`~4.5` tiles below river `y=15` — `placement.building_pull` in [`strategy.py`](../src/droid_harness/domains/clash_royale/policy/strategy.py)) drags building-targeters (`giant`, `hog_rider`, `balloon`) into **both** Princess towers' range simultaneously.
   - **Cross-Lane Melee Kite:** Dropping a cheap mini-tank (`knight`, `ice_golem`) in the *opposite* lane near `(7.5–10.5, 11–13)` pulls `mini_pekka`, `valkyrie`, `prince`, or `pekka` across the center so both Princess towers shred it during the walk.
   - **Shadow Alignment for Air:** Detector/pathing targets units by shadow/foot tile (`arena.y`), matching pro air-kiting.
   - **Ranged Standoff:** Placing ranged DPS (`musketeer`, `archers`) `3–4` tiles behind the contact tile prevents melee targets from cleaving onto them cleanly.
7. **Spell Geometry & Value Discipline:**
   - Never cast `fireball` (4 elixir) on a lone tank when no tower/support hits exist; wait until troops overlap or hit tower + backline DPS (`musketeer`, `wizard`, `witch`, `barbarians`, `archers`).
   - Spread backline defenders `>= 3.5` tiles from our own Princess towers when facing opponent `fireball`/`poison` so we don't donate free dual troop + tower damage.
8. **Out-Cycling & Overtime Spell-Cycle Transition:**
   - In overtime (`ot_x2`, `ot_x3`) with `<= 2–3` spells left to finish a low Princess tower, stop risking bridge Giants into stacked defense: lock down our own side and rotate `fireball` + `arrows` onto the weak tower (`SPELL_CYCLE`).

---

## 4. Capability audit — configurable today vs. high-ROI code upgrades

| Pro mechanic | Configurable in YAML today? | Exact engine mechanism today / missing code piece |
| --- | --- | --- |
| **Elixir-gated bridge pushes & Giant back-build** | **Yes** | `max_opp_elixir_to_bridge`, `min_elixir_advantage_to_bridge`, `giant_safe_build_elixir`, `giant_build_block_tiles`, `defense_reserve`, `reserve_penalty` |
| **Counter availability + affordability tracking** | **Yes** | `counter_gate`, `wc_counters`, `counter_in_hand_mult`, `counter_out_mult`, `counter_unaffordable_punish_mult`, `cycle_punish_partial_deck`, `back_build_counter_in_hand_mult` |
| **Kill-aware spells & Minions vs Arrows bait/avoidance** | **Yes** | `spell_aware`, `kill_aware_spells`, `spell_value_threshold`, `tier0.spell_tower_damage` |
| **Protect tank killers in quiet cycling** | **Yes** | `save_counters_for_in_hand_threats` in [`rules/cycle.py`](../src/droid_harness/domains/clash_royale/policy/rules/cycle.py) |
| **Counter-push tank-ahead snowballing** | **Yes** | `counterpush_tank_ahead`, `counterpush_tank_bonus`, `counterpush_min_value` |
| **Overtime spell cycle (`<= N` casts)** | **Yes** | `posture.py::SPELL_CYCLE` + `spell_cycle_max_casts` + `tier0.lethal_spell` |
| **1. `OPENING` / `x1` guard in `GiantBackBuild`** | **Not yet (code)** | [`offense.py::GiantBackBuild`](../src/droid_harness/domains/clash_royale/policy/rules/offense.py#L166) should block back-builds when `s.posture == "OPENING"` or `s.phase == "x1" and s.my_elixir < 8.5` |
| **2. Wire `min_cycle_elixir_to_return` into scoring** | **Not yet (code)** | Already computed in [`CycleTracker.min_cycle_elixir_to_return`](../src/droid_harness/domains/clash_royale/opponent_model.py#L84); wire into [`offense.py::counter_factor`](../src/droid_harness/domains/clash_royale/policy/rules/offense.py#L24) so counters `>= 2` cards away with high cycle cost get a full punish boost |
| **3. Tower-HP-as-resource chip filter** | **Not yet (code)** | Add `min_defend_threat_value` / healthy-tower chip threshold to [`rules/defend.py`](../src/droid_harness/domains/clash_royale/policy/rules/defend.py) when tower HP `> 60%` max |
| **4. `OppositeLanePunish` rule** | **Not yet (code)** | New rule in [`rules/offense.py`](../src/droid_harness/domains/clash_royale/policy/rules/offense.py): detect enemy `cost >= 5` spawned at `y >= 25` and punish opposite bridge when fast/pressure card is in hand |
| **5. Cross-lane kite & anti-spell tower offset** | **Not yet (code)** | Enhance [`placement.py::defensive`](../src/droid_harness/domains/clash_royale/policy/placement.py#L56) for melee kiting across `x=9` and offsetting ranged backline spawns `>= 3` tiles away from our Princess tower when opponent holds `fireball`/`poison` |
| **6. Predictive pre-hover `arrows` on bridge Giant push** | **Not yet (code)** | When Giant approaches `y >= 16` and `s.opp_can_play_now("skeleton_army", ...)` or `"minion_horde"`, preselect `arrows` centered ahead of the Giant |

---

## 5. Translated YAML strategy blueprints (100% valid today)

Every YAML block below uses **only** keys currently accepted by `StrategySpec` and `DEFAULT_PARAMS` in [`policy/strategy.py`](../src/droid_harness/domains/clash_royale/policy/strategy.py). Card keys match [`configs/domains/clash_royale/cards.json`](../configs/domains/clash_royale/cards.json). Comments note optional parameters to activate once the Section 4 code extensions land.

### Blueprint 1 — Pro Starter Giant Beatdown (`ladder-vision-v6`)

Checked in at [`configs/strategies/ladder-vision-v6.yaml`](../configs/strategies/ladder-vision-v6.yaml) (+ [`counters/ladder-vision-v6.csv`](../configs/strategies/counters/ladder-vision-v6.csv)).
Fixes `v5`'s `t=1.1s` `7.2`-elixir Giant opening blunder by restoring `defense_reserve: 2.5` and raising `giant_safe_build_elixir: 8.6` (so planting Giant leaves `>= 3.6` elixir ready for opposite-lane punishes), while keeping narrowed hard-counter `wc_counters`, moderate multipliers (`0.55` / `1.45`), and `giant_build_block_tiles: 9.5`:

```yaml
strategy: ladder-vision-v6
parent: ladder-vision-v4
deck: [knight, archers, minions, arrows, fireball, giant, musketeer, mini_pekka]
win_conditions: [giant]
params:
  act_threshold: 0.08
  elixir_weight: 0.4
  defend_trigger_tiles: 12.0
  defend_horizon_s: 8.5
  min_elixir_to_attack: 5.5
  press_opp_elixir_max: 4.0
  max_opp_elixir_to_bridge: 5.8
  min_elixir_advantage_to_bridge: 2.0
  spell_value_threshold: 3.5
  spell_cycle_max_casts: 3
  opening_ms: 15000
  min_action_gap_ms: 350
  counter_push_bonus: 0.6
  counterpush_min_value: 2.2
  counterpush_tank_ahead: true
  counterpush_tank_bonus: 0.35
  giant_back_build: true
  giant_safe_build_elixir: 8.6
  giant_back_build_score: 0.65
  giant_build_block_tiles: 9.5
  attack_blocked_by_threat_eta_s: 6.0
  counter_gate: true
  wc_counters: [mini_pekka, goblin_cage, tombstone, cannon, minion_horde, skeleton_army, barbarians, prince, pekka, inferno_tower, hunter]
  counter_in_hand_mult: 0.55
  counter_out_mult: 1.45
  counter_unaffordable_punish_mult: 1.30
  cycle_punish_partial_deck: true
  back_build_counter_in_hand_mult: 0.85
  defense_reserve: 2.5
  reserve_penalty: 0.50
  spell_aware: true
  kill_aware_spells: true
  save_counters_for_in_hand_threats: true
tier0:
  lethal_margin_hp: 20
  spell_tower_damage:
    fireball: 75
    arrows: 30
  leak_threshold: 8.8
  leak_ms: 800
  leak_card_preference: [knight, archers, minions, musketeer, mini_pekka]
  emergency_eta_s: 1.8
  emergency_hp_frac: 0.35
weights:
  defend.lane_threat: 1.0
  defend.air_threat: 1.0
  offense.win_condition_when_opp_low: 1.0
  offense.counter_push: 0.85
  offense.giant_back_build: 1.0
  spell.value: 0.85
  spell.tower_chip: 0.6
  opening.safe_first_play: 1.0
  cycle.cheap_cycle: 1.0
posture_multipliers:
  OPENING:     {offense: 0.5, cycle: 1.2}
  DEFEND_LEAD: {offense: 0.5, defend: 1.3}
  PRESS:       {offense: 1.4}
  SPELL_CYCLE: {spell: 2.0, offense: 0.6}
counters: counters/ladder-vision-v6.csv
learned_ranker: null
intent_advisor: false
advisor_may_set: [DEFEND_LEAD, PRESS, SPELL_CYCLE]
# Future extension knobs once Section 4 code lands:
# giant_back_build_allow_in_opening: false
# use_min_cycle_elixir: true
```

---

### Blueprint 2 — Pro Fast Cycle (2.6 Hog Cycle Archetype)

Designed to out-rotate building counters (`cannon`, `tesla`, `tombstone`, `inferno_tower`) and maintain high bridge tempo (`giant_back_build: false`, aggressive `counter_out_mult: 1.60`, light `defense_reserve: 1.5`, fast cheap-cycle priority). Requires calibrated card templates for these 8 cards + matching counter CSV:

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
counters: null  # add counters CSV covering hog/tank/swarm defense
```

---

### Blueprint 3 — Pro Control / Counter-Push Archetype

Emphasizes positive defensive trades first (`defend_trigger_tiles: 10.5` patient patience near tower), high counter-push reward (`counter_push_bonus: 0.85`), and strict anti-punish `defense_reserve: 3.0`:

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

---

### Blueprint 4 — Pro Dual-Lane / Bridge Spam Archetype

Capitalizes immediately on opponent elixir deficits (`press_opp_elixir_max: 5.0`, `min_elixir_advantage_to_bridge: 1.0`, aggressive `offense.win_condition_when_opp_low` & `PRESS` multiplier) while holding `pekka`/`musketeer` on heavy defense:

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
# Future extension once Section 4 code lands: enable OppositeLanePunish rule when opp plays >=5 elixir at y >= 25
```

---

### Blueprint 5 — Pro Spell Bait (Log / Arrows Bait Archetype)

Targets the 4-card small-spell rotation (`the_log`, `arrows`, `zap`, `barbarian_barrel`): suppresses `goblin_barrel` pushes when their small spell is in hand & affordable (`counter_in_hand_mult: 0.35`) and spikes immediately when it leaves cycle (`counter_out_mult: 1.70`) or cannot be afforded (`counter_unaffordable_punish_mult: 1.45`), with overtime `rocket` finishing:

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
# Note: Goblin Barrel casts on opp tower coordinates benefit from a dedicated spell-wincon placement rule
```
