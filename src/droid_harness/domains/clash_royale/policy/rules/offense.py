"""Offense rules. The Giant-beatdown v3 experiments are opt-in params, one per experiment:

* E3 ``counterpush_tank_ahead``: on a counter-push, the tank goes *ahead* of the surviving
  defenders (and gets a bonus); support still goes behind.
* E4 ``counter_gate``: scale win-condition pushes by whether the opponent's known counters
  (``wc_counters``) may be in hand (``counter_in_hand_mult``) or are confirmed out of cycle
  with their deck fully seen (``counter_out_mult``).
* E5 ``defense_reserve`` / ``giant_back_build``: penalize offense that leaves less than the
  reserve while they may have elixir; build a Giant from the back when the board is quiet
  and elixir is near full.
"""

from __future__ import annotations

from ..candidate import Candidate
from . import RuleContext


def _scale(score: float, f: float) -> float:
    """Multiply only the positive part: a penalty must never make a bad play less bad."""
    return score * f if score > 0 else score


def counter_factor(s, p, card_key: str, ctx: RuleContext | None = None, *,
                   is_back_build: bool = False) -> tuple[float, str]:
    """E4 + cost/cycle awareness: scale pushes by counter availability & affordability."""
    if not p.counter_gate or card_key not in p.win_conditions:
        return 1.0, ""
    seen = set(p.wc_counters) & set(s.opp_deck_seen)
    if not seen:
        return 1.0, ""
    opp_hi = s.opp_elixir[1]
    cost_aware = p.counter_unaffordable_punish_mult != 1.0 or p.cycle_punish_partial_deck
    in_hand_ready = []
    in_hand_unaffordable = []
    out_of_cycle = []
    for c in sorted(seen):
        until = s.opp_cards_until_back(c)
        if until == 0:
            if cost_aware and ctx is not None and c in ctx.cards and opp_hi < ctx.cards.cost(c) - 0.5:
                in_hand_unaffordable.append(c)
            else:
                in_hand_ready.append(c)
        else:
            out_of_cycle.append(c)
    if in_hand_ready:
        mult = (
            p.back_build_counter_in_hand_mult
            if is_back_build and p.back_build_counter_in_hand_mult is not None
            else p.counter_in_hand_mult
        )
        return mult, f"; their {'/'.join(in_hand_ready)} may be in hand"
    deck_full = len(s.opp_deck_seen) >= 8
    if p.use_min_cycle_elixir and ctx is not None and out_of_cycle and not in_hand_ready:
        min_needed = min(
            s.opp_min_cycle_elixir_to_return(c, ctx.cards) + (ctx.cards.cost(c) if c in ctx.cards else 3.0)
            for c in out_of_cycle
        )
        if opp_hi + 1.5 < min_needed:
            mult = max(p.counter_out_mult, 1.50) if deck_full else (1.0 + 0.75 * (p.counter_out_mult - 1.0))
            return mult, f"; {'/'.join(out_of_cycle)} cycle-locked ({min_needed:.0f} elix to return)"
    if deck_full or (p.cycle_punish_partial_deck and len(s.opp_deck_seen) >= 5):
        if out_of_cycle and not in_hand_unaffordable:
            mult = p.counter_out_mult if deck_full else (1.0 + 0.5 * (p.counter_out_mult - 1.0))
            return mult, f"; their {'/'.join(out_of_cycle)} out of cycle"
        if in_hand_unaffordable:
            return p.counter_unaffordable_punish_mult, f"; {'/'.join(in_hand_unaffordable)} unaffordable"
    return 1.0, ""


def reserve_factor(s, p, cost: int) -> tuple[float, str]:
    """E5: keep enough elixir to defend, unless they're low too."""
    if p.defense_reserve <= 0:
        return 1.0, ""
    left = s.my_elixir - cost
    if left >= p.defense_reserve or s.opp_elixir[1] <= p.press_opp_elixir_max:
        return 1.0, ""
    return p.reserve_penalty, f"; leaves {left:.1f} < reserve {p.defense_reserve:.1f}"


def _weaker_lane(s, ctx: RuleContext) -> str:
    """Pick attack lane balancing enemy tower HP and defensive fortification (avoids dead/walled lanes)."""
    alive = {lane: s.towers.get(f"opp_{lane}", 1) > 0 for lane in ("left", "right")}
    if alive["left"] != alive["right"]:
        return "left" if alive["left"] else "right"
    hp = {lane: s.towers.get(f"opp_{lane}", 0) or 2000 for lane in ("left", "right")}
    max_hp = max(hp["left"], hp["right"], 1500)
    load = {"left": 0.0, "right": 0.0}
    for u in s.enemy_units():
        lane = ctx.arena.lane_of(u.tile[0])
        uc = ctx.cards.card_for_class(u.kind)
        mult = 1.5 if (uc is not None and uc.kind == "building") else 1.0
        load[lane] += ctx.cards.unit_value(u.kind) * u.hp_pct * mult
    return min(("left", "right"), key=lambda lane: (hp[lane] / max_hp + 0.12 * load[lane], load[lane]))


def unanswered_threat(s, p, ctx: RuleContext) -> bool:
    """An enemy push is coming and nothing of ours is in that lane to meet it."""
    for th in s.threats(ctx.arena, ctx.cards, max_tiles_to_tower=p.defend_trigger_tiles):
        if th.eta_to_tower_s > p.attack_blocked_by_threat_eta_s:
            continue
        defenders = [u for u in s.my_units() if ctx.arena.lane_of(u.tile[0]) == th.lane]
        if not defenders:
            return True
    return False


class WinConditionWhenOppLow:
    """Play a win condition at the bridge when you have elixir and they likely don't —
    unless an unanswered push is on its way (defend first)."""

    id = "offense.win_condition_when_opp_low"

    def propose(self, s, p, ctx: RuleContext, intent):
        if s.my_elixir < p.min_elixir_to_attack or unanswered_threat(s, p, ctx):
            return
        opp_hi = s.opp_elixir[1]
        if opp_hi > p.max_opp_elixir_to_bridge and (s.my_elixir - opp_hi) < p.min_elixir_advantage_to_bridge:
            return
        aggression = intent.aggression if intent is not None else 0.5
        for slot, key in s.affordable_hand(ctx.cards):
            if key not in p.win_conditions:
                continue
            lane = intent.lane if intent is not None and intent.lane else _weaker_lane(s, ctx)
            tile = ctx.placement.snap(tuple(p.placement.bridge_tiles[lane]), ctx.cards[key], s)
            score = (1.0 - opp_hi / 10) * (0.5 + aggression) - p.elixir_weight * ctx.cards.cost(key) / 20
            fc, why_c = counter_factor(s, p, key, ctx)
            fr, why_r = reserve_factor(s, p, ctx.cards.cost(key))
            yield Candidate(self.id, key, slot, tile, score=_scale(score, fc * fr),
                            reason=f"opp elixir ≤ {opp_hi:.1f}, lane {lane}{why_c}{why_r}")


class CounterPush:
    """Support surviving defenders that are heading back over the bridge."""

    id = "offense.counter_push"

    def propose(self, s, p, ctx: RuleContext, intent):
        if unanswered_threat(s, p, ctx):
            return  # defend an unanswered opposite-lane push before reinforcing a counter-push
        groups: dict[str, float] = {"left": 0.0, "right": 0.0}
        lead: dict[str, tuple[float, float]] = {}
        has_wc_in_lane: dict[str, bool] = {"left": False, "right": False}
        for u in s.my_units():
            if u.vel[1] <= 0.2:  # not advancing
                continue
            lane = ctx.arena.lane_of(u.tile[0])
            groups[lane] += ctx.cards.unit_value(u.kind) * u.hp_pct
            uc = ctx.cards.card_for_class(u.kind)
            if uc is not None and uc.key in p.win_conditions:
                has_wc_in_lane[lane] = True
            if lane not in lead or u.tile[1] > lead[lane][1]:
                lead[lane] = u.tile
        wc_in_hand = [k for k in s.hand if k and k in p.win_conditions and k in ctx.cards]
        wc_cost = min((ctx.cards.cost(k) for k in wc_in_hand), default=0)
        for lane, value in groups.items():
            if value < p.counterpush_min_value or lane not in lead:
                continue
            other = "right" if lane == "left" else "left"
            dead_lane_pen = 0.5 if (s.towers.get(f"opp_{lane}", 1) == 0 and s.towers.get(f"opp_{other}", 1) > 0) else 1.0
            enemies = sum(1 for u in s.enemy_units() if ctx.arena.lane_of(u.tile[0]) == lane)
            lx, ly = lead[lane]
            for slot, key in s.affordable_hand(ctx.cards):
                card = ctx.cards[key]
                if card.is_spell or card.kind == "building":
                    continue
                is_wc = key in p.win_conditions
                # Opening discipline applies to win-condition counter-pushes too: two back-plays
                # before 0:30 (a safe first play + a tank behind it) leave ~1 elixir and the next
                # enemy push walks over an empty arena. Same barrier the giant_back_build rule uses.
                in_opening = s.posture == "OPENING" or (s.t_ms < p.opening_ms and not s.enemy_units())
                if (is_wc and not p.giant_back_build_allow_in_opening and in_opening
                        and s.my_elixir < p.giant_x1_min_elixir):
                    continue
                if not is_wc:
                    if p.counterpush_min_elixir > 0 and s.my_elixir < p.counterpush_min_elixir:
                        continue
                    # Bank elixir for Giant in hand instead of dumping 3e support at the bridge.
                    if (p.counterpush_bank_for_wc and wc_cost > 0
                            and not has_wc_in_lane[lane]
                            and s.my_elixir < wc_cost + 1.5):
                        continue
                score = p.counter_push_bonus * value / 10 - 0.1 * enemies - p.elixir_weight * card.cost / 20
                tank = p.counterpush_tank_ahead and is_wc
                if tank:  # E3: the tank leads, so the survivors shoot from behind it
                    tile = ctx.placement.snap((lx, ly + 1.0), card, s)
                    score += p.counterpush_tank_bonus
                    fc, why = counter_factor(s, p, key, ctx)
                    fr, why_r = reserve_factor(s, p, card.cost)
                    score = _scale(score, fc * fr * dead_lane_pen)
                    why = f"{why}{why_r}"
                else:
                    tile = ctx.placement.snap((lx, ly - 2.0), card, s)
                    tile = ctx.placement.avoid_clump(tile, card, s, radius=p.avoid_clump_radius)
                    fr, why = reserve_factor(s, p, card.cost)
                    score = _scale(score, fr * dead_lane_pen)
                    if p.spell_aware and key == "minions":
                        if s.opp_can_play_now("arrows", ctx.cards):
                            score = _scale(score, 0.65)
                            why = f"{why}; arrows in hand"
                        elif "arrows" in s.opp_deck_seen and s.opp_cards_until_back("arrows") >= 2:
                            score = _scale(score, 1.30)
                            why = f"{why}; arrows out of cycle"
                yield Candidate(self.id, key, slot, tile, score=score,
                                reason=f"{'tank ahead of' if tank else 'support'} {value:.1f} elixir push {lane}{why}")


class GiantBackBuild:
    """E5: quiet board and elixir near full: start a Giant from the back of the weaker lane,
    so support can be added while it walks. Off unless ``giant_back_build: true``."""

    id = "offense.giant_back_build"

    def propose(self, s, p, ctx: RuleContext, intent):
        if not p.giant_back_build or s.my_elixir < p.giant_safe_build_elixir:
            return
        if not p.giant_back_build_allow_in_opening and s.posture == "OPENING":
            return
        if s.phase == "x1" and s.my_elixir < p.giant_x1_min_elixir:
            return
        block_tiles = p.defend_trigger_tiles if p.giant_build_block_tiles is None else p.giant_build_block_tiles
        if s.threats(ctx.arena, ctx.cards, max_tiles_to_tower=block_tiles):
            return  # defend first
        lane = intent.lane if intent is not None and intent.lane else _weaker_lane(s, ctx)
        for slot, key in s.affordable_hand(ctx.cards):
            card = ctx.cards[key]
            if key not in p.win_conditions or card.is_spell or card.kind == "building":
                continue
            tile = ctx.placement.snap(tuple(p.placement.back_tiles[lane]), card, s)
            fc, why_c = counter_factor(s, p, key, ctx, is_back_build=True)
            fr, why_r = reserve_factor(s, p, card.cost)
            yield Candidate(self.id, key, slot, tile, score=_scale(p.giant_back_build_score, fc * fr),
                            reason=f"back-build {key} {lane} at {s.my_elixir:.1f} elixir{why_c}{why_r}")


DEFAULT_PUNISH_FAST = ("hog_rider", "battle_ram", "ram_rider", "mini_pekka", "prince", "bandit", "wall_breakers")


class OppositeLanePunish:
    """Pressure the opposite bridge immediately when opponent invests >=5 elixir in the back."""

    id = "offense.opposite_lane_punish"

    def propose(self, s, p, ctx: RuleContext, intent):
        if not p.opposite_lane_punish or s.my_elixir < p.punish_min_my_elixir or unanswered_threat(s, p, ctx):
            return
        heavy_back = []
        for u in s.enemy_units():
            if u.tile[1] < p.punish_enemy_back_y:
                continue
            c = ctx.cards.card_for_class(u.kind)
            cost = c.cost if c is not None else 0
            if cost >= p.punish_min_enemy_cost:
                heavy_back.append((cost, u))
        if not heavy_back:
            return
        inv_cost, inv_u = max(heavy_back, key=lambda pair: pair[0])
        opp_lane = "right" if ctx.arena.lane_of(inv_u.tile[0]) == "left" else "left"
        allowed = set(p.punish_cards) if p.punish_cards else (set(p.win_conditions) | set(DEFAULT_PUNISH_FAST))
        for slot, key in s.affordable_hand(ctx.cards):
            card = ctx.cards[key]
            if key not in allowed or card.is_spell or card.kind == "building":
                continue
            tile = ctx.placement.snap(tuple(p.placement.bridge_tiles[opp_lane]), card, s)
            fc, why_c = counter_factor(s, p, key, ctx)
            fr, why_r = reserve_factor(s, p, card.cost)
            yield Candidate(
                self.id, key, slot, tile, score=_scale(p.punish_score, fc * fr),
                reason=f"punish {inv_u.kind} ({inv_cost}e back) opposite {opp_lane}{why_c}{why_r}",
            )
