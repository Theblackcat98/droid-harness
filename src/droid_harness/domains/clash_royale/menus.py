"""Menu flow: which button to tap on which screen to keep matches coming.

Screen names come from ``ScreenClassifier``; button regions come from the phone's layout
file. Flows are config so a new mode (Practice Mode, friendly battle) is a YAML edit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ...core.actuate import gestures
from ...core.actuate.gestures import TouchEvent
from .perception.grid import Roi

DEFAULT_FLOWS: dict[str, dict[str, list[str]]] = {
    "training_camp": {
        "HOME": ["menu"],
        "MENU": ["training_camp"],
        "TRAINING_CONFIRM": ["confirm"],
        "RESULT_WIN": ["ok"],
        "RESULT_LOSS": ["ok"],
        "RESULT_DRAW": ["ok"],
        "POPUP": ["close"],
        "UNKNOWN": ["close"],
    },
    "friendly": {  # self-play: one phone hosts, the other accepts (see eval/selfplay.py)
        "HOME": ["social"],
        "SOCIAL": ["friendly_battle"],
        "INVITE": ["accept"],
        "RESULT_WIN": ["ok"],
        "RESULT_LOSS": ["ok"],
        "RESULT_DRAW": ["ok"],
        "POPUP": ["close"],
        "UNKNOWN": ["close"],
    },
    # Two-phone self-play, split by seat. Button and screen names are placeholders until
    # calibrated on the phones (`doctor --mode friendly_host` lists what's missing).
    "friendly_host": {
        "HOME": ["social"],
        "SOCIAL": ["friendly_battle"],
        "FRIENDLY_SELECT": ["invite_friend"],
        "RESULT_WIN": ["ok"], "RESULT_LOSS": ["ok"], "RESULT_DRAW": ["ok"],
        "POPUP": ["close"], "UNKNOWN": ["close"],
    },
    "friendly_guest": {
        "HOME": [],  # wait for the host's invite
        "INVITE": ["accept"],
        "RESULT_WIN": ["ok"], "RESULT_LOSS": ["ok"], "RESULT_DRAW": ["ok"],
        "POPUP": ["close"], "UNKNOWN": ["close"],
    },
    # Practice Mode gauntlet: the bot plays a bot with a real opponent's deck and levels from the
    # battle log. `practice_slot_*` are the Practice buttons of battle-log entries; the runner
    # picks one per experiment pair, so both arms of a pair face the same deck (todo R4).
    "practice": {
        "HOME": ["menu"],
        "MENU": ["battle_log"],
        "BATTLE_LOG": ["practice_slot_0", "practice_slot_1", "practice_slot_2", "practice_slot_3"],
        "PRACTICE_CONFIRM": ["confirm"],
        "TRAINING_CONFIRM": ["confirm"],
        "RESULT_WIN": ["ok"], "RESULT_LOSS": ["ok"], "RESULT_DRAW": ["ok"],
        "POPUP": ["close"], "UNKNOWN": ["close"],
    },
    "ladder": {  # Trophy Road 1v1 with the account's own battle deck: queue straight from HOME
        "HOME": ["battle"],
        # The asset-loading screen between the Battle tap and the arena, and the
        # matchmaking wait, must not be tapped: a stray close/menu tap there can cancel
        # the queue or open a screen the flow can't leave.
        "LOADING": [],
        "MATCHMAKING": [],
        "MENU": ["dismiss"],
        "TROPHY_ROAD": ["trophy_ok"],
        "RESULT_WIN": ["ok"],
        "RESULT_LOSS": ["ok"],
        "RESULT_DRAW": ["ok"],
        "POPUP": ["close"],
        "UNKNOWN": ["dismiss", "ok", "trophy_ok", "close"],
    },
}
# Screens whose button list is a choice of opponent: the navigator taps entry ``slot_index``
# (set per experiment pair) instead of the first calibrated one.
INDEXED_SCREENS = {"BATTLE_LOG"}
BATTLE_SCREENS = {"IN_BATTLE"}


def required_for(mode: str, flows: dict[str, dict[str, list[str]]] | None = None) -> tuple[set[str], set[str]]:
    """(buttons, screens) a mode's flow refers to: what `doctor` checks before a run."""
    flow = (flows or DEFAULT_FLOWS).get(mode, {})
    buttons = {b for names in flow.values() for b in names}
    screens = {s for s in flow if s not in ("UNKNOWN", "POPUP") and not s.startswith("RESULT_")}
    return buttons, screens
RESULT_SCREENS = {"RESULT_WIN": "win", "RESULT_LOSS": "loss", "RESULT_DRAW": "draw"}


@dataclass
class MenuNavigator:
    buttons: dict[str, Roi]
    flow: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_FLOWS["training_camp"]))
    cooldown_s: float = 1.2  # don't hammer a button while the screen transitions
    unknown_settle_s: float = 3.0  # let RESULT->HOME fades finish before tapping UNKNOWN
    _last_tap: float = 0.0
    _last_screen: str | None = None
    _screen_since: float = 0.0
    _unknown_idx: int = 0
    slot_index: int = 0  # which entry to pick on INDEXED_SCREENS (practice gauntlet opponent)

    def events_for(self, screen: str, frame_shape, now_ns: int) -> list[TouchEvent]:
        now = time.monotonic()
        if screen != self._last_screen:
            self._last_screen, self._screen_since, self._unknown_idx = screen, now, 0
        names = self.flow.get(screen, [])
        if not names or now - self._last_tap < self.cooldown_s:
            return []
        if screen == "UNKNOWN" and self.cooldown_s > 0 and now - self._screen_since < self.unknown_settle_s:
            return []
        candidates = [b for b in names if b in self.buttons]
        if not candidates:
            return []
        if screen == "UNKNOWN":
            name = candidates[self._unknown_idx % len(candidates)]
            self._unknown_idx += 1
        elif screen in INDEXED_SCREENS:
            name = candidates[self.slot_index % len(candidates)]
        else:
            name = candidates[0]
        self._last_tap = now
        x, y = self.buttons[name].center_px(frame_shape)
        return gestures.tap(x, y, now_ns, hold_ms=40, action_id=f"menu:{screen}:{name}", tag="menu")
