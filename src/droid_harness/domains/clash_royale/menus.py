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
}
BATTLE_SCREENS = {"IN_BATTLE"}
RESULT_SCREENS = {"RESULT_WIN": "win", "RESULT_LOSS": "loss", "RESULT_DRAW": "draw"}


@dataclass
class MenuNavigator:
    buttons: dict[str, Roi]
    flow: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_FLOWS["training_camp"]))
    cooldown_s: float = 1.2  # don't hammer a button while the screen transitions
    _last_tap: float = 0.0

    def events_for(self, screen: str, frame_shape, now_ns: int) -> list[TouchEvent]:
        names = self.flow.get(screen, [])
        if not names or time.monotonic() - self._last_tap < self.cooldown_s:
            return []
        name = names[0]
        if name not in self.buttons:
            return []
        self._last_tap = time.monotonic()
        x, y = self.buttons[name].center_px(frame_shape)
        return gestures.tap(x, y, now_ns, hold_ms=40, action_id=f"menu:{screen}:{name}", tag="menu")
