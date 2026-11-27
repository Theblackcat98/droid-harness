"""Stuck-screen recovery: click any Clash Royale screen back to HOME.

The normal menu flow assumes the calibrated button ROIs match the current layout. They
do not always: a ladder loss screen that shows a reward chest adds a "Play Again" button
and shifts "OK" to the right, so the calibrated ``ok`` centre lands in the gap between
the two buttons and every tap misses. The screen then never changes and the farm stalls
until a human intervenes.

This module is the deterministic fallback. It watches how long the *stable* screen has
not changed, and once a screen is stuck it walks an escalation ladder:

1. the calibrated buttons for that screen (the normal flow, again),
2. large saturated buttons found visually in the lower part of the frame (rightmost
   first, because Android puts the positive/dismiss action on the right),
3. BACK (closes most overlays; on a result screen it opens "Return to the Main Menu?"),
4. an app restart request (``BotRunner`` force-stops and relaunches).

Once an episode starts it keeps acting on every subsequent menu screen until HOME is
seen, so a whole chain of popups is cleared without waiting out the stuck timeout again.
Everything here emits ordinary ``TouchEvent`` taps or scrcpy keycodes; nothing runs on
the battle hot path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...core.actuate import gestures
from ...core.actuate.gestures import TouchEvent
from ...core.transport.scrcpy_protocol import MotionAction
from .perception.grid import Layout, Roi
from .perception.screens import read_result_detail

HOME = "HOME"
RESULT_SCREENS = {"RESULT_WIN", "RESULT_LOSS", "RESULT_DRAW"}
# These screens legitimately last a long time: the queue and the battle itself.
WAIT_SCREENS = {"LOADING", "MATCHMAKING", "IN_BATTLE"}
KEYCODE_BACK = 4

# Calibrated button order per screen. Result screens try the dismiss button first;
# every other screen starts with the overlay closers. Missing names are skipped.
RESULT_ORDER = ("ok", "trophy_ok", "close", "dismiss", "confirm", "menu")
GENERIC_ORDER = ("dismiss", "close", "ok", "trophy_ok", "confirm", "menu")


def home_visible(img: np.ndarray, buttons: dict[str, Roi], *, min_saturated: float = 0.55) -> bool:
    """True when the HOME Battle button is actually on screen.

    Trophy Road shares HOME's blue top HUD and can hash as HOME for a frame while its
    reward track scrolls in, but it has no Battle button. Requiring the button's gold/
    blue fill stops recovery from declaring victory on a reward screen.
    """
    roi = buttons.get("battle")
    if roi is None:
        return True  # layout without a Battle button: trust the classifier
    crop = roi.crop(img)
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturated = ((hsv[..., 1] > 110) & (hsv[..., 2] > 120)).mean()
    return float(saturated) >= min_saturated


@dataclass(frozen=True)
class ButtonCandidate:
    """A large saturated rectangle in the lower part of the frame (a game button)."""

    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


def find_buttons(img: np.ndarray, *, y_min: float = 0.55, max_buttons: int = 2) -> list[ButtonCandidate]:
    """Return wide, saturated button-like blobs in the lower part of ``img``.

    Sorted rightmost first. The filter (wide, short, solid) rejects card tiles, unit
    sprites, portraits and arena decoration; only chrome buttons survive. This is what
    recovers a screen whose calibrated ROI is off.
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 110, 120], np.uint8), np.array([180, 255, 255], np.uint8))
    mask[: int(y_min * h)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 60), max(3, h // 180)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    frame_area = float(h * w)
    out: list[ButtonCandidate] = []
    for x, y, bw, bh, area in stats[1:]:
        if not (0.08 * w <= bw <= 0.80 * w):
            continue
        if not (0.02 * h <= bh <= 0.16 * h):
            continue
        if not (1.2 <= bw / max(bh, 1) <= 8.0):
            continue
        if area < 0.004 * frame_area:
            continue
        out.append(ButtonCandidate(int(x), int(y), int(bw), int(bh), int(area)))
    out.sort(key=lambda b: b.x, reverse=True)
    return out[:max_buttons]


@dataclass(frozen=True)
class RecoveryAction:
    kind: str  # "tap" | "key"
    x: int = 0
    y: int = 0
    label: str = ""
    keycode: int = 0

    def events(self, now_ns: int) -> list[TouchEvent]:
        if self.kind != "tap":
            return []
        return gestures.tap(self.x, self.y, now_ns, hold_ms=40, action_id=f"recover:{self.label}",
                            tag="recover")


@dataclass
class HomeRecovery:
    """Decide when a screen is stuck and which tap/key walks it toward HOME."""

    buttons: dict[str, Roi]
    result_stuck_s: float = 8.0  # a result screen is stuck after this long
    screen_stuck_s: float = 15.0  # any other menu/overlay screen
    episode_settle_s: float = 1.5  # once recovering, act quickly on each new screen
    step_interval_s: float = 1.0  # minimum gap between recovery actions
    max_visual_taps: int = 2
    max_steps: int = 14  # then ask for an app restart
    _screen: str | None = None
    _since: float = 0.0
    _last_action: float = 0.0
    _actions: list[RecoveryAction] = field(default_factory=list)
    _step: int = 0
    _episode: bool = False
    _forced: bool = False
    _restart_requested: bool = False

    # --- observation ------------------------------------------------------
    def observe(self, screen: str, now: float) -> None:
        """Record the stable screen; a screen change restarts the escalation."""
        if screen == self._screen:
            return
        self._screen, self._since = screen, now
        self._last_action = 0.0
        self._actions.clear()
        self._step = 0
        self._forced = False
        if screen == HOME:
            self._episode = False
            self._restart_requested = False

    def age(self, now: float) -> float:
        return max(0.0, now - self._since)

    def is_stuck(self, screen: str, now: float) -> bool:
        if screen in WAIT_SCREENS or screen == HOME:
            return False
        if self._step > 0:
            return True  # already walking this screen back to HOME
        if self._forced or self._episode:
            # A new screen inside an episode gets a short settle, not the full timeout.
            return self.age(now) >= self.episode_settle_s
        limit = self.result_stuck_s if screen in RESULT_SCREENS else self.screen_stuck_s
        return self.age(now) >= limit

    def force(self) -> None:
        """Arm recovery now (liveness found the screen stuck) instead of waiting."""
        self._forced = True
        self._last_action = 0.0
        self._actions.clear()
        self._step = 0

    def reset(self) -> None:
        self._screen, self._since = None, 0.0
        self._last_action = 0.0
        self._actions.clear()
        self._step = 0
        self._episode = False
        self._forced = False
        self._restart_requested = False

    # --- planning ---------------------------------------------------------
    def _plan(self, screen: str, img: np.ndarray) -> list[RecoveryAction]:
        result = screen in RESULT_SCREENS
        order = RESULT_ORDER if result else GENERIC_ORDER
        plan: list[RecoveryAction] = []
        points: list[tuple[int, int]] = []

        def add(name: str) -> None:
            roi = self.buttons.get(name)
            if roi is None:
                return
            x, y = roi.center_px(img.shape)
            if any((x - px) ** 2 + (y - py) ** 2 < (0.04 * img.shape[1]) ** 2 for px, py in points):
                return
            points.append((x, y))
            plan.append(RecoveryAction("tap", x, y, label=name))

        # On a result screen the calibrated `ok` is usually the button that already
        # failed (the layout moved it), so try it once, then go straight to the
        # visually found buttons before the low-confidence overlay taps.
        for name in (order[:1] if result else ()):
            add(name)
        for i, button in enumerate(find_buttons(img, max_buttons=self.max_visual_taps)):
            x, y = button.center
            if any((x - px) ** 2 + (y - py) ** 2 < (0.04 * img.shape[1]) ** 2 for px, py in points):
                continue
            points.append((x, y))
            plan.append(RecoveryAction("tap", x, y, label=f"visual:{i}"))
        for name in (order[1:] if result else order):
            add(name)
        plan.append(RecoveryAction("key", label="back", keycode=KEYCODE_BACK))
        return plan

    def next_action(self, screen: str, img: np.ndarray, now: float) -> RecoveryAction | None:
        """Return the next recovery step, or ``None`` while waiting/not stuck."""
        if not self.is_stuck(screen, now):
            return None
        self._episode = True
        if now - self._last_action < self.step_interval_s:
            return None
        if not self._actions:
            self._actions = self._plan(screen, img)
            self._step = 0
        if self._step >= self.max_steps or self._step >= len(self._actions):
            # Every tap and BACK failed: hand off to BotRunner for a fresh app.
            self._restart_requested = True
            return None
        action = self._actions[self._step]
        self._step += 1
        self._last_action = now
        return action

    # --- restart handoff --------------------------------------------------
    def take_restart_request(self) -> bool:
        if not self._restart_requested:
            return False
        self._restart_requested = False
        return True


def _roi_mean(img: np.ndarray, roi: Roi) -> float:
    crop = roi.crop(img)
    return float(crop.mean()) if crop.size else 0.0


def walk_to_home(client, screens, layout: Layout, *, timeout_s: float = 120.0,
                 stable_s: float = 2.0, stuck_after: float = 3.0, battle_stuck_after: float = 20.0,
                 restart_after: float = 45.0, dry_run: bool = False, log=None) -> bool:
    """Drive a phone from any screen back to HOME and return whether it got there.

    Shared by ``droid-harness recover`` (harness stopped) and the batch-end settle in
    ``BotRunner``. Only the ``HomeRecovery`` ladder is used; the normal menu flow is
    deliberately not driven here, because its HOME/MENU taps can queue a battle.
    ``log`` receives one line per screen and action.
    """
    log = log or (lambda *_a, **_k: None)
    rec = HomeRecovery(layout.buttons, result_stuck_s=stuck_after, screen_stuck_s=stuck_after)
    last_seq, prev_name, home_since, first_stuck, restarts = 0, None, None, None, 0
    battle_sig: tuple[float, float] | None = None
    battle_since: float | None = None
    battle_frozen = False
    # During a chest/reward animation the classifier flaps between these; treat them
    # as one screen so the stuck timer is not reset every frame.
    ambiguous = {"IN_BATTLE", "UNKNOWN", "MATCHMAKING", "LOADING"}

    def tap(x: int, y: int, label: str) -> None:
        log("tap %s (%d,%d)", label, x, y)
        if dry_run:
            return
        client.touch(MotionAction.DOWN, x, y)
        time.sleep(0.04)
        client.touch(MotionAction.UP, x, y)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = client.slot.wait_newer(last_seq, timeout=0.25)
        if frame is not None:
            last_seq = frame.seq
        else:
            frame = client.slot.peek()
        if frame is None:
            continue
        name, _ = screens.classify(frame.img)
        if name == "UNKNOWN":
            # Ladder result screens hash UNKNOWN (new nameplate, no elixir bar);
            # verify the banners/crowns so they get the shorter stuck timeout.
            outcome, _, _ = read_result_detail(frame.img)
            if outcome is not None:
                name = f"RESULT_{outcome.upper()}"
        if name == "HOME" and not home_visible(frame.img, layout.buttons):
            name = "UNKNOWN"  # Trophy Road scrolls in over HOME's HUD; no Battle button
        now = time.monotonic()
        if name in ambiguous:
            # A live battle's clock and elixir change every second. If both are frozen
            # (the purple post-match chest reads as a full bar), it is not a battle:
            # reclassify the whole ambiguous group as UNKNOWN so the ladder can tap it.
            sig = (_roi_mean(frame.img, layout.clock), _roi_mean(frame.img, layout.elixir_bar))
            if battle_sig is None or any(abs(a - b) > 1.0 for a, b in zip(sig, battle_sig, strict=True)):
                battle_sig, battle_since, battle_frozen = sig, now, False
            elif battle_since is not None and now - battle_since >= battle_stuck_after:
                battle_frozen = True
            if battle_frozen:
                name = "UNKNOWN"
        else:
            battle_sig, battle_since, battle_frozen = None, None, False
        rec.observe(name, now)
        if name != prev_name:
            log("screen: %s", name)
            prev_name, first_stuck = name, None
        if name == "HOME":
            home_since = home_since or now
            if now - home_since >= stable_s:
                log("HOME reached")
                return True
            continue  # never tap the HOME flow (it queues a battle) while waiting
        home_since = None
        action = rec.next_action(name, frame.img, now)
        if action is not None:
            if action.kind == "tap":
                tap(action.x, action.y, f"recover:{action.label}")
            elif action.kind == "key":
                log("key BACK")
                if not dry_run:
                    client.keycode(action.keycode)
        if rec.is_stuck(name, now) and first_stuck is None:
            first_stuck = now
        if rec.take_restart_request() or (first_stuck and now - first_stuck > restart_after):
            restarts += 1
            log("restart %d: force-stop + launch", restarts)
            if not dry_run:
                client.adb.force_stop("com.supercell.clashroyale")
                client.adb.launch("com.supercell.clashroyale")
            rec.reset()
            first_stuck, home_since, prev_name = None, None, None
            time.sleep(2.0)
    log("HOME not reached in %.0fs", timeout_s)
    return False
