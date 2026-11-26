"""T5 liveness: tell a broken session from a static screen.

scrcpy produces new frames only when the screen changes, so a frame gap on a static
menu or a settled app is normal. Failure means a socket error, a dead server process,
or a failed ADB ping. During a battle the screen always animates, so a gap there *is*
an anomaly.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class LivenessSnapshot:
    socket_error: bool
    server_alive: bool
    seconds_since_ping_ok: float
    in_battle: bool
    ms_since_packet: float
    screen_id: str | None = None
    seconds_on_screen: float = 0.0


@dataclass
class LivenessPolicy:
    ping_fail_s: float = 5.0
    battle_gap_anomaly_ms: float = 250.0
    battle_gap_reconnect_ms: float = 2000.0
    stuck_screen_s: float = 30.0


@dataclass(frozen=True)
class Finding:
    kind: str  # session_broken | server_exited | ping_failed | battle_frame_gap | battle_stall | stuck_screen
    action: str  # reconnect | restart_server | adb_reconnect | log_latency_anomaly | recover_ui
    detail: str = ""


def assess(s: LivenessSnapshot, p: LivenessPolicy | None = None) -> list[Finding]:
    p = p or LivenessPolicy()
    out: list[Finding] = []
    if s.socket_error:
        out.append(Finding("session_broken", "reconnect"))
    if not s.server_alive:
        out.append(Finding("server_exited", "restart_server"))
    if s.seconds_since_ping_ok > p.ping_fail_s:
        out.append(Finding("ping_failed", "adb_reconnect", f"{s.seconds_since_ping_ok:.1f}s"))
    if s.in_battle:
        if s.ms_since_packet > p.battle_gap_reconnect_ms:
            out.append(Finding("battle_stall", "reconnect", f"{s.ms_since_packet:.0f}ms"))
        elif s.ms_since_packet > p.battle_gap_anomaly_ms:
            out.append(Finding("battle_frame_gap", "log_latency_anomaly", f"{s.ms_since_packet:.0f}ms"))
    elif s.screen_id is not None and s.seconds_on_screen > p.stuck_screen_s:
        out.append(Finding("stuck_screen", "recover_ui", f"{s.screen_id} for {s.seconds_on_screen:.0f}s"))
    # Outside battle, a frame gap alone is never a finding: it means the screen is static.
    return out


class LivenessMonitor:
    """Runs ``assess`` every ``period_s`` and pings ADB off the hot path."""

    def __init__(self, client, *, in_battle: Callable[[], bool], screen: Callable[[], tuple[str | None, float]],
                 on_finding: Callable[[Finding], None], policy: LivenessPolicy | None = None,
                 period_s: float = 0.25, ping_every_s: float = 2.0) -> None:
        self.client = client
        self.in_battle = in_battle
        self.screen = screen
        self.on_finding = on_finding
        self.policy = policy or LivenessPolicy()
        self.period_s = period_s
        self.ping_every_s = ping_every_s
        self.last_ping_ok = time.monotonic()
        self._last_ping = 0.0
        self._running = False

    def snapshot(self) -> LivenessSnapshot:
        sid, secs = self.screen()
        return LivenessSnapshot(
            socket_error=self.client.error is not None,
            server_alive=self.client.server_alive(),
            seconds_since_ping_ok=time.monotonic() - self.last_ping_ok,
            in_battle=self.in_battle(),
            ms_since_packet=self.client.ms_since_last_packet(),
            screen_id=sid, seconds_on_screen=secs)

    def _ping(self) -> None:
        if self.client.adb.ping():
            self.last_ping_ok = time.monotonic()

    def _loop(self) -> None:
        while self._running:
            now = time.monotonic()
            if now - self._last_ping > self.ping_every_s:
                self._last_ping = now
                threading.Thread(target=self._ping, daemon=True).start()
            for f in assess(self.snapshot(), self.policy):
                self.on_finding(f)
            time.sleep(self.period_s)

    def start(self) -> LivenessMonitor:
        self._running = True
        threading.Thread(target=self._loop, name="T5-liveness", daemon=True).start()
        return self

    def stop(self) -> None:
        self._running = False
