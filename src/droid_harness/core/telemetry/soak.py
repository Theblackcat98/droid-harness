"""Stream soak: hold a session open for a long time and count what liveness would do.

The M1 check: on static menus scrcpy sends no frames, which is normal. A soak proves
that a long quiet stream produces zero reconnect-class findings (``session_broken``,
``server_exited``, ``ping_failed``); any it does produce are real, and the soak
reconnects the way ``BotRunner`` would and keeps counting.
"""

from __future__ import annotations

import queue
import time
from collections import Counter
from collections.abc import Callable

from ..runtime.liveness import Finding, LivenessMonitor, LivenessPolicy

RECONNECT_ACTIONS = ("reconnect", "restart_server", "adb_reconnect")


def soak(make_client: Callable[[], object], seconds: float, *, start_kw: dict | None = None,
         policy: LivenessPolicy | None = None, ping_every_s: float = 2.0,
         report_every_s: float = 60.0, connect_attempts: int = 5, backoff_s: float = 2.0,
         log: Callable[[str], None] | None = None) -> dict:
    """Run for ``seconds``; returns frames, reconnects and findings by kind."""
    start_kw = start_kw or {}
    findings: Counter = Counter()
    q: queue.Queue[Finding] = queue.Queue()
    reconnects, frames, max_gap_ms, connect_failures = 0, 0, 0.0, 0

    def connect():
        nonlocal connect_failures
        for attempt in range(connect_attempts):
            try:
                c = make_client()
                c.start(**start_kw)
                break
            except Exception as e:  # the phone may need a moment after a broken session
                connect_failures += 1
                if log:
                    log(f"connect attempt {attempt + 1} failed: {e}")
                if attempt == connect_attempts - 1:
                    raise
                time.sleep(backoff_s)
        m = LivenessMonitor(c, in_battle=lambda: False, screen=lambda: (None, 0.0), on_finding=q.put,
                            policy=policy, ping_every_s=ping_every_s).start()
        return c, m

    client, monitor = connect()
    t0 = last_report = time.monotonic()
    last_seq = 0
    try:
        while time.monotonic() - t0 < seconds:
            f = client.slot.peek()
            if f is not None and f.seq != last_seq:
                frames += max(1, f.seq - last_seq)
                last_seq = f.seq
            gap = client.ms_since_last_packet()
            if gap != float("inf"):
                max_gap_ms = max(max_gap_ms, gap)
            try:
                finding = q.get(timeout=0.25)
            except queue.Empty:
                finding = None
            if finding is not None:
                findings[finding.kind] += 1
                if finding.action in RECONNECT_ACTIONS:
                    reconnects += 1
                    if log:
                        log(f"reconnect after {finding.kind} ({finding.detail}) at "
                            f"{time.monotonic() - t0:.0f}s")
                    monitor.stop()
                    client.stop()
                    while not q.empty():  # findings from the dead session are stale now
                        q.get_nowait()
                    time.sleep(min(1.0, backoff_s))
                    client, monitor = connect()
                    last_seq = 0
            if log and time.monotonic() - last_report >= report_every_s:
                last_report = time.monotonic()
                log(f"{last_report - t0:.0f}s: frames {frames}, reconnects {reconnects}, "
                    f"findings {dict(findings)}")
    finally:
        monitor.stop()
        client.stop()
    elapsed = time.monotonic() - t0
    return {"seconds": round(elapsed, 1), "frames": frames, "reconnects": reconnects,
            "connect_failures": connect_failures,
            "findings": dict(findings), "max_packet_gap_ms": round(max_gap_ms, 1),
            "verdict": "ok" if reconnects == 0 else "reconnects: inspect findings"}
