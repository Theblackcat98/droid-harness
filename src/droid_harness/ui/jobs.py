"""Run harness commands from the Studio as child processes.

Every button in the app runs the same ``droid-harness`` command you'd type in a terminal, so
the app adds no second code path into the bot. A job's output is kept for the page (and in
``.studio/jobs/<id>.log``); its final JSON on stdout becomes the job's result. Jobs that use
the phone or show a live view are *foreground* jobs and run one at a time: they serve their
own Studio on a private port, which the app proxies as its live view.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CLI = [sys.executable, "-m", "droid_harness.cli"]


class JobError(RuntimeError):
    pass


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def parse_json_output(text: str) -> Any:
    """The JSON a command printed last: the whole stdout, or the last line or block that parses."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i].strip()
        if ln.startswith("{") or ln.startswith("["):
            try:
                return json.loads("\n".join(lines[i:]))
            except ValueError:
                try:
                    return json.loads(ln)
                except ValueError:
                    continue
    return None


@dataclass
class Job:
    id: str
    kind: str
    title: str
    argv: list[str]
    foreground: bool = False
    live: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    status: str = "starting"  # starting, running, stopping, done, failed, stopped
    returncode: int | None = None
    started: float = field(default_factory=time.time)
    ended: float | None = None
    live_port: int | None = None
    result: Any = None
    error: str | None = None
    stop_file: Path | None = None
    proc: subprocess.Popen | None = None
    lines: deque = field(default_factory=lambda: deque(maxlen=3000))
    seq: int = 0
    stdout: list[str] = field(default_factory=list)
    stdout_bytes: int = 0
    log_path: Path | None = None
    on_done: list[Callable[[Job], None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running", "stopping")

    def add_line(self, text: str, stream: str) -> None:
        with self._lock:
            self.seq += 1
            self.lines.append((self.seq, stream, text))
            if stream == "out" and self.stdout_bytes < 4_000_000:
                self.stdout.append(text)
                self.stdout_bytes += len(text)
        if self.log_path is not None:
            try:
                with self.log_path.open("a") as f:
                    f.write(text + "\n")
            except OSError:
                pass

    def log_since(self, since: int = 0, limit: int = 400) -> dict[str, Any]:
        with self._lock:
            rows = [r for r in self.lines if r[0] > since]
            first = self.lines[0][0] if self.lines else 0
        rows = rows[-limit:]
        return {"lines": [{"n": n, "s": s, "t": t} for n, s, t in rows], "last": rows[-1][0] if rows else since,
                "truncated": bool(rows) and rows[0][0] > since + 1 and since + 1 < first}

    def summary(self) -> dict[str, Any]:
        end = self.ended or time.time()
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "foreground": self.foreground, "live": self.live, "live_port": self.live_port if self.active else None,
                "returncode": self.returncode, "started": self.started, "ended": self.ended,
                "elapsed_s": round(end - self.started, 1), "result": self.result, "error": self.error,
                "params": self.params, "command": " ".join(["droid-harness", *self.argv]),
                "graceful_stop": self.stop_file is not None, "log_last": self.seq}


class JobManager:
    def __init__(self, root: Path = Path("."), *, logs_dir: Path | None = None,
                 env: Callable[[], dict[str, str]] | None = None, base_cmd: list[str] | None = None,
                 history: int = 40) -> None:
        self.root = Path(root)
        self.logs_dir = logs_dir or self.root / ".studio/jobs"
        self.env = env or (lambda: dict(os.environ, PYTHONUNBUFFERED="1"))
        self.base_cmd = list(base_cmd or CLI)
        self.history = history
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # --- queries ------------------------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self.jobs.values())
        return [j.summary() for j in sorted(jobs, key=lambda j: j.started, reverse=True)]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self.jobs.get(job_id)

    def foreground(self) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs.values() if j.foreground and j.active), None)

    def live_job(self) -> Job | None:
        j = self.foreground()
        return j if j is not None and j.live and j.live_port else None

    # --- lifecycle ----------------------------------------------------------------------
    def start(self, kind: str, title: str, args: list[str], *, foreground: bool = False, live: bool = False,
              params: dict[str, Any] | None = None, stop_file: Path | None = None,
              on_done: Callable[[Job], None] | None = None, raw: bool = False, ui_flag: bool = True) -> Job:
        """Start ``droid-harness <args>`` (or ``args`` itself with ``raw``).

        A live job gets ``--ui --ui-port <free port>`` appended so the app can proxy its view
        (``ui_flag=False`` for commands whose Studio is always on, like ``watch``).
        """
        with self._lock:
            if foreground:
                busy = next((j for j in self.jobs.values() if j.foreground and j.active), None)
                if busy is not None:
                    raise JobError(f"{busy.title} is still running: stop it first")
            job = Job(id=uuid.uuid4().hex[:8], kind=kind, title=title, argv=list(args), foreground=foreground,
                      live=live, params=dict(params or {}), stop_file=stop_file)
            if on_done:
                job.on_done.append(on_done)
            self.jobs[job.id] = job
            self._trim()
        if live:
            job.live_port = free_port()
            job.argv += [*(["--ui"] if ui_flag else []), "--ui-host", "127.0.0.1", "--ui-port", str(job.live_port)]
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        job.log_path = self.logs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}-{job.id}.log"
        cmd = list(job.argv) if raw else [*self.base_cmd, *job.argv]
        job.add_line("$ " + " ".join(["droid-harness", *job.argv] if not raw else job.argv), "app")
        try:
            job.proc = subprocess.Popen(cmd, cwd=self.root, env=self.env(), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                                        errors="replace")
        except OSError as e:
            job.status, job.error, job.ended = "failed", f"couldn't start: {e}", time.time()
            job.add_line(job.error, "app")
            self._finish_callbacks(job)
            return job
        job.status = "running"
        readers = [threading.Thread(target=self._pump, args=(job, job.proc.stdout, "out"), daemon=True),
                   threading.Thread(target=self._pump, args=(job, job.proc.stderr, "err"), daemon=True)]
        for t in readers:
            t.start()
        threading.Thread(target=self._wait, args=(job, readers), name=f"job-{job.id}", daemon=True).start()
        return job

    def _trim(self) -> None:
        done = sorted((j for j in self.jobs.values() if not j.active), key=lambda j: j.started)
        for j in done[: max(0, len(self.jobs) - self.history)]:
            self.jobs.pop(j.id, None)

    @staticmethod
    def _pump(job: Job, stream, name: str) -> None:
        try:
            for line in stream:
                job.add_line(line.rstrip("\n"), name)
        except (OSError, ValueError):
            pass

    def _wait(self, job: Job, readers: list[threading.Thread]) -> None:
        rc = job.proc.wait()
        for t in readers:
            t.join(timeout=5)
        job.returncode, job.ended = rc, time.time()
        job.result = parse_json_output("\n".join(job.stdout))
        if job.status == "stopping":
            job.status = "stopped"
        elif rc == 0:
            job.status = "done"
        else:
            job.status = "failed"
            errs = [t for _, s, t in list(job.lines) if s == "err" and t.strip()]
            job.error = (errs[-1] if errs else f"exited with code {rc}")[-400:]
        job.add_line(f"[{job.status}, exit code {rc}, {job.ended - job.started:.0f} s]", "app")
        self._finish_callbacks(job)

    @staticmethod
    def _finish_callbacks(job: Job) -> None:
        for cb in job.on_done:
            try:
                cb(job)
            except Exception:  # noqa: BLE001
                log.exception("job %s callback failed", job.id)

    def stop(self, job_id: str, *, now: bool = False, grace_s: float = 20.0) -> Job:
        """Ask a job to stop: its stop file if it has one (finishes the current cycle), else Ctrl+C.

        ``now`` skips the stop file. Escalates to SIGTERM, then SIGKILL, if the job lingers.
        """
        job = self.get(job_id)
        if job is None:
            raise JobError(f"no job {job_id}")
        if not job.active or job.proc is None:
            return job
        if job.stop_file is not None and not now and job.status != "stopping":
            job.stop_file.parent.mkdir(parents=True, exist_ok=True)
            job.stop_file.touch()
            job.add_line(f"[stop requested: {job.stop_file} written; the job ends after its current step]", "app")
            job.status = "stopping"
            return job
        job.status = "stopping"
        job.add_line("[stopping: Ctrl+C sent]", "app")
        try:
            job.proc.send_signal(signal.SIGINT)
        except OSError:
            pass
        threading.Thread(target=self._escalate, args=(job, grace_s), daemon=True).start()
        return job

    @staticmethod
    def _escalate(job: Job, grace_s: float) -> None:
        for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5.0)):
            try:
                job.proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                job.add_line(f"[still running: sending {sig.name}]", "app")
                try:
                    job.proc.send_signal(sig)
                except OSError:
                    return

    def stop_all(self, timeout_s: float = 10.0) -> None:
        with self._lock:
            active = [j for j in self.jobs.values() if j.active]
        for j in active:
            try:
                self.stop(j.id, now=True, grace_s=timeout_s)
            except JobError:
                pass
        deadline = time.time() + timeout_s
        for j in active:
            if j.proc is not None:
                try:
                    j.proc.wait(timeout=max(0.1, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    j.proc.kill()
