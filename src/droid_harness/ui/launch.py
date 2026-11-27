"""Start the Studio next to a running bot, or on its own for replay, in a browser or a window."""

from __future__ import annotations

import logging
import threading
import webbrowser
from collections.abc import Callable
from pathlib import Path

from .library import MatchLibrary
from .server import Studio, StudioServer
from .snapshot import CostBook, StrategyIndex, geometry
from .telemetry import LivePublisher, TelemetryTap, experiment_state

log = logging.getLogger(__name__)


def serve_live(tap: TelemetryTap, context: Callable[[], dict], *, cards=None, layout=None, tiles=None,
               arena=None, replay_roots: list[Path] | None = None, title: str = "droid-harness Studio",
               host: str = "127.0.0.1", port: int = 8765, fps: float = 15.0) -> StudioServer:
    """Serve a live view fed by ``tap``; ``context()`` supplies perception/params/runner info."""
    pub = LivePublisher(tap, context, costs=CostBook(cards), fps=fps).start()
    library = None
    if replay_roots:
        library = MatchLibrary(replay_roots, strategies=StrategyIndex(), costs=CostBook(cards))
    studio = Studio(title=title, live=pub, live_config=geometry(layout, tiles, arena), library=library)
    return StudioServer(studio, host, port).start()


def start_live_studio(bot, *, host: str = "127.0.0.1", port: int = 8765, fps: float = 15.0,
                      title: str | None = None, replay_roots: list[Path] | None = None) -> StudioServer:
    """Serve the live view of ``bot`` (a ``BotRunner`` built with ``ui=TelemetryTap()``).

    The same page also browses this run's finished matches (``replay_roots`` defaults to the
    run's ``--data`` directory).
    """
    if bot.ui is None:
        bot.ui = TelemetryTap()
    mode = "watching (observe only)" if bot.cfg.observe_only else f"{bot.cfg.mode} on {bot.profile.name}"
    return serve_live(bot.ui, bot.ui_context, cards=bot.cards, layout=bot.layout, tiles=bot.tiles,
                      arena=bot.arena, replay_roots=replay_roots or [bot.cfg.data],
                      title=title or f"droid-harness Studio: {mode}", host=host, port=port, fps=fps)


def attach_experiment(bot, runner) -> None:
    """Show the paired experiment's record and SPRT state in the live strip."""
    bot.ui_experiment = lambda: experiment_state(runner, active_arm=bot.cfg.arm)


def start_replay_studio(roots: list[Path], *, host: str = "127.0.0.1", port: int = 8765,
                        cards: Path | None = None, strategies: Path = Path("configs/strategies"),
                        ledger: Path = Path("ledger/experiments.jsonl"), initial_match: str | None = None,
                        title: str = "droid-harness Studio: replay", value: Path | None = None) -> StudioServer:
    vm = None
    if value is not None and Path(value).exists():
        from ..learn.value_model import ValueModel

        try:
            vm = ValueModel.load(value)
        except ValueError as e:
            log.warning("ignoring %s: %s", value, e)
    library = MatchLibrary(roots, strategies=StrategyIndex(strategies), costs=CostBook.load(cards), value_model=vm)
    studio = Studio(title=title, library=library, ledger=ledger, initial_match=initial_match)
    return StudioServer(studio, host, port).start()


def run_with_window(work: Callable[[], None], url: str, *, native: bool = False, open_browser: bool = False,
                    title: str = "droid-harness Studio", on_close: Callable[[], None] | None = None) -> None:
    """Run ``work`` (the bot loop, or a wait) while the Studio is shown.

    ``native``: a desktop window via pywebview (it must own the main thread, so ``work`` moves to
    a worker thread and closing the window calls ``on_close``). Without pywebview, or without
    ``native``, the page is served for any browser and ``work`` runs here.
    """
    webview = None
    if native:
        try:
            import webview  # type: ignore[import-not-found]
        except ImportError:
            log.warning("--native needs pywebview (pip install -e '.[gui]'); opening %s in a browser instead", url)
            open_browser = True
    if webview is not None:
        errors: list[BaseException] = []

        def runner() -> None:
            try:
                work()
            except BaseException as e:  # surfaced after the window closes
                errors.append(e)

        th = threading.Thread(target=runner, name="studio-work", daemon=True)
        th.start()
        webview.create_window(title, url, width=1480, height=940, min_size=(960, 640))
        webview.start()
        if on_close is not None:
            on_close()
        th.join(timeout=60)
        if errors:
            raise errors[0]
        return
    print(f"Studio: {url}  (Ctrl+C to stop; over SSH: ssh -L {url.rsplit(':', 1)[-1].strip('/')}:localhost:"
          f"{url.rsplit(':', 1)[-1].strip('/')} <host>)", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    work()


def wait_forever() -> None:
    stop = threading.Event()
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
