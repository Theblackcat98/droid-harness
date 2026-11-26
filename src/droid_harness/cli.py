"""droid-harness command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml


def _profile(path: str):
    from .core.config import DeviceProfile

    p = Path(path)
    return DeviceProfile(**{**yaml.safe_load(p.read_text()), "name": p.stem})


def cmd_stream(a) -> None:
    import cv2

    from .core.telemetry.latency import percentiles
    from .core.transport.scrcpy_bridge import ScrcpyClient

    c = ScrcpyClient(_profile(a.profile))
    c.start()
    print(f"connected: {c.device_name} {c.size[0]}x{c.size[1]} {c.codec}")
    decode_ms, n, last, t0 = [], 0, 0, time.time()
    try:
        while time.time() - t0 < a.seconds:
            f = c.slot.wait_newer(last, timeout=1.0)
            if f is None:
                continue
            last, n = f.seq, n + 1
            decode_ms.append((f.t_decoded_ns - f.t_recv_ns) / 1e6)
        if a.save_frame:
            cv2.imwrite(a.save_frame, c.slot.peek().img)
            print(f"saved {a.save_frame}")
    finally:
        c.stop()
    print(f"frames: {n} in {a.seconds}s ({n / a.seconds:.1f} fps; static screens send fewer)")
    print("decode ms:", json.dumps(percentiles(decode_ms)))


def cmd_loopback(a) -> None:
    from .core.telemetry.loopback import loopback_rtt
    from .core.transport.scrcpy_bridge import ScrcpyClient

    c = ScrcpyClient(_profile(a.profile))
    c.start()
    try:
        c.adb.shell("settings put system show_touches 1", check=False)
        c.slot.wait_newer(0, timeout=2)
        r = loopback_rtt(c, (a.x, a.y), samples=a.samples)
    finally:
        c.adb.shell("settings put system show_touches 0", check=False)
        c.stop()
    r.pop("samples_ms")
    print(json.dumps(r, indent=1))
    out = Path(a.log)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"profile": a.profile, "t": time.time(), **r}) + "\n")
    print(f"Set exec_latency_ms in {a.profile} to ~{r.get('p50', 0):.0f} (the input+render share of it).")


def cmd_calibrate_tiles(a) -> None:
    from .domains.clash_royale.perception.grid import TileMapper

    lm = yaml.safe_load(Path(a.landmarks).read_text())
    w, h = lm["frame_size"]
    screen = [(p["px"][0] / w, p["px"][1] / h) for p in lm["points"]]
    tiles = [tuple(p["tile"]) for p in lm["points"]]
    tm = TileMapper.fit(screen, tiles)
    err = max(((tm.to_tiles([s])[0] - t) ** 2).sum() ** 0.5 for s, t in zip(screen, tiles, strict=True))
    tm.save(a.out)
    print(f"saved {a.out}; max landmark error {err:.2f} tiles")


def cmd_template(a) -> None:
    import cv2

    from .domains.clash_royale.perception.cards import to_lightness
    from .domains.clash_royale.perception.grid import Layout

    lay = Layout.load(a.layout)
    img = cv2.imread(a.frame)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    crop = (lay.next_card if a.slot == "next" else lay.hand[int(a.slot)]).crop(img)
    i = len(list(out.glob(f"{a.card}__*.png")))
    cv2.imwrite(str(out / f"{a.card}__{i}.png"), to_lightness(crop))
    print(f"added template {a.card}__{i}.png ({a.state})")


def cmd_screen(a) -> None:
    import shutil

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    i = len(list(out.glob(f"{a.name}__*.png")))
    shutil.copy(a.frame, out / f"{a.name}__{i}.png")
    print(f"registered screen {a.name} ({out / f'{a.name}__{i}.png'})")


def cmd_run(a) -> None:
    from .app import BotRunner, RunConfig

    cfg = RunConfig(**{k: (Path(v) if isinstance(v, str) and k not in ("mode", "opponent_id", "balance_epoch",
                                                                         "experiment_id", "arm") else v)
                       for k, v in vars(a).items() if k in RunConfig.__dataclass_fields__ and v is not None})
    BotRunner(cfg).run()


def cmd_sim(a) -> None:
    from .app import run_sim

    r = run_sim(a.seconds, Path(a.out))
    print(json.dumps(r, indent=1, default=str))


def cmd_analyze(a) -> None:
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .flywheel import warehouse
    from .flywheel.report import batch_report

    cards, arena = CardDB.load(a.cards), Arena()
    con = warehouse.connect(a.db)
    dirs = [d for d in sorted(Path(a.data).iterdir()) if (d / "summary.json").exists()]
    analyzed = []
    for d in dirs:
        tl, engs = warehouse.analyze_match(d, arena, cards)
        warehouse.ingest(con, tl, engs, str(d / "stream.h264"))
        analyzed.append((tl, engs))
    print(json.dumps(batch_report(analyzed), indent=1))
    for row in warehouse.win_rates(con, a.game_version):
        print(row)


def cmd_sprt(a) -> None:
    from .flywheel.eval import sprt as S

    if a.wins is None:
        for p1 in (0.55, 0.58, 0.60, 0.65):
            print(f"H1={p1:.2f} (≈{S.elo_from_p(p1):+.0f} Elo): ~{S.expected_games(a.p0, p1):.0f} games")
        return
    r = S.sprt(a.wins, a.losses, a.p0, a.p1, a.alpha, a.beta)
    print(f"{r.decision}  LLR {r.llr:+.3f}  bounds [{r.lower:.3f}, {r.upper:.3f}]")


def cmd_schedule(a) -> None:
    from .flywheel.eval.selfplay import Seat, balance_report, schedule

    sa, sb = Seat(*a.a.split(":")), Seat(*a.b.split(":"))
    games = schedule(a.games, sa, sb)
    for g in games[: min(len(games), 8)]:
        print(g)
    print(json.dumps(balance_report(games)))


def cmd_ledger(a) -> None:
    from .flywheel.ledger import Ledger

    print(Ledger(a.store).render_markdown(a.out))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="droid-harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stream", help="connect to a phone and report fps/decode time")
    s.add_argument("--profile", required=True)
    s.add_argument("--seconds", type=float, default=10)
    s.add_argument("--save-frame")
    s.set_defaults(fn=cmd_stream)

    s = sub.add_parser("loopback", help="software round-trip latency test (needs a quiet screen spot)")
    s.add_argument("--profile", required=True)
    s.add_argument("--x", type=int, required=True)
    s.add_argument("--y", type=int, required=True)
    s.add_argument("--samples", type=int, default=100)
    s.add_argument("--log", default="data/latency_runs.jsonl")
    s.set_defaults(fn=cmd_loopback)

    s = sub.add_parser("calibrate-tiles", help="fit the pixel->tile homography from landmarks")
    s.add_argument("--landmarks", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_calibrate_tiles)

    s = sub.add_parser("template", help="add a hand-card template from a saved frame")
    s.add_argument("--frame", required=True)
    s.add_argument("--layout", required=True)
    s.add_argument("--slot", required=True, help="0-3 or 'next'")
    s.add_argument("--card", required=True)
    s.add_argument("--state", default="affordable", choices=["affordable", "greyed", "selected"])
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_template)

    s = sub.add_parser("screen", help="register a saved frame as a reference screen (HOME, IN_BATTLE, ...)")
    s.add_argument("--frame", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_screen)

    s = sub.add_parser("run", help="play matches on a phone")
    for k in ("profile", "layout", "tiles", "templates", "screens", "strategy", "cards"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--data", default="data/matches")
    s.add_argument("--detector")
    s.add_argument("--classes")
    s.add_argument("--digits")
    s.add_argument("--mode", default="training_camp", choices=["training_camp", "friendly"])
    s.add_argument("--opponent-id", dest="opponent_id")
    s.add_argument("--matches", type=int, default=10)
    s.add_argument("--balance-epoch", dest="balance_epoch")
    s.add_argument("--experiment-id", dest="experiment_id")
    s.add_argument("--arm")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("sim", help="closed-loop demo with no phone (synthetic scene)")
    s.add_argument("--seconds", type=float, default=10)
    s.add_argument("--out", default="data/sim")
    s.set_defaults(fn=cmd_sim)

    s = sub.add_parser("analyze", help="engagements, triage, report, and warehouse ingest")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--cards", required=True)
    s.add_argument("--db", default="data/warehouse.duckdb")
    s.add_argument("--game-version", dest="game_version")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("sprt", help="SPRT decision, or the sample-size table with no results")
    s.add_argument("--wins", type=int)
    s.add_argument("--losses", type=int)
    s.add_argument("--p0", type=float, default=0.5)
    s.add_argument("--p1", type=float, default=0.58)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--beta", type=float, default=0.05)
    s.set_defaults(fn=cmd_sprt)

    s = sub.add_parser("selfplay-schedule", help="two-phone schedule with strategy<->seat swapping")
    s.add_argument("--games", type=int, default=200)
    s.add_argument("--a", required=True, help="phone:account")
    s.add_argument("--b", required=True, help="phone:account")
    s.set_defaults(fn=cmd_schedule)

    s = sub.add_parser("ledger", help="render the Strategy Ledger markdown")
    s.add_argument("--store", default="ledger/experiments.jsonl")
    s.add_argument("--out", default="ledger/STRATEGY_LEDGER.md")
    s.set_defaults(fn=cmd_ledger)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
