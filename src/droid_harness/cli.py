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

    if a.duration:
        from .core.telemetry.soak import soak

        prof = _profile(a.profile)
        report = soak(lambda: ScrcpyClient(prof), a.duration, log=print)
        print(json.dumps(report, indent=1))
        return
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


def cmd_tower_template(a) -> None:
    import cv2

    from .domains.clash_royale.perception.grid import Layout

    lay = Layout.load(a.layout)
    img = cv2.imread(a.frame)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    towers = list(lay.towers) if a.tower == "all" else [a.tower]
    for t in towers:
        i = len(list(out.glob(f"{t}__{a.state}__*.png")))
        cv2.imwrite(str(out / f"{t}__{a.state}__{i}.png"), cv2.cvtColor(lay.towers[t].crop(img), cv2.COLOR_BGR2GRAY))
        print(f"added {t}__{a.state}__{i}.png")


def cmd_synth(a) -> None:
    from .domains.clash_royale.perception.classes import load_class_names
    from .learn import synth

    names = load_class_names(a.classes)
    backgrounds = [img for d in a.backgrounds for img in synth.load_backgrounds(d)]
    sprites, counts = synth.load_sprites(a.sprites, names)
    print(json.dumps({"backgrounds": len(backgrounds), "sprites": dict(counts)}, indent=1))
    cfg = synth.SynthConfig(imgsz=tuple(a.imgsz), seed=a.seed)
    data = synth.generate_dataset(backgrounds, sprites, Path(a.out), a.n, names, cfg,
                                  bars=synth.load_bars(a.bars), val_frac=a.val_frac)
    print(f"wrote {a.n} images; dataset {data}")


def cmd_convert_katacr(a) -> None:
    from .domains.clash_royale.perception.classes import load_class_names
    from .learn import katacr

    data, counts = katacr.convert(a.part2, a.out, load_class_names(a.classes), katacr.load_alias(a.alias),
                                  split=a.split, empty_keep=a.empty_keep, link=not a.copy)
    print(json.dumps(dict(counts), indent=1))
    print(f"dataset {data}")


def cmd_train_detector(a) -> None:
    from .learn.train_detector import main as train_main

    train_main(a.args)


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


def cmd_experiment(a) -> None:
    """Alternate champion/candidate Training Camp matches until SPRT decides (M5)."""
    from .app import BotRunner, RunConfig
    from .domains.clash_royale.policy.strategy import load_strategy
    from .flywheel import warehouse
    from .flywheel.eval.runner import ExperimentConfig, ExperimentRunner
    from .flywheel.ledger import Ledger

    _, _, champ_sid = load_strategy(a.champion)
    _, _, cand_sid = load_strategy(a.candidate)
    store = Ledger(a.store)
    exp_id = a.experiment_id or store.next_id()
    # Detect the game version once before the loop; the runner voids on change.
    probe = RunConfig(**{k: (Path(v) if isinstance(v, str) and k not in ("mode", "opponent_id",
                         "balance_epoch", "experiment_id", "arm") else v)
                         for k, v in vars(a).items()
                         if k in RunConfig.__dataclass_fields__ and v is not None})
    game_version = BotRunner(probe).client_version()
    runner = ExperimentRunner(ExperimentConfig(
        experiment_id=exp_id, title=a.title, hypothesis=a.hypothesis, proposer=a.proposer,
        champion_id=champ_sid, candidate_id=cand_sid, test=a.test,
        p0=a.p0, p1=a.p1, alpha=a.alpha, beta=a.beta,
        game_version=game_version or "", balance_epoch=a.balance_epoch or "",
        max_games=a.max_games))
    seen: set[str] = set()
    data_root = Path(a.data)
    for _ in range(a.max_games):
        if runner.done:
            break
        arm = runner.next_arm()
        strat = a.champion if arm == "champion" else a.candidate
        cfg = RunConfig(**{k: (Path(v) if isinstance(v, str) and k not in ("mode", "opponent_id",
                             "balance_epoch", "experiment_id", "arm") else v)
                            for k, v in vars(a).items()
                            if k in RunConfig.__dataclass_fields__ and v is not None})
        cfg.strategy = Path(strat)
        cfg.experiment_id, cfg.arm, cfg.matches = exp_id, arm, 1
        BotRunner(cfg).run()
        for p in sorted(data_root.glob("*/summary.json")):
            if str(p) in seen:
                continue
            seen.add(str(p))
            try:
                s = json.loads(p.read_text())
            except Exception:
                continue
            if s.get("experiment_id") != exp_id:
                continue
            r = runner.record(s)
            print(json.dumps({"match": s.get("match_id"), "arm": arm, "result": s.get("result"),
                              "sprt": r.decision, "llr": round(r.llr, 3)}))
        runner.write_ledger(a.store)
        try:
            con = warehouse.connect(a.db)
            runner.write_warehouse(con)
            con.close()
        except Exception as e:  # warehouse must never stop the loop
            print(f"warehouse write failed: {e}")
    print(json.dumps({"experiment_id": exp_id, "status": runner.status,
                      "matches": len(runner.matches), "decision": runner.decision.__dict__
                      if runner.decision else None, "void": runner.void_reason}))


def cmd_replay_diff(a) -> None:
    """Gate 2 of the funnel: candidate vs champion on logged real states, before any match."""
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .domains.clash_royale.policy.strategy import load_strategy
    from .domains.clash_royale.policy.utility import CRPolicy
    from .flywheel.eval.replay_diff import gate, load_states, replay_diff

    cards = CardDB.load(a.cards)
    arena = Arena.load(a.arena) if a.arena else Arena()
    pols = [CRPolicy(cards, arena, load_strategy(p)[1]) for p in (a.champion, a.candidate)]
    dirs = sorted(d for d in Path(a.data).iterdir() if (d / "decisions.jsonl").exists())
    summary = replay_diff(*pols, load_states(dirs))
    ok, why = gate(summary, a.expected_pct, a.tolerance)
    print(json.dumps({"matches": len(dirs), "states": summary.states, "changed": summary.changed,
                      "changed_pct": round(summary.changed_pct, 2), "errors": summary.errors,
                      "by_champion_rule": dict(summary.by_rule), "examples": summary.examples[:10],
                      "gate": "pass" if ok else "fail", "reason": why}, indent=1, default=str))
    if not ok:
        raise SystemExit(1)


def cmd_replay(a) -> None:
    """Re-run recorded matches through perception; diff events against a previous run (M3)."""
    from .app import load_digits
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .domains.clash_royale.perception.cards import CardMatcher
    from .domains.clash_royale.perception.detector import OnnxDetector
    from .domains.clash_royale.perception.grid import Layout, TileMapper
    from .domains.clash_royale.perception.pipeline import CRPerception
    from .domains.clash_royale.perception.towers import TowerStateReader
    from .domains.clash_royale.replay import diff_runs, replay_match

    cards = CardDB.load(a.cards)
    detector = None
    if a.detector:
        detector = OnnxDetector(a.detector, a.classes, conf=a.det_conf, iou=a.det_iou)
        detector.warmup(2)

    def make():
        return CRPerception(layout=Layout.load(a.layout), tiles=TileMapper.load(a.tiles), cards=cards,
                            arena=Arena.load(a.arena) if a.arena else Arena(),
                            matcher=CardMatcher.load(a.templates), digits=load_digits(a.digits),
                            detector=detector, detect_every=a.detect_every,
                            tower_states=TowerStateReader.load(a.towers) if a.towers else None)

    dirs = sorted(d for d in Path(a.data).iterdir() if (d / "stream.h264").exists())
    if a.limit:
        dirs = dirs[: a.limit]
    run = {}
    for d in dirs:
        r = replay_match(d, make())
        run[r["match_id"]] = r
        print(f"{r['match_id']}: {r['frames']} frames, {len(r['events'])} events, warnings {r['warnings']}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, default=str))
    print(f"wrote {out}")
    if a.baseline:
        report = diff_runs(json.loads(Path(a.baseline).read_text()), run, tol_ms=a.tol_ms)
        print(json.dumps(report, indent=1, default=str))


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
    s.add_argument("--duration", type=float, default=0,
                   help="soak mode: hold the stream this many seconds (e.g. 1800) and count reconnects")
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

    s = sub.add_parser("tower-template", help="add a tower-state reference (alive/rubble) from a saved frame")
    s.add_argument("--frame", required=True)
    s.add_argument("--layout", required=True)
    s.add_argument("--tower", required=True, help="me_left, opp_right, ... or 'all'")
    s.add_argument("--state", required=True, choices=["alive", "rubble"])
    s.add_argument("--out", required=True, help="e.g. configs/domains/clash_royale/towers/phone-a")
    s.set_defaults(fn=cmd_tower_template)

    s = sub.add_parser("synth", help="synthetic detector data in the inference geometry (M3)")
    s.add_argument("--backgrounds", nargs="+", required=True, help="dirs of arena crops")
    s.add_argument("--sprites", required=True, help="segment dir from tools/fetch_katacr.py")
    s.add_argument("--bars", help="HP-bar sprites dir from tools/fetch_katacr.py")
    s.add_argument("--classes", default="models/classes.txt")
    s.add_argument("--out", default="data/synth")
    s.add_argument("--n", type=int, default=5000)
    s.add_argument("--imgsz", type=int, nargs=2, default=[640, 384], metavar=("H", "W"))
    s.add_argument("--val-frac", dest="val_frac", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("convert-katacr", help="KataCR labeled real frames -> our classes (held-out val)")
    s.add_argument("--part2", required=True, help="data/katacr/_repo/images/part2")
    s.add_argument("--classes", default="models/classes.txt")
    s.add_argument("--alias", default="models/katacr_alias.json")
    s.add_argument("--out", default="data/katacr-real")
    s.add_argument("--split", default="val")
    s.add_argument("--empty-keep", dest="empty_keep", type=float, default=0.1)
    s.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    s.set_defaults(fn=cmd_convert_katacr)

    s = sub.add_parser("train-detector", help="YOLO26-n -> static ONNX (needs ultralytics + PyTorch-ROCm)",
                       add_help=False)
    s.add_argument("args", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_train_detector)

    s = sub.add_parser("screen", help="register a saved frame as a reference screen (HOME, IN_BATTLE, ...)")
    s.add_argument("--frame", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_screen)

    s = sub.add_parser("run", help="play matches on a phone")
    for k in ("profile", "layout", "tiles", "templates", "screens", "strategy", "cards"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--arena", help="calibrated arena geometry (defaults to Arena())")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--detector")
    s.add_argument("--classes", help="cross-check against the model's embedded names (refuses on mismatch)")
    s.add_argument("--digits")
    s.add_argument("--towers", help="tower-state templates dir (droid-harness tower-template)")
    s.add_argument("--detect-every", dest="detect_every", type=int, default=1,
                   help="run the detector on every Nth battle frame (units are extrapolated between)")
    s.add_argument("--det-conf", dest="det_conf", type=float, default=0.35)
    s.add_argument("--det-iou", dest="det_iou", type=float, default=0.5)
    s.add_argument("--async-detect", dest="async_detect", action="store_true",
                   help="run the detector on its own thread so HUD reads never wait for inference")
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

    s = sub.add_parser("run-experiment", help="alternate champion/candidate until SPRT decides")
    for k in ("profile", "layout", "tiles", "templates", "screens", "cards"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--champion", required=True, help="champion strategy YAML")
    s.add_argument("--candidate", required=True, help="candidate strategy YAML")
    s.add_argument("--arena", help="calibrated arena geometry")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--detector")
    s.add_argument("--classes", help="cross-check against the model's embedded names (refuses on mismatch)")
    s.add_argument("--digits")
    s.add_argument("--towers", help="tower-state templates dir (droid-harness tower-template)")
    s.add_argument("--detect-every", dest="detect_every", type=int, default=1,
                   help="run the detector on every Nth battle frame (units are extrapolated between)")
    s.add_argument("--det-conf", dest="det_conf", type=float, default=0.35)
    s.add_argument("--det-iou", dest="det_iou", type=float, default=0.5)
    s.add_argument("--async-detect", dest="async_detect", action="store_true",
                   help="run the detector on its own thread so HUD reads never wait for inference")
    s.add_argument("--mode", default="training_camp", choices=["training_camp", "friendly"])
    s.add_argument("--opponent-id", dest="opponent_id")
    s.add_argument("--balance-epoch", dest="balance_epoch")
    s.add_argument("--experiment-id", dest="experiment_id")
    s.add_argument("--title", default="")
    s.add_argument("--hypothesis", default="")
    s.add_argument("--proposer", default="human")
    s.add_argument("--test", default="sprt_h2h", choices=["sprt_h2h", "sprt_paired"])
    s.add_argument("--p0", type=float, default=0.5)
    s.add_argument("--p1", type=float, default=0.58)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--beta", type=float, default=0.05)
    s.add_argument("--max-games", dest="max_games", type=int, default=400)
    s.add_argument("--store", default="ledger/experiments.jsonl")
    s.add_argument("--db", default="data/warehouse.duckdb")
    s.set_defaults(fn=cmd_experiment)

    s = sub.add_parser("replay", help="re-run recorded matches through perception and diff the events (M3)")
    for k in ("layout", "tiles", "templates", "cards"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--data", default="data/matches")
    s.add_argument("--arena")
    s.add_argument("--detector")
    s.add_argument("--classes")
    s.add_argument("--det-conf", dest="det_conf", type=float, default=0.35)
    s.add_argument("--det-iou", dest="det_iou", type=float, default=0.5)
    s.add_argument("--detect-every", dest="detect_every", type=int, default=1)
    s.add_argument("--digits")
    s.add_argument("--towers")
    s.add_argument("--limit", type=int, default=0, help="only the first N matches")
    s.add_argument("--out", required=True, help="events of this run, e.g. data/replay/detector-v2.json")
    s.add_argument("--baseline", help="a previous --out to diff against")
    s.add_argument("--tol-ms", dest="tol_ms", type=int, default=600)
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("replay-diff", help="candidate vs champion on logged states (promotion gate 2)")
    s.add_argument("--champion", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--data", default="data/matches")
    s.add_argument("--cards", required=True)
    s.add_argument("--arena")
    s.add_argument("--expected-pct", dest="expected_pct", type=float, required=True,
                   help="share of decisions the change is expected to alter")
    s.add_argument("--tolerance", type=float, default=3.0)
    s.set_defaults(fn=cmd_replay_diff)

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
