"""droid-harness command line."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
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
    sprites, counts = synth.load_sprites(a.sprites, names, synthesize_missing_sides=a.synthesize_missing_sides)
    print(json.dumps({"backgrounds": len(backgrounds), "sprites": dict(counts)}, indent=1))
    cfg = synth.SynthConfig(imgsz=tuple(a.imgsz), seed=a.seed)
    data = synth.generate_dataset(backgrounds, sprites, Path(a.out), a.n, names, cfg,
                                  bars=synth.load_bars(a.bars), val_frac=a.val_frac)
    print(f"wrote {a.n} images; dataset {data}")


def cmd_convert_katacr(a) -> None:
    from .domains.clash_royale.perception.classes import load_class_names
    from .learn import katacr

    data, counts = katacr.convert(a.part2, a.out, load_class_names(a.classes),
                                  katacr.load_alias(a.alias, a.generic or None),
                                  val_frac=a.val_frac, empty_keep=a.empty_keep, link=not a.copy)
    print(json.dumps(dict(counts), indent=1))
    print(f"dataset {data}")


def cmd_dataset(a) -> None:
    from .learn.datasets import combine

    def parse(items):
        out = []
        for it in items:
            y, _, sp = it.rpartition(":")
            out.append((y, sp)) if y else out.append((it, "train"))
        return out

    out = combine(a.out, parse(a.train), parse(a.val))
    print(f"wrote {out}")


def cmd_eval_detector(a) -> None:
    from .domains.clash_royale.perception.detector import load_detector
    from .learn.evaluate import evaluate, threat_gate

    det = load_detector(a.model, a.classes, conf=a.conf, iou=a.iou,
                        providers=["CPUExecutionProvider"] if a.cpu else None,
                        device="cpu" if a.cpu else None)
    report = evaluate(det, a.data, a.split, a.match_iou, a.limit or None)
    ok, failing = threat_gate(report, a.min_threat_recall, a.min_support)
    report["gate"] = {"pass": ok, "failing": failing, "min_threat_recall": a.min_threat_recall}
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
    for cls, m in report["per_class"].items():
        r = "-" if m["recall"] is None else f"{m['recall']:.2f}"
        pr = "-" if m["precision"] is None else f"{m['precision']:.2f}"
        print(f"{cls:20s} n={m['support']:5d} recall={r:>5} precision={pr:>5}")
    print(json.dumps(report["gate"], indent=1))
    if not ok:
        raise SystemExit(1)


def cmd_frames(a) -> None:
    from .domains.clash_royale.perception.classes import load_class_names
    from .domains.clash_royale.perception.detector import load_detector
    from .domains.clash_royale.perception.grid import Layout
    from .learn.datasets import export_frames

    names = load_class_names(a.classes)
    det = load_detector(a.detector, names) if a.detector else None
    dirs = sorted(d for d in Path(a.data).iterdir() if (d / "stream.h264").exists())
    data, counts = export_frames(dirs, a.out, Layout.load(a.layout), names, every_ms=a.every_ms,
                                 val_frac=a.val_frac, detector=det)
    print(json.dumps(counts), f"dataset {data}")


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


def cmd_recover(a) -> None:
    """Walk any Clash Royale screen back to HOME with deterministic taps.

    The same escalation the live loop uses: normal flow -> calibrated recovery taps ->
    visually found buttons -> BACK -> app restart. Safe to run while the harness is
    stopped, e.g. when a phone was left on a loss screen.
    """
    from .app import load_screens, resolve_device_bundle
    from .core.transport.scrcpy_bridge import ScrcpyClient
    from .domains.clash_royale.perception.grid import Layout, Roi
    from .domains.clash_royale.recovery import walk_to_home

    bundle = resolve_device_bundle(a.device, with_detector=False)
    for key in ("profile", "layout", "screens"):
        if bundle[key] is None or not Path(bundle[key]).exists():
            raise SystemExit(f"missing {key}: {bundle[key]}")
    layout = Layout.load(bundle["layout"])
    raw = yaml.safe_load(Path(bundle["layout"]).read_text())
    regions = [Roi(*r) for r in raw.get("screen_regions", [[0, 0, 1, 0.5], [0, 0.5, 1, 1]])]
    screens = load_screens(bundle["screens"], regions, battle_roi=layout.elixir_bar,
                           battle_hsv=(layout.elixir_hsv_lo, layout.elixir_hsv_hi))

    client = ScrcpyClient(_profile(bundle["profile"]))
    client.start()
    try:
        ok = walk_to_home(client, screens, layout, timeout_s=a.timeout, stable_s=a.stable_s,
                          stuck_after=a.stuck_after, battle_stuck_after=a.battle_stuck_after,
                          restart_after=a.restart_after,
                          dry_run=a.dry_run, log=lambda fmt, *args: print(fmt % args))
    finally:
        client.stop()
    if not ok:
        raise SystemExit(f"did not reach HOME within {a.timeout:.0f}s")


def cmd_train_value(a) -> None:
    from .learn.value_model import train

    m = train([Path(p) for p in a.data], l2=a.l2, every=a.every)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1))
    print(json.dumps({"out": str(out), "matches": m["matches"], "base_rate": m["base_rate"],
                      "val": {k: v for k, v in m["metrics"]["val"].items() if k != "calibration"},
                      "top_weights": m["top_weights"]}, indent=1))


def cmd_check_classes(a) -> None:
    from .domains.clash_royale.cards import CardDB
    from .domains.clash_royale.perception.classes import load_class_names
    from .learn.unseen_eval import check_classes

    _fill_device_defaults(a, with_detector=False)
    rep = check_classes(load_class_names(a.classes), CardDB.load(a.cards))
    print(json.dumps(rep, indent=1))
    if rep["unresolved"]:
        raise SystemExit(1)


def cmd_eval_unseen(a) -> None:
    from .domains.clash_royale.perception.classes import load_class_names
    from .domains.clash_royale.perception.detector import load_detector
    from .learn.unseen_eval import unseen_report

    det = load_detector(a.model, a.classes, conf=a.conf)
    names = load_class_names(a.classes) if a.classes else list(det.class_names)
    rep = unseen_report(det, a.part2, names, alias_path=a.alias, generic_path=a.generic, limit=a.limit)
    print(json.dumps(rep, indent=1))


def cmd_promote_templates(a) -> None:
    from .app import resolve_device_bundle
    from .domains.clash_royale.perception.harvest import promote

    to = a.to or resolve_device_bundle(a.device, with_detector=False)["templates"]
    rep = promote(a.source, to, min_crops=a.min_crops, min_agreement=a.min_agreement, dry_run=a.dry_run)
    print(json.dumps(rep, indent=1))


def cmd_hp_audit(a) -> None:
    """Tower-HP label coverage on recorded video, fixed regions vs the offset search."""
    from .app import load_digits, resolve_device_bundle
    from .domains.clash_royale.hp_audit import audit_match, suggest_rois
    from .domains.clash_royale.perception.grid import Layout

    bundle = resolve_device_bundle(a.device, with_detector=False)
    layout = Layout.load(a.layout or bundle["layout"])
    digits = load_digits(Path(a.digits) if a.digits else bundle["digits"])
    if digits is None:
        raise SystemExit("need 0.png..9.png digit templates (--digits)")
    root = Path(a.data)
    dirs = [root] if (root / "stream.h264").exists() else sorted(d for d in root.iterdir() if (d / "stream.h264").exists())
    if a.limit:
        dirs = dirs[: a.limit]
    reports = []
    for d in dirs:
        r = audit_match(d, layout, digits, every_ms=a.every_ms, crops_out=Path(a.crops) if a.crops else None)
        reports.append(r)
        print(json.dumps(r))
    print(json.dumps({"matches": len(reports), "suggested_tower_hp_rois": suggest_rois(reports, layout)}, indent=1))


def _fill_device_defaults(a, *, with_detector: bool = True, default_async: bool = False) -> None:
    from .app import resolve_device_bundle, resolve_strategy_path

    device = getattr(a, "device", None) or "phone-a"
    use_det = with_detector and not getattr(a, "no_detector", False)
    bundle = resolve_device_bundle(device, with_detector=use_det)
    for k, v in bundle.items():
        if hasattr(a, k) and getattr(a, k) is None and v is not None:
            setattr(a, k, str(v))
    for skey in ("strategy", "champion", "candidate"):
        if hasattr(a, skey) and getattr(a, skey):
            setattr(a, skey, str(resolve_strategy_path(getattr(a, skey))))
    if default_async and getattr(a, "detector", None) and not getattr(a, "sync_detect", False):
        a.async_detect = True


def _build_run_config(a):
    from .app import RunConfig

    non_paths = {"mode", "opponent_id", "balance_epoch", "experiment_id", "arm", "pair_id"}
    return RunConfig(**{
        k: (Path(v) if isinstance(v, str) and k not in non_paths else v)
        for k, v in vars(a).items()
        if k in RunConfig.__dataclass_fields__ and v is not None
    })


def cmd_doctor(a) -> None:
    from .app import run_doctor

    rep = run_doctor(device=a.device, strategy=a.strategy, live=a.live, mode=a.mode, strict=a.strict)
    print(json.dumps(rep, indent=1, default=str))
    if not rep["ok"]:
        raise SystemExit(1)


def cmd_upgrade(a) -> None:
    """Automatically upgrade ready Battle Deck cards & Tower Princess using Gold (never Gems)."""
    from dataclasses import asdict

    from .app import load_screens, resolve_device_bundle
    from .core.transport.scrcpy_bridge import ScrcpyClient
    from .domains.clash_royale.account import load_account
    from .domains.clash_royale.perception.cards import CardMatcher
    from .domains.clash_royale.perception.grid import Layout, Roi
    from .domains.clash_royale.upgrader import run_auto_upgrade

    bundle = resolve_device_bundle(a.device, with_detector=False)
    prof = _profile(str(bundle["profile"]))
    layout_p = Path(bundle["layout"])
    layout = Layout.load(layout_p)
    regions = [Roi(*r) for r in yaml.safe_load(layout_p.read_text()).get("screen_regions",
                                                                         [[0, 0, 1, 0.5], [0, 0.5, 1, 1]])]
    screens = load_screens(Path(bundle["screens"]), regions, battle_roi=layout.elixir_bar,
                           battle_hsv=(layout.elixir_hsv_lo, layout.elixir_hsv_hi))
    matcher = CardMatcher.load(Path(bundle["templates"])) if bundle["templates"] else None
    account = load_account(prof.account)
    client = ScrcpyClient(prof)
    client.start()
    try:
        rep = run_auto_upgrade(
            client, screens, layout, matcher=matcher, account=account,
            max_upgrades=a.max_upgrades, dry_run=a.dry_run,
            log=lambda fmt, *args: print(fmt % args if args else fmt, flush=True),
        )
    finally:
        client.stop()
    print(json.dumps(asdict(rep), indent=1))


def cmd_claim(a) -> None:
    """Claim ready Collection Level / King's Journey reward tiers (never spends Gems)."""
    from dataclasses import asdict

    from .app import load_screens, resolve_device_bundle
    from .core.transport.scrcpy_bridge import ScrcpyClient
    from .domains.clash_royale.perception.grid import Layout, Roi
    from .domains.clash_royale.upgrader import claim_rewards

    bundle = resolve_device_bundle(a.device, with_detector=False)
    prof = _profile(str(bundle["profile"]))
    layout_p = Path(bundle["layout"])
    layout = Layout.load(layout_p)
    regions = [Roi(*r) for r in yaml.safe_load(layout_p.read_text()).get("screen_regions",
                                                                         [[0, 0, 1, 0.5], [0, 0.5, 1, 1]])]
    screens = load_screens(Path(bundle["screens"]), regions, battle_roi=layout.elixir_bar,
                           battle_hsv=(layout.elixir_hsv_lo, layout.elixir_hsv_hi))
    client = ScrcpyClient(prof)
    client.start()
    try:
        rep = claim_rewards(client, screens, layout, max_claims=a.max_claims, dry_run=a.dry_run,
                            log=lambda fmt, *args: print(fmt % args if args else fmt, flush=True))
    finally:
        client.stop()
    print(json.dumps(asdict(rep), indent=1))


def cmd_run(a) -> None:
    from .app import BotRunner

    _fill_device_defaults(a, with_detector=True, default_async=getattr(a, "cmd", "") in ("play", "watch"))
    cfg = _build_run_config(a)
    if not getattr(a, "ui", False):
        BotRunner(cfg).run()
        return
    import threading

    from .ui.launch import run_with_window, start_live_studio
    from .ui.telemetry import TelemetryTap

    bot = BotRunner(cfg, ui=TelemetryTap())
    server = start_live_studio(bot, host=a.ui_host, port=a.ui_port, fps=a.ui_fps)
    stop = threading.Event()
    try:
        run_with_window(lambda: bot.run(stop_when=stop.is_set), server.url, native=a.native,
                        open_browser=a.open, title=server.studio.title, on_close=stop.set)
    finally:
        server.stop()


def cmd_watch(a) -> None:
    """Observe-only live Studio: you play, the bot perceives and shows what it would do."""
    a.ui = True
    a.observe_only = True
    a.matches = a.matches or 10**9
    cmd_run(a)


def cmd_inspect(a) -> None:
    """Replay inspector over recorded matches (decisions.jsonl, plus video when kept)."""
    from .app import resolve_device_bundle
    from .ui.launch import run_with_window, start_replay_studio, wait_forever

    cards = a.cards or resolve_device_bundle(a.device, with_detector=False)["cards"]
    roots = [Path(p) for p in (a.data or ["data"])]
    missing = [str(p) for p in roots if not p.exists()]
    if missing:
        raise SystemExit(f"no such directory: {', '.join(missing)}")
    server = start_replay_studio(roots, host=a.ui_host, port=a.ui_port, cards=Path(cards) if cards else None,
                                 strategies=Path(a.strategies), ledger=Path(a.store), initial_match=a.match,
                                 value=Path(a.value) if a.value else None)
    n = len(server.studio.library.refresh())
    print(f"inspect: {n} recorded matches under {', '.join(map(str, roots))}", flush=True)
    try:
        run_with_window(wait_forever, server.url, native=a.native, open_browser=a.open,
                        title=server.studio.title)
    finally:
        server.stop()


def arm_ctrl_c() -> None:
    """Make Ctrl+C work again in a process that inherited it as "ignored".

    A non-interactive shell starts background commands with SIGINT ignored (POSIX), and an
    ignored signal survives ``exec``: a Studio started with ``droid-harness &`` or ``nohup``
    could not be quit from its page (it stops itself with SIGINT), and every job it started
    ignored the Stop button's Ctrl+C until the job manager killed it with SIGTERM 20 s later.
    """
    if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)


def cmd_studio(a) -> None:
    """The Studio app: buttons, settings, jobs and the guided refine flow (no flags needed)."""
    from .ui.launch import run_app

    arm_ctrl_c()  # quitting from the page is a Ctrl+C this process sends itself
    run_app(host=getattr(a, "host", None), port=getattr(a, "port", None),
            open_browser=False if getattr(a, "no_open", False) else None,
            native=True if getattr(a, "native", False) else None)


def cmd_sim(a) -> None:
    from .app import run_sim

    ui = {"host": a.ui_host, "port": a.ui_port, "fps": a.ui_fps} if a.ui else None
    r = run_sim(a.seconds, Path(a.out), ui=ui)
    print(json.dumps(r, indent=1, default=str))


def cmd_analyze(a) -> None:
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .flywheel import warehouse
    from .flywheel.report import batch_report

    _fill_device_defaults(a, with_detector=False)
    cards = CardDB.load(a.cards)
    arena = Arena.load(a.arena) if getattr(a, "arena", None) else Arena()
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


def cmd_digest(a) -> None:
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .flywheel.digest import build_digest, render_agent_markdown

    _fill_device_defaults(a, with_detector=False)
    cards = CardDB.load(a.cards)
    arena = Arena.load(a.arena) if getattr(a, "arena", None) else Arena()
    vm = None
    if getattr(a, "value", None) and Path(a.value).exists():
        from .learn.value_model import ValueModel

        vm = ValueModel.load(a.value)
    dig = build_digest(a.data, cards, arena, ledger_path=a.store, max_loss_stories=a.max_stories, value_model=vm)
    md = render_agent_markdown(dig)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    json_out = Path(a.json_out) if a.json_out else out.with_suffix(".json")
    json_out.write_text(json.dumps(dig, indent=1, default=str))
    print(f"wrote agent digest {out} ({len(md.encode())} bytes) + {json_out}")


def cmd_retain(a) -> None:
    from .flywheel.retention import retain_dir

    rep = retain_dir(a.data, prune_loss_video=a.prune_loss_video, dry_run=a.dry_run)
    print(json.dumps(rep, indent=1))


def cmd_shadow(a) -> None:
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .domains.clash_royale.policy.strategy import load_strategy
    from .domains.clash_royale.policy.utility import CRPolicy
    from .flywheel.shadow import ShadowRunner
    from .flywheel.tuning.offline_study import load_valued_states

    _fill_device_defaults(a, with_detector=False)
    cards = CardDB.load(a.cards)
    arena = Arena.load(a.arena) if getattr(a, "arena", None) else Arena()
    champ = CRPolicy(cards, arena, load_strategy(a.champion)[1])
    cand = CRPolicy(cards, arena, load_strategy(a.candidate)[1])
    dirs = sorted(d for d in Path(a.data).iterdir() if (d / "decisions.jsonl").exists())
    valued = load_valued_states(dirs, cards, arena)
    sr = ShadowRunner(champ, cand)
    recs = [sr.compare(st, value=val, reset=True) for st, val in valued]
    ranked = sr.report(recs)
    bad_flipped = sum(1 for r in recs if not r.agreement and r.value < -0.5)
    good_broken = sum(1 for r in recs if not r.agreement and r.value > 0.5)
    print(json.dumps({
        "matches": len(dirs),
        "states": len(recs),
        "disagreements": sum(1 for r in recs if not r.agreement),
        "bad_engagements_flipped": bad_flipped,
        "good_engagements_changed": good_broken,
        "by_rule": [{"rule": r, "count": n, "mean_engagement_value": round(v, 3)} for r, n, v in ranked],
    }, indent=1))


def cmd_tune_offline(a) -> None:
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .flywheel.tuning.offline_study import search_offline

    _fill_device_defaults(a, with_detector=False)
    cards = CardDB.load(a.cards)
    arena = Arena.load(a.arena) if getattr(a, "arena", None) else Arena()
    dirs = sorted(d for d in Path(a.data).iterdir() if (d / "decisions.jsonl").exists())
    res = search_offline(
        a.champion, dirs, cards, arena,
        n_trials=a.trials, out_yaml=a.out, seed=a.seed,
        min_diff_pct=a.min_diff_pct, max_diff_pct=a.max_diff_pct,
    )
    print(json.dumps(res, indent=1, default=str))


def cmd_autopilot(a) -> None:
    """Farm -> diagnose -> propose -> gate -> test -> promote -> report, unattended (README §5)."""
    from .flywheel.autopilot import Autopilot, AutopilotConfig, BotPlayer, config_dict

    _fill_device_defaults(a, with_detector=True, default_async=True)
    cfg = AutopilotConfig(
        device=a.device, champion=a.champion, farm_mode=a.farm_mode, farm_matches=a.farm_matches,
        testbed=a.testbed, test=a.test, max_games=a.max_games, futility_pairs=a.futility_pairs,
        data_root=Path(a.data_root), store=Path(a.store), reports_dir=Path(a.reports),
        value_model=Path(a.value), trials=a.trials, min_removed=a.min_removed,
        max_collateral_pct=a.max_collateral, cycles=a.cycles, rest_s=a.rest_s,
        auto_upgrade=not a.no_auto_upgrade, auto_upgrade_every=a.auto_upgrade_every,
        notify_url=a.notify_url, publish_git=a.publish_git, publish_branch=a.publish_branch,
        cards=Path(a.cards) if a.cards else None, arena=Path(a.arena) if a.arena else None)
    tap = None
    if a.ui:
        from .ui.telemetry import TelemetryTap

        tap = TelemetryTap()
    player = BotPlayer(vars(a), store=cfg.store, db=Path(a.db), test=cfg.test, max_games=cfg.max_games,
                       futility_pairs=cfg.futility_pairs, ui=tap,
                       auto_upgrade=cfg.auto_upgrade, auto_upgrade_every=cfg.auto_upgrade_every)
    pilot = Autopilot(cfg, player, log_fn=lambda m: print(f"autopilot: {m}", flush=True))
    print(json.dumps({"autopilot": config_dict(cfg)}, default=str), flush=True)

    def work() -> None:
        for rep in pilot.run():
            print(json.dumps({k: rep.get(k) for k in ("stamp", "champion", "farm", "experiment", "promoted")},
                             default=str), flush=True)

    if tap is None:
        work()
        return
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.perception.grid import Layout, TileMapper
    from .ui.launch import run_with_window, serve_live

    server = serve_live(tap, player.ui_context, layout=Layout.load(a.layout), tiles=TileMapper.load(a.tiles),
                        arena=Arena.load(a.arena) if a.arena else Arena(), replay_roots=[cfg.data_root],
                        title="droid-harness Studio: autopilot", host=a.ui_host, port=a.ui_port, fps=a.ui_fps)
    try:
        run_with_window(work, server.url, native=a.native, open_browser=a.open, title=server.studio.title,
                        on_close=lambda: (cfg.data_root.mkdir(parents=True, exist_ok=True),
                                          (cfg.data_root / "STOP").touch()))
    finally:
        server.stop()


def cmd_tune_targeted(a) -> None:
    """Speed-1, targeted: remove one failure pattern with a one- or two-knob change (todo R5)."""
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .flywheel.tuning.targeted import scan_patterns, search_targeted

    _fill_device_defaults(a, with_detector=False)
    cards = CardDB.load(a.cards)
    arena = Arena.load(a.arena) if getattr(a, "arena", None) else Arena()
    root = Path(a.data)
    dirs = sorted({p.parent for p in root.rglob("decisions.jsonl")})
    if a.scan:
        print(json.dumps(scan_patterns(a.champion, dirs, cards, arena, max_states=a.max_states), indent=1))
        return
    res = search_targeted(a.champion, dirs, cards, arena, pattern=a.pattern, trials=a.trials, seed=a.seed,
                          min_removed=a.min_removed, max_collateral_pct=a.max_collateral, max_states=a.max_states,
                          out_yaml=a.out)
    print(json.dumps(res, indent=1, default=str))
    if a.out and not res.get("candidate_yaml"):
        raise SystemExit(2)  # nothing passed: no candidate written


def cmd_sprt(a) -> None:
    from .flywheel.eval import sprt as S

    if a.wins is None:
        for p1 in (0.55, 0.58, 0.60, 0.65):
            print(f"H1={p1:.2f} (≈{S.elo_from_p(p1):+.0f} Elo): ~{S.expected_games(a.p0, p1):.0f} games")
        return
    r = S.sprt(a.wins, a.losses, a.p0, a.p1, a.alpha, a.beta)
    print(f"{r.decision}  LLR {r.llr:+.3f}  bounds [{r.lower:.3f}, {r.upper:.3f}]")


def cmd_experiment(a) -> None:
    """Persistent-session champion/candidate paired evaluation (M5 + Stage-A dense futility)."""
    from .app import BotRunner
    from .domains.clash_royale.policy.strategy import load_strategy
    from .flywheel.eval.runner import ExperimentConfig, ExperimentRunner
    from .flywheel.ledger import Ledger

    _fill_device_defaults(a, with_detector=True, default_async=getattr(a, "cmd", "") == "experiment")
    _, _, champ_sid = load_strategy(a.champion)
    _, _, cand_sid = load_strategy(a.candidate)
    store = Ledger(a.store)

    # Build ONE persistent BotRunner (loads GPU weights & connects scrcpy ONCE across all pairs).
    a.strategy = a.champion
    cfg = _build_run_config(a)
    cfg.matches = a.max_games * 2
    tap = None
    if getattr(a, "ui", False):
        from .ui.telemetry import TelemetryTap

        tap = TelemetryTap()
    bot = BotRunner(cfg, ui=tap)
    game_version = bot.client_version(retries=3)
    bot.game_version = game_version
    exp_id = a.experiment_id or store.reserve_id(
        title=a.title, hypothesis=a.hypothesis, proposer=a.proposer, champion_id=champ_sid,
        candidate_id=cand_sid, test=a.test, game_version=game_version or "")

    runner = ExperimentRunner(ExperimentConfig(
        experiment_id=exp_id, title=a.title, hypothesis=a.hypothesis, proposer=a.proposer,
        champion_id=champ_sid, candidate_id=cand_sid, test=a.test,
        p0=a.p0, p1=a.p1, alpha=a.alpha, beta=a.beta,
        game_version=game_version or "", balance_epoch=a.balance_epoch or "",
        max_games=a.max_games, futility_min_pairs=a.futility_pairs, dense_tau=a.dense_tau,
        account_levels_id=bot.account.levels_id if bot.account else "", mode=a.mode))

    from .flywheel.eval.live import run_paired

    def _play(stop_when=lambda: False) -> None:
        try:
            run_paired(bot, runner, a.champion, a.candidate, store=a.store, db=a.db, stop=stop_when)
        except BaseException as e:
            if not runner.matches:  # never got going: don't leave a "running" placeholder behind
                store.void_reservation(exp_id, f"never started: {type(e).__name__}: {e}")
            raise

    if tap is None:
        _play()
    else:
        import threading

        from .ui.launch import attach_experiment, run_with_window, start_live_studio

        attach_experiment(bot, runner)
        server = start_live_studio(bot, host=a.ui_host, port=a.ui_port, fps=a.ui_fps,
                                   title=f"droid-harness Studio: {exp_id}")
        stop = threading.Event()
        try:
            run_with_window(lambda: _play(stop.is_set), server.url, native=a.native, open_browser=a.open,
                            title=server.studio.title, on_close=stop.set)
        finally:
            server.stop()
    print(json.dumps({"experiment_id": exp_id, "status": runner.status,
                      "matches": len(runner.matches), "decision": runner.decision.__dict__
                      if runner.decision else None, "void": runner.void_reason,
                      "futility": runner.futility_reason}))


def cmd_replay_diff(a) -> None:
    """Gate 2 of the funnel: candidate vs champion on logged real states, before any match."""
    from .domains.clash_royale.arena import Arena
    from .domains.clash_royale.cards import CardDB
    from .domains.clash_royale.policy.strategy import load_strategy
    from .domains.clash_royale.policy.utility import CRPolicy
    from .flywheel.eval.replay_diff import gate, load_states, replay_diff

    _fill_device_defaults(a, with_detector=False)
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
    from .domains.clash_royale.perception.detector import load_detector, warmup_once
    from .domains.clash_royale.perception.grid import Layout, TileMapper
    from .domains.clash_royale.perception.pipeline import CRPerception
    from .domains.clash_royale.perception.towers import TowerStateReader
    from .domains.clash_royale.replay import diff_runs, replay_match

    _fill_device_defaults(a, with_detector=True)
    cards = CardDB.load(a.cards)
    detector = None
    if a.detector:
        detector = load_detector(a.detector, a.classes, conf=a.det_conf, iou=a.det_iou)
        warmup_once(detector, 2)

    def make():
        lay = Layout.load(a.layout)
        return CRPerception(layout=lay, tiles=TileMapper.load(a.tiles), cards=cards,
                            arena=Arena.load(a.arena) if a.arena else Arena(),
                            matcher=CardMatcher.load(a.templates), digits=load_digits(a.digits),
                            detector=detector, detect_every=a.detect_every,
                            tower_states=TowerStateReader.load(a.towers, rois=lay.towers) if a.towers else None)

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


def cmd_advise(a) -> None:
    """L2 advisor: replay recorded states through the LLM, or print prompts offline."""
    import os

    from .core.runtime.advisor import POSTURES, Advisor, build_prompt
    from .core.runtime.intent_bus import IntentBus

    allowed = {p.strip().upper() for p in (a.allowed or "").split(",") if p.strip()}
    if unknown := allowed - set(POSTURES):
        raise SystemExit(f"advise: unknown postures: {sorted(unknown)} (choose from {list(POSTURES)})")
    states = _iter_advise_states(a.data) if a.data else []
    if a.dry_run:
        if not states:
            raise SystemExit("advise --dry-run needs --data <match dir | decisions.jsonl | matches root>")
        for label, st in states:  # no network: print the prompts for review
            print(f"--- {label} ---")
            print(build_prompt(st))
        return
    api_key = a.api_key or os.environ.get("DROID_ADVISOR_KEY")
    if not a.api_base or not api_key:
        raise SystemExit("advise needs --api-base and --api-key (or DROID_ADVISOR_KEY env), or use --dry-run")
    if not states:
        raise SystemExit("advise needs --data <match dir | decisions.jsonl | matches root> to replay")
    bus = IntentBus(max_age_s=2.0 * a.cadence)
    adv = Advisor(bus, a.api_base, api_key, a.model, cadence_s=a.cadence, allowed=allowed or set(POSTURES))
    for label, st in states:
        intent = adv.poll(st)
        print(json.dumps({"state": label, "posture": intent.posture if intent else None,
                          "note": intent.note if intent else ""}))


def _iter_advise_states(data: str) -> list[tuple[str, dict]]:
    """Unwrap {state} dicts from decisions.jsonl under --data (file, match dir, or root)."""
    p = Path(data)
    if p.is_file():
        files = [p]
    elif (p / "decisions.jsonl").exists():
        files = [p / "decisions.jsonl"]
    else:
        files = sorted(p.glob("*/decisions.jsonl"))
    out = []
    for f in files:
        for i, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            st = rec.get("state") if isinstance(rec, dict) else None
            if isinstance(st, dict):
                out.append((f"{f.parent.name}:{i}", st))
            elif isinstance(rec, dict) and "t_ms" in rec:
                out.append((f"{f.parent.name}:{i}", rec))
    return out


def _add_ui_args(s, *, always_on: bool = False) -> None:
    if not always_on:
        s.add_argument("--ui", action="store_true",
                       help="serve the live Studio (annotated phone view + strategy numbers) while playing")
    s.add_argument("--ui-host", dest="ui_host", default="127.0.0.1",
                   help="bind address (default localhost; use an SSH tunnel for remote viewing)")
    s.add_argument("--ui-port", dest="ui_port", type=int, default=8765)
    s.add_argument("--ui-fps", dest="ui_fps", type=float, default=15.0,
                   help="snapshot rate of the live view; the bot loop never waits on it")
    s.add_argument("--native", action="store_true", help="open in a desktop window (needs the [gui] extra)")
    s.add_argument("--open", action="store_true", help="open the Studio in the default browser")


def _add_runtime_args(s, *, default_mode: str = "training_camp", require_strategy: bool = False,
                      default_strategy: str | None = None) -> None:
    s.add_argument("--device", default="phone-a",
                   help="device preset under configs/ (auto-resolves profile, rois, tiles, templates, digits, detector)")
    for k in ("profile", "layout", "tiles", "templates", "screens", "cards"):
        s.add_argument(f"--{k}", default=None)
    if require_strategy or default_strategy is not None:
        s.add_argument("--strategy", default=default_strategy, required=require_strategy and default_strategy is None,
                       help="strategy path or short alias (e.g. 'v6', 'v4', 'ladder-vision-v6')")
    s.add_argument("--arena", default=None, help="calibrated arena geometry")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--detector", default=None)
    s.add_argument("--no-detector", dest="no_detector", action="store_true",
                   help="disable auto-loading models/yolo26n-cr.pt (for blind strategies)")
    s.add_argument("--classes", default=None, help="cross-check against the model's embedded names")
    s.add_argument("--digits", default=None, help="tower-HP & clock digits dir")
    s.add_argument("--towers", default=None, help="tower-state templates dir")
    s.add_argument("--detect-every", dest="detect_every", type=int, default=1)
    s.add_argument("--det-conf", dest="det_conf", type=float, default=0.35)
    s.add_argument("--det-iou", dest="det_iou", type=float, default=0.5)
    s.add_argument("--async-detect", dest="async_detect", action="store_true",
                   help="run the detector on its own thread so HUD reads never wait for inference")
    s.add_argument("--sync-detect", dest="sync_detect", action="store_true",
                   help="force synchronous detection on `play`/`experiment`")
    s.add_argument("--prune-win-video", dest="prune_win_video", action="store_true",
                   help="delete raw stream.h264 on clean completed wins once the match is written")
    s.add_argument("--reconnect-deadline", dest="reconnect_deadline_s", type=float, default=None,
                   help="seconds to keep retrying a phone that dropped off before ending the run (default 300)")
    s.add_argument("--mode", default=default_mode,
                   choices=["training_camp", "ladder", "practice", "friendly", "friendly_host", "friendly_guest"])
    s.add_argument("--opponent-id", dest="opponent_id")
    s.add_argument("--balance-epoch", dest="balance_epoch", default="2026-11")
    s.add_argument("--harvest-templates", dest="harvest_templates", default=None,
                   help="save hand crops of cards the matcher can't read yet (then: promote-templates)")
    s.add_argument("--auto-upgrade", dest="auto_upgrade", action="store_true",
                   help="upgrade ready Battle Deck cards & Tower Princess between farm matches (gold only)")
    s.add_argument("--auto-upgrade-every", dest="auto_upgrade_every", type=int, default=4,
                   help="matches between mid-batch upgrade passes (0 = start of batch only)")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    from .ui.jobs import JOB_ENV

    if os.environ.get(JOB_ENV):
        arm_ctrl_c()  # a Studio job: Stop sends Ctrl+C, which must not be inherited as "ignored"
    if raw_argv and raw_argv[0] == "train-detector":
        from .learn.train_detector import main as train_main

        train_main(raw_argv[1:])
        return

    ap = argparse.ArgumentParser(prog="droid-harness",
                                 description="With no command, opens the Studio app (same as `droid-harness studio`).")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("studio", help="the Studio app: live view, buttons, settings and guided Speed 1 and 2 "
                                      "(also what `droid-harness` alone opens)")
    s.add_argument("--host", default=None, help="listen address (default: the app's setting, 127.0.0.1)")
    s.add_argument("--port", type=int, default=None, help="port (default: the app's setting, 8765)")
    s.add_argument("--no-open", dest="no_open", action="store_true", help="don't open a browser")
    s.add_argument("--native", action="store_true", help="open in a desktop window (needs the [gui] extra)")
    s.set_defaults(fn=cmd_studio)

    s = sub.add_parser("doctor", help="pre-flight check of calibration, templates, digits, GPU detector & ADB")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--strategy", default="champion", help="strategy to check (default: configs/strategies/CHAMPION)")
    s.add_argument("--live", action="store_true", help="also probe connected ADB device state")
    s.add_argument("--mode", default=None, help="also check the buttons/screens a mode's menu flow needs")
    s.add_argument("--strict", action="store_true", help="exit 1 on warnings too (level/tower-damage mismatches)")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("recover", help="click any Clash Royale screen back to HOME (stuck loss/popup recovery)")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--timeout", type=float, default=120.0)
    s.add_argument("--stuck-after", dest="stuck_after", type=float, default=3.0,
                   help="seconds on an unchanged screen before recovery taps start")
    s.add_argument("--battle-stuck-after", dest="battle_stuck_after", type=float, default=20.0,
                   help="seconds of frozen clock+elixir before an IN_BATTLE screen is treated as stuck")
    s.add_argument("--stable-s", dest="stable_s", type=float, default=2.0,
                   help="seconds HOME must hold before declaring success")
    s.add_argument("--restart-after", dest="restart_after", type=float, default=45.0,
                   help="force-stop + relaunch after this many seconds stuck")
    s.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print the taps and keycodes without sending them")
    s.set_defaults(fn=cmd_recover)

    s = sub.add_parser("upgrade", help="upgrade ready Battle Deck cards & Tower Princess using Gold (never Gems)")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--max-upgrades", dest="max_upgrades", type=int, default=10)
    s.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="inspect ready upgrades without tapping Confirm")
    s.set_defaults(fn=cmd_upgrade)

    s = sub.add_parser("claim", help="claim ready Collection Level / King's Journey reward tiers")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--max-claims", dest="max_claims", type=int, default=12)
    s.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="open the panel and report what is claimable without tapping Claim")
    s.set_defaults(fn=cmd_claim)

    s = sub.add_parser("play", help="one-command live play with auto-resolved device, digits & async GPU detector")
    _add_runtime_args(s, default_mode="ladder", default_strategy="champion")
    s.add_argument("--matches", type=int, default=10)
    s.add_argument("--join-battle", dest="join_battle", action="store_true",
                   help="a battle is already on the phone: play it with the selected strategy. No menu flow, "
                        "upgrade pass or recovery tap runs before it is recognised; with --matches 1 the run "
                        "ends on HOME when the battle does")
    s.add_argument("--join-wait", dest="join_wait_s", type=float, default=None,
                   help="seconds to wait for the battle to appear before giving up (default 45; nothing is tapped)")
    s.add_argument("--experiment-id", dest="experiment_id")
    s.add_argument("--arm")
    _add_ui_args(s)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("watch", help="live Studio in observe-only mode: you play, the bot shows what it sees "
                                     "and would do (never taps)")
    _add_runtime_args(s, default_mode="ladder", default_strategy="champion")
    s.set_defaults(data="data/watch")
    s.add_argument("--matches", type=int, default=0, help="stop after N matches (default: until Ctrl+C)")
    _add_ui_args(s, always_on=True)
    s.set_defaults(fn=cmd_watch)

    s = sub.add_parser("inspect", help="replay inspector: scrub recorded matches (state, candidates, video)")
    s.add_argument("data", nargs="*", help="match dirs or folders of matches (default: data)")
    s.add_argument("--match", default=None, help="open this match id first (e.g. matches-exp-v4-v6/32dc3e473260)")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--cards", default=None, help="card DB for costs (default: the device bundle's cards.json)")
    s.add_argument("--strategies", default="configs/strategies", help="resolves strategy ids to names/params")
    s.add_argument("--value", default="models/value.json", help="win-probability model (train-value) for the tape")
    s.add_argument("--store", default="ledger/experiments.jsonl")
    _add_ui_args(s, always_on=True)
    s.set_defaults(fn=cmd_inspect)

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
    s.add_argument("--synthesize-missing-sides", dest="synthesize_missing_sides", action="store_true",
                   help="synthesize missing <unit>_me sprites from <unit>_opp when real _me sprites are scarce")
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("convert-katacr", help="KataCR labeled real frames -> our classes (held-out val)")
    s.add_argument("--part2", required=True, help="data/katacr/_repo/images/part2")
    s.add_argument("--classes", default="models/classes.txt")
    s.add_argument("--alias", default="models/katacr_alias.json")
    s.add_argument("--out", default="data/katacr-real")
    s.add_argument("--generic", default="models/katacr_generic.json", help="catch-alls; '' for named only")
    s.add_argument("--val-frac", dest="val_frac", type=float, default=0.2,
                   help="share of source *sequences* (not frames) held out for validation")
    s.add_argument("--empty-keep", dest="empty_keep", type=float, default=0.1)
    s.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    s.set_defaults(fn=cmd_convert_katacr)

    s = sub.add_parser("dataset", help="combine datasets into one data.yaml (refuses train/val overlap)")
    s.add_argument("--train", nargs="+", required=True, help="data.yaml[:split] ..., e.g. data/synth/data.yaml:train")
    s.add_argument("--val", nargs="+", required=True, help="e.g. data/katacr-real/data.yaml:val data/own/data.yaml:val")
    s.add_argument("--out", default="data/train.yaml")
    s.set_defaults(fn=cmd_dataset)

    s = sub.add_parser("eval-detector", help="per-class recall/precision on held-out real frames; gate on threats")
    s.add_argument("--model", required=True)
    s.add_argument("--classes")
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="val")
    s.add_argument("--conf", type=float, default=0.35)
    s.add_argument("--iou", type=float, default=0.5, help="NMS IoU")
    s.add_argument("--match-iou", dest="match_iou", type=float, default=0.5, help="IoU for a correct detection")
    s.add_argument("--min-threat-recall", dest="min_threat_recall", type=float, default=0.85)
    s.add_argument("--min-support", dest="min_support", type=int, default=20)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--cpu", action="store_true")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_eval_detector)

    s = sub.add_parser("frames", help="arena crops from recorded matches for labeling, split by match")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--layout", required=True)
    s.add_argument("--classes", default="models/classes.txt")
    s.add_argument("--detector", help="pre-label with this model (fix the labels, retrain, repeat)")
    s.add_argument("--every-ms", dest="every_ms", type=int, default=1000)
    s.add_argument("--val-frac", dest="val_frac", type=float, default=0.2)
    s.add_argument("--out", default="data/own")
    s.set_defaults(fn=cmd_frames)

    s = sub.add_parser("train-detector", help="YOLO26-n -> static ONNX (needs ultralytics + PyTorch-ROCm)",
                       add_help=False)
    s.add_argument("args", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_train_detector)

    s = sub.add_parser("screen", help="register a saved frame as a reference screen (HOME, IN_BATTLE, ...)")
    s.add_argument("--frame", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_screen)

    s = sub.add_parser("run", help="play matches on a phone (supports --device auto-resolution)")
    _add_runtime_args(s, default_mode="training_camp", require_strategy=True)
    s.add_argument("--matches", type=int, default=10)
    s.add_argument("--experiment-id", dest="experiment_id")
    s.add_argument("--arm")
    _add_ui_args(s)
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("sim", help="closed-loop demo with no phone (synthetic scene)")
    s.add_argument("--seconds", type=float, default=10)
    s.add_argument("--out", default="data/sim")
    _add_ui_args(s)
    s.set_defaults(fn=cmd_sim)

    s = sub.add_parser("analyze", help="engagements, triage, report, and warehouse ingest")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--cards", default=None)
    s.add_argument("--arena", default=None)
    s.add_argument("--db", default="data/warehouse.duckdb")
    s.add_argument("--game-version", dest="game_version")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("digest", help="generate <30 KB deterministic Agent Handoff Artifact (Speed-3)")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--cards", default=None)
    s.add_argument("--arena", default=None)
    s.add_argument("--store", default="ledger/experiments.jsonl")
    s.add_argument("--max-stories", dest="max_stories", type=int, default=8)
    s.add_argument("--out", default="data/agent_digest.md")
    s.add_argument("--json-out", dest="json_out", default=None)
    s.add_argument("--value", default="models/value.json", help="win-probability model (adds swing moments)")
    s.set_defaults(fn=cmd_digest)

    s = sub.add_parser("retain", help="compact clean wins (delete stream.h264) while preserving loss keyframes")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--prune-loss-video", dest="prune_loss_video", action="store_true")
    s.add_argument("--dry-run", dest="dry_run", action="store_true")
    s.set_defaults(fn=cmd_retain)

    s = sub.add_parser("shadow", help="engagement-weighted counterfactual disagreement report")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--champion", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--data", default="data/matches")
    s.add_argument("--cards", default=None)
    s.add_argument("--arena", default=None)
    s.set_defaults(fn=cmd_shadow)

    s = sub.add_parser("tune-offline", help="Speed-1 CPU counterfactual parameter search over logged matches")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--champion", default="champion", help="base strategy YAML or alias (default: CHAMPION)")
    s.add_argument("--data", default="data/matches")
    s.add_argument("--cards", default=None)
    s.add_argument("--arena", default=None)
    s.add_argument("--trials", type=int, default=80)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-diff-pct", dest="min_diff_pct", type=float, default=1.0)
    s.add_argument("--max-diff-pct", dest="max_diff_pct", type=float, default=18.0)
    s.add_argument("--out", default=None, help="write single best candidate YAML here")
    s.set_defaults(fn=cmd_tune_offline)

    s = sub.add_parser("autopilot", help="unattended farm -> diagnose -> propose -> test -> promote -> report loop")
    _add_runtime_args(s, default_mode="ladder")
    s.add_argument("--champion", default=None, help="starting champion if configs/strategies/CHAMPION is missing")
    s.add_argument("--farm-mode", dest="farm_mode", default="ladder")
    s.add_argument("--farm-matches", dest="farm_matches", type=int, default=20)
    s.add_argument("--testbed", default="ladder", help="mode for experiments: training_camp | practice | ladder")
    s.add_argument("--test", default="pairs", choices=["pairs", "dense"])
    s.add_argument("--max-games", dest="max_games", type=int, default=60)
    s.add_argument("--futility-pairs", dest="futility_pairs", type=int, default=8)
    s.add_argument("--data-root", dest="data_root", default="data/autopilot")
    s.add_argument("--store", default="ledger/experiments.jsonl")
    s.add_argument("--db", default="data/warehouse.duckdb")
    s.add_argument("--reports", default="reports")
    s.add_argument("--value", default="models/value.json")
    s.add_argument("--trials", type=int, default=40)
    s.add_argument("--min-removed", dest="min_removed", type=float, default=0.5)
    s.add_argument("--max-collateral", dest="max_collateral", type=float, default=3.0)
    s.add_argument("--cycles", type=int, default=1, help="0 = run until data-root/STOP exists")
    s.add_argument("--rest-s", dest="rest_s", type=float, default=60.0)
    s.add_argument("--no-auto-upgrade", dest="no_auto_upgrade", action="store_true",
                   help="disable automatic Battle Deck & Tower Princess upgrades during farm batches")
    s.add_argument("--notify-url", dest="notify_url", default=None,
                   help="POST a one-line summary here each cycle (e.g. https://ntfy.sh/<topic>)")
    s.add_argument("--publish-git", dest="publish_git", action="store_true",
                   help="commit reports to --publish-branch and push (for the weekly agent job)")
    s.add_argument("--publish-branch", dest="publish_branch", default="autopilot/reports")
    _add_ui_args(s)
    s.set_defaults(fn=cmd_autopilot)

    s = sub.add_parser("tune-targeted", help="Speed-1: remove one failure pattern with a 1-2 knob change (R5)")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--champion", default="champion", help="strategy to fix (default: CHAMPION)")
    s.add_argument("--data", default="data/matches", help="searched recursively for decisions.jsonl")
    s.add_argument("--cards", default=None)
    s.add_argument("--arena", default=None)
    s.add_argument("--pattern", default="auto", help="auto (the champion's most frequent) or one of: opening_throw, "
                   "bad_counter_timing, over_defend_bleed, wincon_starved, spell_inefficiency")
    s.add_argument("--scan", action="store_true", help="only count the champion's hits per pattern")
    s.add_argument("--trials", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-removed", dest="min_removed", type=float, default=0.5)
    s.add_argument("--max-collateral", dest="max_collateral", type=float, default=3.0,
                   help="max %% of other states whose decision may change")
    s.add_argument("--max-states", dest="max_states", type=int, default=6000)
    s.add_argument("--out", default=None, help="write the best passing candidate here (a new file)")
    s.set_defaults(fn=cmd_tune_targeted)

    s = sub.add_parser("sprt", help="SPRT decision, or the sample-size table with no results")
    s.add_argument("--wins", type=int)
    s.add_argument("--losses", type=int)
    s.add_argument("--p0", type=float, default=0.5)
    s.add_argument("--p1", type=float, default=0.58)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--beta", type=float, default=0.05)
    s.set_defaults(fn=cmd_sprt)

    for exp_cmd, def_mode, def_fut in (("run-experiment", "training_camp", 0), ("experiment", "ladder", 12)):
        s = sub.add_parser(exp_cmd, help="persistent-session champion/candidate paired evaluation")
        _add_runtime_args(s, default_mode=def_mode, require_strategy=False)
        s.add_argument("--champion", required=True, help="champion strategy YAML or alias (e.g. 'v4')")
        s.add_argument("--candidate", required=True, help="candidate strategy YAML or alias (e.g. 'v6')")
        s.add_argument("--experiment-id", dest="experiment_id")
        s.add_argument("--title", default="")
        s.add_argument("--hypothesis", default="")
        s.add_argument("--proposer", default="human")
        s.add_argument("--test", default="pairs", choices=["pairs", "dense", "h2h", "fixed", "sprt_h2h", "sprt_paired"],
                       help="pairs: SPRT on discordant pairs; dense: always-valid test on the paired dense margin")
        s.add_argument("--dense-tau", dest="dense_tau", type=float, default=0.5,
                       help="dense test: the paired margin difference you care about (mixture scale)")
        s.add_argument("--p0", type=float, default=0.5)
        s.add_argument("--p1", type=float, default=0.58)
        s.add_argument("--alpha", type=float, default=0.05)
        s.add_argument("--beta", type=float, default=0.05)
        s.add_argument("--max-games", dest="max_games", type=int, default=400)
        s.add_argument("--futility-pairs", dest="futility_pairs", type=int, default=def_fut,
                       help="Stage-A early futility stop when dense-margin UCB95 < 0 after N pairs (0 = off)")
        s.add_argument("--store", default="ledger/experiments.jsonl")
        s.add_argument("--db", default="data/warehouse.duckdb")
        _add_ui_args(s)
        s.set_defaults(fn=cmd_experiment)

    s = sub.add_parser("replay", help="re-run recorded matches through perception and diff the events (M3)")
    s.add_argument("--device", default="phone-a")
    for k in ("layout", "tiles", "templates", "cards"):
        s.add_argument(f"--{k}", default=None)
    s.add_argument("--data", default="data/matches")
    s.add_argument("--arena", default=None)
    s.add_argument("--detector", default=None)
    s.add_argument("--no-detector", dest="no_detector", action="store_true")
    s.add_argument("--classes", default=None)
    s.add_argument("--det-conf", dest="det_conf", type=float, default=0.35)
    s.add_argument("--det-iou", dest="det_iou", type=float, default=0.5)
    s.add_argument("--detect-every", dest="detect_every", type=int, default=1)
    s.add_argument("--digits", default=None)
    s.add_argument("--towers", default=None)
    s.add_argument("--limit", type=int, default=0, help="only the first N matches")
    s.add_argument("--out", required=True, help="events of this run, e.g. data/replay/detector-v2.json")
    s.add_argument("--baseline", help="a previous --out to diff against")
    s.add_argument("--tol-ms", dest="tol_ms", type=int, default=600)
    s.set_defaults(fn=cmd_replay)

    s = sub.add_parser("train-value", help="win-probability model from finished matches (Studio line, digest swings)")
    s.add_argument("data", nargs="*", default=["data"], help="folders searched for matches")
    s.add_argument("--out", default="models/value.json")
    s.add_argument("--l2", type=float, default=1.0)
    s.add_argument("--every", type=int, default=4, help="use every Nth logged state")
    s.set_defaults(fn=cmd_train_value)

    s = sub.add_parser("check-classes", help="detector classes that don't map to a card in cards.json")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--classes", default="models/classes.txt")
    s.add_argument("--cards", default=None)
    s.set_defaults(fn=cmd_check_classes)

    s = sub.add_parser("eval-unseen", help="what the detector does with cards it has no name for (R6)")
    s.add_argument("--model", required=True)
    s.add_argument("--classes", default=None)
    s.add_argument("--part2", required=True, help="KataCR labeled frames: data/katacr/_repo/images/part2")
    s.add_argument("--alias", default="models/katacr_alias.json", help="named cards of THIS model")
    s.add_argument("--generic", default="models/katacr_generic.json")
    s.add_argument("--conf", type=float, default=0.35)
    s.add_argument("--limit", type=int, default=0, help="stop after N frames with unseen units")
    s.set_defaults(fn=cmd_eval_unseen)

    s = sub.add_parser("promote-templates", help="turn harvested hand crops into card templates (after a look)")
    s.add_argument("--from", dest="source", default="data/harvest")
    s.add_argument("--to", default=None, help="templates dir (default: the device bundle's)")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--min-crops", dest="min_crops", type=int, default=3)
    s.add_argument("--min-agreement", dest="min_agreement", type=float, default=0.6)
    s.add_argument("--dry-run", dest="dry_run", action="store_true")
    s.set_defaults(fn=cmd_promote_templates)

    s = sub.add_parser("hp-audit", help="tower-HP label coverage on recorded video (fixed regions vs search)")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--data", default="data/matches", help="a match dir or a folder of matches with stream.h264")
    s.add_argument("--layout", default=None)
    s.add_argument("--digits", default=None)
    s.add_argument("--every-ms", dest="every_ms", type=int, default=1000)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--crops", default=None, help="save a few matched label crops here")
    s.set_defaults(fn=cmd_hp_audit)

    s = sub.add_parser("replay-diff", help="candidate vs champion on logged states (promotion gate 2)")
    s.add_argument("--device", default="phone-a")
    s.add_argument("--champion", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--data", default="data/matches")
    s.add_argument("--cards", default=None)
    s.add_argument("--arena", default=None)
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

    s = sub.add_parser("advise", help="L2 advisor: replay states via an OpenAI-compatible LLM (or --dry-run)")
    s.add_argument("--api-base", dest="api_base", default=None, help="e.g. http://localhost:11434/v1")
    s.add_argument("--api-key", dest="api_key", default=None, help="or DROID_ADVISOR_KEY env (not needed --dry-run)")
    s.add_argument("--model", default="deepseek-v4.1-flash")
    s.add_argument("--cadence", type=float, default=4.0, help="seconds between polls; intent TTL = 2x this")
    s.add_argument("--data", default=None,
                   help="match dir, decisions.jsonl file, or matches root to replay offline")
    s.add_argument("--allowed", default="OPENING,BALANCED,DEFEND_LEAD,PRESS,SPELL_CYCLE",
                   help="comma-separated posture allowlist (no strategy file here, so defaults to all)")
    s.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="print prompts only, no network — fully offline")
    s.set_defaults(fn=cmd_advise)

    a = ap.parse_args(raw_argv)
    if a.cmd is None:
        cmd_studio(a)
        return
    from .app import RunAborted
    from .core.transport.device_lock import DeviceBusy

    try:
        a.fn(a)
    except (RunAborted, DeviceBusy) as e:  # a clear reason, not a traceback
        print(f"droid-harness {a.cmd}: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    except KeyboardInterrupt:  # Ctrl+C, or the Studio's Stop: the run's own cleanup has already run
        print(f"droid-harness {a.cmd}: stopped (Ctrl+C)", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main(sys.argv[1:])
