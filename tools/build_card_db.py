#!/usr/bin/env python3
"""Build ``configs/domains/clash_royale/cards.json`` from a local clone of RoyaleAPI/cr-api-data.

    git clone --depth 1 https://github.com/RoyaleAPI/cr-api-data ~/src/cr-api-data
    python tools/build_card_db.py ~/src/cr-api-data

Caveats (also written into the output's ``meta``):
* That dataset was last updated in Oct 2023 and carries no license, so the output is
  gitignored rather than committed. Cards and balance changes since then are missing or stale.
* Unit conversions are assumptions to verify: speed is treated as tiles/minute (60 = 1
  tile/s), ranges and radii as 1/1000 tile, times as ms. Projectile speeds use the same
  speed convention — check spell travel against recorded video before trusting it.
* Spell damage to towers depends on your card levels: set it per spell in your strategy file
  (``tier0.spell_tower_damage``) rather than relying on level-1 numbers here.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
from pathlib import Path


def snake(s: str) -> str:
    s = s.replace("-", "_").replace(" ", "_").replace(".", "")
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", s)
    return re.sub(r"_+", "_", s).lower().strip("_")


def load(base: Path, name: str):
    return json.loads((base / "docs" / "json" / name).read_text())


def build(repo: Path) -> dict:
    cards = load(repo, "cards.json")
    chars = {c["name"]: c for c in load(repo, "cards_stats_characters.json")}
    troops = {t["name"]: t for t in load(repo, "cards_stats_troop.json")}
    buildings = {b["name"]: b for b in load(repo, "cards_stats_building.json")}
    spells = {s["name"]: s for s in load(repo, "cards_stats_spell.json")}
    projectiles = {p["name"]: p for p in load(repo, "cards_stats_projectile.json")}

    out, skipped = {}, []
    for c in cards:
        key, sc_key, kind = snake(c["key"]), c.get("sc_key") or c["name"].replace(" ", ""), c["type"].lower()
        rec = {"name": c["name"], "cost": int(c["elixir"]), "kind": kind}
        if kind == "troop" and sc_key in troops:
            t = troops[sc_key]
            ch = chars.get(t.get("summon_character") or "")
            if ch is None:
                skipped.append(key)
                continue
            units = max(1, int(t.get("summon_number") or 0))
            classes = [snake(t["summon_character"])]
            if t.get("summon_character_second"):
                classes.append(snake(t["summon_character_second"]))
                units += int(t.get("summon_character_second_count") or 0)
            rec.update(
                deploy_time_s=(ch.get("deploy_time") or 1000) / 1000, units=units,
                speed_tiles_s=(ch.get("speed") or 0) / 60, range_tiles=(ch.get("range") or 0) / 1000,
                hitpoints=int(ch.get("hitpoints") or 0), targets_air=bool(ch.get("attacks_air")),
                targets_ground=bool(ch.get("attacks_ground", True)),
                buildings_only=bool(ch.get("target_only_buildings")), flying=bool(ch.get("flying_height")),
                unit_classes=classes)
        elif kind == "building":
            b = buildings.get(sc_key) or chars.get(sc_key)
            if b is None:
                skipped.append(key)
                continue
            rec.update(deploy_time_s=(b.get("deploy_time") or 1000) / 1000, range_tiles=(b.get("range") or 0) / 1000,
                       hitpoints=int(b.get("hitpoints") or 0), targets_air=bool(b.get("attacks_air")),
                       unit_classes=[snake(sc_key)])
        elif kind == "spell":
            s = spells.get(sc_key, {})
            names = [s.get("projectile"), f"{sc_key}Spell", f"{sc_key}ProjectileRolling", f"{sc_key}Projectile"]
            p = next((projectiles[n] for n in names if n and n in projectiles), {})
            ct = p.get("crown_tower_damage_percent", s.get("crown_tower_damage_percent")) or 0
            rec.update(
                damage=int(p.get("damage") or s.get("damage") or 0),
                # stored as a reduction (e.g. -70 means towers take 30%)
                ct_damage_pct=float(100 + ct if ct <= 0 else ct),
                projectile_speed_tiles_s=((p.get("speed") or 0) / 60) or None,
                radius_tiles=(p.get("radius") or s.get("radius") or 0) / 1000,
                unit_classes=[snake(sc_key) + "_effect"])
        else:
            skipped.append(key)
            continue
        out[key] = rec

    try:
        commit = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H %cs"], capture_output=True,
                                text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    return {"meta": {"source": "RoyaleAPI/cr-api-data (local clone)", "source_commit": commit,
                     "built": dt.date.today().isoformat(), "skipped": skipped,
                     "warning": "Stale snapshot; unit conversions are assumptions. See tools/build_card_db.py."},
            "cards": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", type=Path, help="path to a clone of RoyaleAPI/cr-api-data")
    ap.add_argument("-o", "--out", type=Path, default=Path("configs/domains/clash_royale/cards.json"))
    args = ap.parse_args()
    db = build(args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(db, indent=1, sort_keys=True))
    print(f"wrote {len(db['cards'])} cards to {args.out} (skipped {len(db['meta']['skipped'])})")


if __name__ == "__main__":
    main()
