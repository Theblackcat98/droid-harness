#!/usr/bin/env python3
"""Build models/katacr_generic.json: every KataCR unit we don't detect by name -> a catch-all.

A closed-set detector trained only on our deck treats every other card as background, so
an enemy Hog Rider is invisible. Mapping all other KataCR units to ``other_ground`` /
``other_air`` (side-aware like everything else) lets the bot at least see *something*
coming and defend with generic counters.

Reads KataCR's ``label_list.py`` (MIT; ``tools/fetch_katacr.py --label-list-only``) with
``ast`` (never imports it). Evolutions of cards we detect by name map to that card.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_lists(path: Path) -> dict[str, list[str]]:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = [ast.literal_eval(e) for e in node.value.elts]
            except ValueError:
                pass
    return out


# KataCR's ground/flying split follows its own drawing layers, not the game: these fly in
# the game (only air-targeting cards can hit them), whatever list KataCR put them in.
AIR = {"balloon", "lava-hound", "lava-pup", "electro-dragon", "baby-dragon", "inferno-dragon",
       "mega-minion", "minion", "bat", "bat-evolution", "skeleton-dragon", "phoenix-big",
       "phoenix-small", "flying-machine", "skeleton-barrel"}
# Spell effects and rolling spells KataCR lists as ground units: not units to defend against.
NOT_UNITS = {"barbarian-barrel", "the-log", "royal-delivery"}


def build(lists: dict[str, list[str]], alias: dict[str, str]) -> dict[str, object]:
    not_units = set(lists.get("spell_unit_list", [])) | set(lists.get("object_unit_list", [])) \
        | set(lists.get("tower_unit_list", [])) | set(lists.get("other_unit_list", [])) | NOT_UNITS
    evolution_of = {f"{k}-evolution": v for k, v in alias.items()}
    named = {k: v for k, v in evolution_of.items()
             if k in set(lists.get("ground_unit_list", [])) | set(lists.get("flying_unit_list", []))}
    skip = set(alias) | set(named) | not_units
    units = (set(lists.get("ground_unit_list", [])) | set(lists.get("flying_unit_list", []))) - skip
    ground = sorted(u for u in units if u not in AIR)
    air = sorted(u for u in units if u in AIR)
    return {"_doc": "KataCR unit -> catch-all base class (tools/build_katacr_generic.py). "
                    "'named' maps evolutions of cards we detect by name to that card.",
            "named": dict(sorted(named.items())), "other_ground": ground, "other_air": air}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-list", default=str(ROOT / "models" / "katacr_label_list.py"))
    ap.add_argument("--alias", default=str(ROOT / "models" / "katacr_alias.json"))
    ap.add_argument("--out", default=str(ROOT / "models" / "katacr_generic.json"))
    a = ap.parse_args()
    data = build(read_lists(Path(a.label_list)), json.loads(Path(a.alias).read_text()))
    Path(a.out).write_text(json.dumps(data, indent=1) + "\n")
    print(f"{a.out}: {len(data['other_ground'])} ground, {len(data['other_air'])} air, "
          f"{len(data['named'])} evolutions mapped to named classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
