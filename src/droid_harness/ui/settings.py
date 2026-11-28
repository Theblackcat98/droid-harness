"""Studio app settings: one JSON file per checkout, with a schema the page renders as a form.

Everything the app does with no flags comes from here. The file lives at
``.studio/settings.json`` in the repository (gitignored); unknown keys are dropped and bad
values fall back to their defaults, so a hand-edited file can't stop the app from starting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODES = ["ladder", "training_camp", "practice", "friendly_host", "friendly_guest"]


@dataclass(frozen=True)
class Field:
    key: str
    default: Any
    group: str
    label: str
    help: str = ""
    choices: list[str] | None = None
    lo: float | None = None
    hi: float | None = None

    @property
    def kind(self) -> str:
        if self.choices:
            return "choice"
        if isinstance(self.default, bool):
            return "bool"
        if isinstance(self.default, int):
            return "int"
        if isinstance(self.default, float):
            return "float"
        return "text"

    def coerce(self, value: Any) -> Any:
        """The value as this field's type, or ValueError."""
        k = self.kind
        if k == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if k in ("int", "float"):
            v = int(value) if k == "int" else float(value)
            if (self.lo is not None and v < self.lo) or (self.hi is not None and v > self.hi):
                raise ValueError(f"{self.label} must be between {self.lo:g} and {self.hi:g}")
            return v
        v = "" if value is None else str(value).strip()
        if k == "choice" and v not in (self.choices or []):
            raise ValueError(f"{self.label} must be one of {', '.join(self.choices or [])}")
        return v


FIELDS: list[Field] = [
    # Phone
    Field("device", "phone-a", "Phone", "Device preset",
          "Name of the calibration bundle under configs/ (profile, ROIs, tiles, templates)."),
    Field("serial", "", "Phone", "ADB serial override",
          "Leave empty to use the profile's serial. For adb over Wi-Fi, use the phone's address and port, e.g. <phone-ip>:5555. "
          "Sets DROID_HARNESS_SERIAL for every job."),
    Field("adb_path", "", "Phone", "adb program", "Leave empty to find adb on PATH."),
    Field("preview_detector", True, "Phone", "Run the detector in the preview",
          "Off makes the preview lighter (no unit boxes), useful without a GPU."),
    # Folders
    Field("data_dir", "data/matches", "Folders", "Match folder for new games",
          "Where Play and the guided flow's data collection write matches."),
    Field("screenshot_dir", "data/screenshots", "Folders", "Screenshot folder",
          "Screenshots from the preview are saved here as PNG files."),
    Field("screenshot_overlay", False, "Folders", "Also save a copy with overlays",
          "Next to the clean frame, saves what the preview shows (boxes, candidates, touches)."),
    # Playing
    Field("champion", "champion", "Playing", "Strategy",
          "Strategy for Play and the champion in the guided flow. 'champion' follows configs/strategies/CHAMPION."),
    Field("play_mode", "ladder", "Playing", "Mode for Play", choices=MODES),
    Field("play_matches", 10, "Playing", "Matches per Play", lo=1, hi=500),
    Field("prune_win_video", True, "Playing", "Delete video of clean wins",
          "Keeps the disk from filling up; losses keep their video for review."),
    Field("auto_upgrade", False, "Playing", "Upgrade cards between matches",
          "Gold only, never gems. Changes your levels, which voids a running experiment."),
    # Speed 1
    Field("min_removed", 0.5, "Speed 1 (offline search)", "Share of pattern hits a fix must remove",
          lo=0.05, hi=1.0),
    Field("max_collateral", 3.0, "Speed 1 (offline search)", "Most other decisions a fix may change (%)",
          lo=0.1, hi=50.0),
    Field("trials", 40, "Speed 1 (offline search)", "Parameter sets to try", lo=4, hi=1000),
    Field("cards", "", "Speed 1 (offline search)", "Card database",
          "Leave empty to use the device bundle's cards.json."),
    # Speed 2
    Field("test_mode", "training_camp", "Speed 2 (phone test)", "Where to test", choices=MODES,
          help="Training Camp and Practice keep the opponent fixed, so results come sooner than on the ladder."),
    Field("test", "dense", "Speed 2 (phone test)", "Statistical test", choices=["dense", "pairs"],
          help="dense uses every pair's margin; pairs counts only pairs where one side won and the other lost."),
    Field("max_games", 60, "Speed 2 (phone test)", "Most games per test", lo=4, hi=1000),
    Field("futility_pairs", 8, "Speed 2 (phone test)", "Stop a clearly worse candidate after this many pairs",
          lo=0, hi=200),
    # Studio
    Field("open_browser", True, "Studio", "Open the browser when the app starts"),
    Field("native_window", False, "Studio", "Use a desktop window",
          "Needs pywebview: pip install -e '.[gui]'. Falls back to the browser."),
    Field("host", "127.0.0.1", "Studio", "Listen address",
          "127.0.0.1 keeps the app on this computer; 0.0.0.0 shares it with your network. Restart to apply."),
    Field("port", 8765, "Studio", "Port", "Restart to apply.", lo=1024, hi=65535),
    Field("ui_fps", 15.0, "Studio", "Live view updates per second", lo=1.0, hi=30.0),
]
BY_KEY = {f.key: f for f in FIELDS}


def find_repo_root(start: Path | None = None) -> Path:
    """The checkout the app works in: the current directory if it holds configs/strategies,
    else the checkout this package was installed from (editable installs), else cwd."""
    cwd = Path.cwd() if start is None else Path(start)
    for p in (cwd, *cwd.parents):
        if (p / "configs/strategies").is_dir():
            return p
    pkg_root = Path(__file__).resolve().parents[3]
    if (pkg_root / "configs/strategies").is_dir():
        return pkg_root
    return cwd


@dataclass
class Settings:
    path: Path
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Settings:
        s = cls(Path(path), {f.key: f.default for f in FIELDS})
        try:
            raw = json.loads(s.path.read_text())
        except (OSError, ValueError):
            raw = {}
        for k, v in (raw if isinstance(raw, dict) else {}).items():
            f = BY_KEY.get(k)
            if f is None:
                continue
            try:
                s.values[k] = f.coerce(v)
            except (TypeError, ValueError):
                pass  # keep the default
        return s

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def update(self, changes: dict[str, Any]) -> dict[str, str]:
        """Apply what validates, return {key: error} for what didn't. Saves on success."""
        errors: dict[str, str] = {}
        for k, v in changes.items():
            f = BY_KEY.get(k)
            if f is None:
                errors[k] = "unknown setting"
                continue
            try:
                self.values[k] = f.coerce(v)
            except (TypeError, ValueError) as e:
                errors[k] = str(e) or f"invalid value for {f.label}"
        self.save()
        return errors

    def reset(self) -> None:
        self.values = {f.key: f.default for f in FIELDS}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.values, indent=1))
        tmp.replace(self.path)

    def job_env(self) -> dict[str, str]:
        """Environment for jobs: the serial override and unbuffered output."""
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if self.values.get("serial"):
            env["DROID_HARNESS_SERIAL"] = str(self.values["serial"])
        return env

    @staticmethod
    def schema() -> list[dict[str, Any]]:
        return [{"key": f.key, "default": f.default, "group": f.group, "label": f.label, "help": f.help,
                 "kind": f.kind, "choices": f.choices, "lo": f.lo, "hi": f.hi} for f in FIELDS]
