"""Train the M3 arena detector: YOLO26-n → static-shape ONNX + classes.txt (plan §6.3).

Ultralytics is an optional dependency (AGPL-3.0): importing this module is cheap,
training requires ``pip install ultralytics`` and PyTorch-ROCm on the host.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_yaml: Path  # Ultralytics dataset yaml (train/val paths, names)
    model: str = "yolo26n.pt"
    imgsz: int = 640
    epochs: int = 120
    batch: int = 16
    device: str = "0"  # ROCm device
    project: Path = Path("models/runs")
    name: str = "yolo26n-arena"
    out_onnx: Path = Path("models/detector.onnx")


def train(cfg: TrainConfig) -> Path:
    """Train and export static-shape ONNX. Returns the ONNX path. Requires ultralytics."""
    from ultralytics import YOLO  # lazy: AGPL code only loads when training

    model = YOLO(cfg.model)
    model.train(data=str(cfg.data_yaml), imgsz=cfg.imgsz, epochs=cfg.epochs,
                batch=cfg.batch, device=cfg.device, project=str(cfg.project), name=cfg.name)
    best = cfg.project / cfg.name / "weights" / "best.pt"
    export_onnx(best, cfg.out_onnx, cfg.imgsz)
    return cfg.out_onnx


def export_onnx(weights: Path, out: Path, imgsz: int = 640) -> Path:
    """Export weights to static-shape ONNX (MIGraphX compiles per shape, so no dynamic axes)."""
    from ultralytics import YOLO

    out.parent.mkdir(parents=True, exist_ok=True)
    YOLO(str(weights)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True)
    produced = weights.parent / (weights.stem + ".onnx")
    if produced != out:
        shutil.move(str(produced), str(out))
    return out


def write_classes(names: list[str], out: Path = Path("models/classes.txt")) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(names) + "\n")
    return out
