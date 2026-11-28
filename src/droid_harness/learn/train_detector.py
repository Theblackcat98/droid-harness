"""Train the M3 arena detector: YOLO26-n -> static-shape ONNX with embedded class names (plan §6.3).

Ultralytics is an optional dependency (AGPL-3.0): importing this module is cheap,
training requires ``pip install ultralytics`` and PyTorch-ROCm on the host.

* Training runs square (``imgsz``, Ultralytics letterboxes mosaics); the export is the
  rectangular static shape inference uses (``export_imgsz``, (h, w)). Arena crops are
  ~0.6 aspect, so 640x384 wastes no compute on padding. The network is fully
  convolutional, so the square-trained weights run at the rectangular shape.
* Ultralytics embeds ``names`` in the ONNX metadata; ``OnnxDetector`` reads them and
  refuses a mismatched ``--classes`` file. ``export_onnx`` checks that after export.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data_yaml: Path  # Ultralytics dataset yaml (train/val paths, names)
    model: str = "yolo26n.pt"
    imgsz: int = 640
    export_imgsz: tuple[int, int] = (640, 384)  # (h, w) static ONNX input
    epochs: int = 120
    batch: int = 16
    device: str = "0"  # ROCm device
    project: Path = Path("models/runs")
    name: str = "yolo26n-arena"
    out_onnx: Path = Path("models/detector.onnx")
    cache: str | bool = False
    workers: int = 8
    patience: int = 15


def train(cfg: TrainConfig) -> Path:
    """Train and export static-shape ONNX. Returns the ONNX path. Requires ultralytics."""
    from ultralytics import YOLO  # lazy: AGPL code only loads when training

    model = YOLO(cfg.model)
    model.train(data=str(cfg.data_yaml), imgsz=cfg.imgsz, epochs=cfg.epochs, batch=cfg.batch,
                device=cfg.device, project=str(cfg.project.resolve()), name=cfg.name,
                cache=cfg.cache, workers=cfg.workers, patience=cfg.patience,
                exist_ok=True)  # without it Ultralytics auto-increments the dir and we'd read a stale best.pt
    best = Path(getattr(model.trainer, "best", "") or Path(model.trainer.save_dir) / "weights" / "best.pt")
    return export_onnx(best, cfg.out_onnx, cfg.export_imgsz, expected_names=dataset_names(cfg.data_yaml))


def dataset_names(data_yaml: str | Path) -> list[str]:
    import yaml

    names = yaml.safe_load(Path(data_yaml).read_text())["names"]
    return list(names) if isinstance(names, list) else [names[k] for k in sorted(names, key=int)]


def export_onnx(weights: Path, out: Path, imgsz: int | tuple[int, int] = (640, 384),
                expected_names: list[str] | None = None) -> Path:
    """Export weights to static-shape ONNX (MIGraphX compiles per shape, so no dynamic axes)."""
    from ultralytics import YOLO

    out.parent.mkdir(parents=True, exist_ok=True)
    size = list(imgsz) if isinstance(imgsz, (tuple, list)) else imgsz
    produced = Path(YOLO(str(weights)).export(format="onnx", imgsz=size, dynamic=False, simplify=True))
    if produced.resolve() != out.resolve():
        shutil.move(str(produced), str(out))
    if expected_names is not None:
        verify_onnx(out, expected_names)
    return out


def verify_onnx(path: Path, expected_names: list[str]) -> None:
    """Load the export the way the bot will; raises if names, shape or head are off."""
    try:
        from ..domains.clash_royale.perception.detector import OnnxDetector
        det = OnnxDetector(path, expected_names, providers=["CPUExecutionProvider"])
    except ImportError:
        log.warning("onnxruntime not installed; skipped export verification")
        return
    log.info("verified %s: input %s, %s head, %d classes", path, det.imgsz, det.mode, len(det.class_names))


def write_classes(names: list[str], out: Path) -> Path:
    """Write a plain class list. ``out`` is required: never clobber the curated models/classes.txt."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(names) + "\n")
    return out


def _parse_cache(v: str) -> str | bool:
    low = v.strip().lower()
    if low in ("", "false", "0", "no"):
        return False
    if low in ("true", "1", "yes"):
        return True
    return v.strip()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="train_detector")
    ap.add_argument("--data", required=True, help="Ultralytics data.yaml (from droid-harness synth / convert-katacr)")
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--export-imgsz", type=int, nargs=2, default=[640, 384], metavar=("H", "W"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="yolo26n-arena")
    ap.add_argument("--out", default="models/detector.onnx")
    ap.add_argument("--cache", default="false")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15)
    a = ap.parse_args(argv)
    out = train(TrainConfig(Path(a.data), a.model, a.imgsz, tuple(a.export_imgsz), a.epochs, a.batch, a.device,
                            name=a.name, out_onnx=Path(a.out),
                            cache=_parse_cache(a.cache), workers=a.workers, patience=a.patience))
    print(f"exported {out}")


if __name__ == "__main__":
    main()
