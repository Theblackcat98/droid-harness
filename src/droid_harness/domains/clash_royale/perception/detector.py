"""Arena unit detector (YOLO26 exported to ONNX), plus a protocol for test doubles.

Runtime on an AMD GPU: ONNX Runtime's ROCm EP was removed upstream; MIGraphX is AMD's
supported path. Benchmark it against PyTorch-ROCm and keep the faster (plan §6.3).
Use a static input shape and call ``warmup()`` at startup — MIGraphX compiles per shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in the image passed in

    @property
    def foot(self) -> tuple[float, float]:
        """Ground contact point (bottom-center), used for tile mapping."""
        return ((self.box[0] + self.box[2]) / 2, self.box[3])


class Detector(Protocol):
    def __call__(self, img: np.ndarray) -> list[Detection]: ...


PREFERRED_PROVIDERS = ("MIGraphXExecutionProvider", "ROCMExecutionProvider", "CUDAExecutionProvider",
                       "CPUExecutionProvider")


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, (left, top)


class OnnxDetector:
    def __init__(self, model_path: str | Path, class_names: list[str], *, imgsz: int = 640, conf: float = 0.35,
                 iou: float = 0.5, providers: list[str] | None = None) -> None:
        import onnxruntime as ort  # optional dependency

        available = ort.get_available_providers()
        chosen = [p for p in (providers or PREFERRED_PROVIDERS) if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=chosen)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def warmup(self, n: int = 5) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), np.uint8)
        for _ in range(n):
            self(dummy)

    def __call__(self, img: np.ndarray) -> list[Detection]:
        canvas, r, (left, top) = letterbox(img, self.imgsz)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        out = self.session.run(None, {self.input_name: blob})[0]
        return self.decode(out, r, (left, top))

    def decode(self, out: np.ndarray, r: float, pad: tuple[int, int]) -> list[Detection]:
        out = np.asarray(out)
        if out.ndim == 3:
            out = out[0]
        n_raw = 4 + len(self.class_names)
        if out.shape[-1] == 6 and out.shape[0] != n_raw:  # end-to-end (NMS-free): [x1,y1,x2,y2,score,cls]
            rows = out[out[:, 4] >= self.conf]
            boxes, scores, classes = rows[:, :4], rows[:, 4], rows[:, 5].astype(int)
        else:  # raw head: (4 + nc, anchors) with xywh centers -> needs NMS
            preds = out.T if out.shape[0] == n_raw else out
            cls_scores = preds[:, 4:]
            classes = cls_scores.argmax(1)
            scores = cls_scores.max(1)
            keep = scores >= self.conf
            preds, classes, scores = preds[keep], classes[keep], scores[keep]
            xywh = preds[:, :4]
            boxes = np.column_stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                                     xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2])
            idx = cv2.dnn.NMSBoxes([list(map(float, (b[0], b[1], b[2] - b[0], b[3] - b[1]))) for b in boxes],
                                   scores.astype(float).tolist(), self.conf, self.iou)
            idx = np.array(idx).reshape(-1)
            boxes, scores, classes = boxes[idx], scores[idx], classes[idx]
        dets = []
        for b, s, c in zip(boxes, scores, classes, strict=False):
            x1, y1, x2, y2 = ((b[0] - pad[0]) / r, (b[1] - pad[1]) / r, (b[2] - pad[0]) / r, (b[3] - pad[1]) / r)
            name = self.class_names[c] if 0 <= c < len(self.class_names) else str(c)
            dets.append(Detection(name, float(s), (float(x1), float(y1), float(x2), float(y2))))
        return dets
