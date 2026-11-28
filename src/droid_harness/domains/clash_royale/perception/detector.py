"""Arena unit detector (YOLO26 exported to ONNX), plus a protocol for test doubles.

Runtime on an AMD GPU: ONNX Runtime's ROCm EP was removed upstream; MIGraphX is AMD's
supported path. Benchmark it against PyTorch-ROCm and keep the faster (plan §6.3).
Use a static input shape and call ``warmup()`` at startup — MIGraphX compiles per shape.

Class names: Ultralytics embeds ``names`` in the ONNX metadata. That is the source of
truth; a ``classes.txt`` passed alongside is only a cross-check, and a mismatch refuses
to load (a shifted list silently turns every box into the wrong card).

Output layout is pinned once at load time from the model's output shape (and the
``end2end`` metadata key when present):

* ``e2e``  — NMS-free head, ``(N, 6)`` rows of ``x1, y1, x2, y2, score, cls``
* ``raw``  — ``(4 + nc, anchors)`` xywh centers + class scores, needs NMS
* ``raw_t``— the same, transposed ``(anchors, 4 + nc)``
"""

from __future__ import annotations

import ast
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .classes import load_class_names

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in the image passed in


class Detector(Protocol):
    def __call__(self, img: np.ndarray) -> list[Detection]: ...


# ROCMExecutionProvider was removed from ONNX Runtime upstream; MIGraphX replaces it.
PREFERRED_PROVIDERS = ("MIGraphXExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")

_NMS_CLASS_OFFSET = 4096.0  # > any input side; keeps per-class NMS boxes from overlapping


def _hw(size: int | tuple[int, int]) -> tuple[int, int]:
    """``size`` is a square side or ``(h, w)``."""
    if isinstance(size, int):
        return size, size
    h, w = size
    return int(h), int(w)


def letterbox(img: np.ndarray, size: int | tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Aspect-preserving resize + gray padding (Ultralytics-compatible). ``size``: side or (h, w)."""
    H, W = _hw(size)
    h, w = img.shape[:2]
    r = min(H / h, W / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (H - nh) // 2, (W - nw) // 2
    canvas = np.full((H, W, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = resized[:, :, :3]
    return canvas, r, (left, top)


def _normalize_names(val: Any) -> list[str] | None:
    if isinstance(val, dict):
        return [str(val[k]) for k in sorted(val, key=int)]
    if isinstance(val, (list, tuple)):
        return [str(v) for v in val]
    return None


def names_from_metadata(meta: dict[str, str]) -> list[str] | None:
    """Ultralytics stores ``names`` as ``str({0: 'a', 1: 'b'})`` in ONNX custom metadata."""
    raw = meta.get("names")
    if not raw:
        return None
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    return _normalize_names(val)


def resolve_class_names(class_names: list[str] | str | Path | None, embedded: list[str] | None,
                        model_path: str | Path, *, source_label: str = "ONNX 'names' metadata") -> list[str]:
    """Resolve detector class list and refuse mismatches against embedded model names."""
    given: list[str] | None
    if class_names is None:
        given = None
    elif isinstance(class_names, (str, Path)):
        given = load_class_names(class_names)
    else:
        given = [str(c) for c in class_names]
    # A classes file may be a superset of what the model predicts (e.g. the 214-name roadmap
    # list whose first 31 entries are the shipped model's classes). Decoding only ever sees the
    # model's own indices, so a matching prefix is safe: keep the model's names and ignore the
    # extra tail. Anything else is still a hard error.
    if embedded is not None and given is not None and len(given) > len(embedded) \
            and given[:len(embedded)] == embedded:
        log.warning("detector: %s: classes file lists %d names, model predicts %d; using the model's",
                    model_path, len(given), len(embedded))
        given = embedded
    if embedded is not None and given is not None and embedded != given:
        diff = next((i for i, (a, b) in enumerate(zip(embedded, given, strict=False)) if a != b),
                    min(len(embedded), len(given)))
        raise ValueError(
            f"{model_path}: class list disagrees with the model's embedded names at index {diff} "
            f"({len(given)} given vs {len(embedded)} embedded). Refusing to load: every box would "
            f"decode as the wrong class.")
    names = embedded if embedded is not None else given
    if not names:
        raise ValueError(f"{model_path}: no class names (no {source_label} and no classes file)")
    if embedded is None:
        log.warning("detector: %s has no embedded class names; trusting the classes file unchecked", model_path)
    return names


def output_mode(shape: list[Any] | tuple[Any, ...], nc: int, end2end: bool | None = None) -> str | None:
    """Pin the decode branch from the output shape. ``None`` if the shape is symbolic."""
    dims = [d if isinstance(d, int) and d > 0 else None for d in shape]
    if len(dims) == 3:
        dims = dims[1:]
    if len(dims) != 2:
        return None
    a, b = dims
    if end2end is True:
        return "e2e"
    # Six columns is e2e unless it's a transposed raw head of a 2-class model; raw heads
    # have thousands of anchors, e2e heads a few hundred rows (max_det).
    if end2end is None and b == 6 and (a is None or a <= 1000 or 4 + nc != 6):
        return "e2e"
    if a == 4 + nc:
        return "raw"
    if b == 4 + nc:
        return "raw_t"
    return None


class OnnxDetector:
    def __init__(self, model_path: str | Path, class_names: list[str] | str | Path | None = None, *,
                 imgsz: int | tuple[int, int] | None = None, conf: float = 0.35, iou: float = 0.5,
                 agnostic_nms: bool = False, providers: list[str] | None = None) -> None:
        import onnxruntime as ort  # optional dependency

        available = ort.get_available_providers()
        chosen = [p for p in (providers or PREFERRED_PROVIDERS) if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=chosen)
        self.provider = self.session.get_providers()[0]
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        meta = dict(self.session.get_modelmeta().custom_metadata_map or {})
        self.class_names = self._resolve_names(class_names, meta, model_path)
        self.conf, self.iou, self.agnostic_nms = conf, iou, agnostic_nms

        # Static input shape wins over the argument: MIGraphX compiles per shape anyway.
        shape = list(inp.shape)
        if len(shape) == 4 and all(isinstance(d, int) and d > 0 for d in shape[2:]):
            self.imgsz = (int(shape[2]), int(shape[3]))
            if imgsz is not None and _hw(imgsz) != self.imgsz:
                log.warning("detector: imgsz %s ignored; the model's static input is %s", imgsz, self.imgsz)
        else:
            self.imgsz = _hw(imgsz if imgsz is not None else 640)

        e2e_meta = meta.get("end2end")
        end2end = None if e2e_meta is None else str(e2e_meta).strip().lower() == "true"
        self.mode = output_mode(self.session.get_outputs()[0].shape, len(self.class_names), end2end)
        self._end2end_meta = end2end
        if self.mode is not None:
            self._check_nc_for_mode(self.session.get_outputs()[0].shape)

    # --- loading ----------------------------------------------------------------
    @staticmethod
    def _resolve_names(class_names, meta: dict[str, str], model_path) -> list[str]:
        return resolve_class_names(class_names, names_from_metadata(meta), model_path)

    def _check_nc_for_mode(self, shape) -> None:
        dims = [d for d in shape if isinstance(d, int)]
        nc = len(self.class_names)
        if self.mode == "raw" and len(dims) >= 2 and dims[-2] != 4 + nc:
            raise ValueError(f"raw head has {dims[-2] - 4} classes, class list has {nc}")
        if self.mode == "raw_t" and dims and dims[-1] != 4 + nc:
            raise ValueError(f"raw head has {dims[-1] - 4} classes, class list has {nc}")

    def warmup(self, n: int = 5) -> None:
        h, w = self.imgsz
        dummy = np.zeros((h, w, 3), np.uint8)
        for _ in range(n):
            self(dummy)

    # --- inference ----------------------------------------------------------------
    def __call__(self, img: np.ndarray) -> list[Detection]:
        canvas, r, pad = letterbox(img, self.imgsz)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        out = self.session.run(None, {self.input_name: blob})[0]
        return self.decode(out, r, pad)

    def decode(self, out: np.ndarray, r: float, pad: tuple[int, int]) -> list[Detection]:
        out = np.asarray(out)
        if out.ndim == 3:
            out = out[0]
        mode = self.mode or output_mode(out.shape, len(self.class_names), self._end2end_meta)
        if mode is None:
            raise ValueError(f"cannot interpret detector output of shape {out.shape}")
        if self.mode is None:
            self.mode = mode  # symbolic dims at load: pin on the first real output
        if mode == "e2e":
            boxes, scores, classes = self._decode_e2e(out)
        else:
            boxes, scores, classes = self._decode_raw(out.T if mode == "raw" else out)
        dets = []
        for b, s, c in zip(boxes, scores, classes, strict=True):
            x1, y1 = (b[0] - pad[0]) / r, (b[1] - pad[1]) / r
            x2, y2 = (b[2] - pad[0]) / r, (b[3] - pad[1]) / r
            name = self.class_names[c] if 0 <= c < len(self.class_names) else str(c)
            dets.append(Detection(name, float(s), (float(x1), float(y1), float(x2), float(y2))))
        return dets

    def _decode_e2e(self, out: np.ndarray):
        rows = out[out[:, 4] >= self.conf]  # conf filter only; e2e heads have no NMS (and no iou)
        return rows[:, :4], rows[:, 4], rows[:, 5].astype(int)

    def _decode_raw(self, preds: np.ndarray):
        empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
        if preds.size == 0:
            return empty
        cls_scores = preds[:, 4:]
        classes = cls_scores.argmax(1)
        scores = cls_scores.max(1)
        keep = scores >= self.conf
        if not keep.any():
            return empty  # empty arena: nothing above conf (used to crash NMS indexing)
        preds, classes, scores = preds[keep], classes[keep], scores[keep]
        xywh = preds[:, :4]
        boxes = np.column_stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                                 xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2])
        # Per-class NMS (Ultralytics' default): offset each class into its own region so a
        # unit can't suppress an overlapping unit of another class.
        off = 0.0 if self.agnostic_nms else _NMS_CLASS_OFFSET
        shifted = boxes + (classes[:, None] * off)
        rects = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in shifted]
        idx = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), self.conf, self.iou)
        idx = np.asarray(idx, dtype=int).reshape(-1)
        if idx.size == 0:
            return empty
        return boxes[idx], scores[idx], classes[idx]


def _load_yolo(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def _to_numpy(arr: Any) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach()
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    return np.asarray(arr)


class TorchDetector:
    """Ultralytics ``.pt`` detector running directly via PyTorch (ROCm/CUDA or CPU)."""

    def __init__(self, model_path: str | Path, class_names: list[str] | str | Path | None = None, *,
                 imgsz: int | tuple[int, int] = (640, 384), conf: float = 0.35, iou: float = 0.5,
                 agnostic_nms: bool = False, device: str | None = None, half: bool = True) -> None:
        import torch

        self.model = _load_yolo(str(model_path))
        self.device = ("0" if torch.cuda.is_available() else "cpu") if device is None else str(device)
        self.half = bool(half and self.device.lower() != "cpu")
        self.imgsz = _hw(imgsz)
        self.conf, self.iou, self.agnostic_nms = conf, iou, agnostic_nms
        embedded = _normalize_names(getattr(self.model, "names", None))
        self.class_names = resolve_class_names(class_names, embedded, model_path,
                                               source_label="PyTorch 'names' metadata")
        self.provider = f"PyTorch({self.device})"
        self.mode = "e2e"

    def warmup(self, n: int = 3) -> None:
        h, w = self.imgsz
        dummy = np.zeros((h, w, 3), np.uint8)
        for _ in range(n):
            self(dummy)

    def __call__(self, img: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            img, imgsz=list(self.imgsz), conf=self.conf, iou=self.iou,
            agnostic_nms=self.agnostic_nms, device=self.device, verbose=False,
        )[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        clss = _to_numpy(boxes.cls).astype(int)
        dets = []
        for b, s, c in zip(xyxy, confs, clss, strict=True):
            name = self.class_names[c] if 0 <= c < len(self.class_names) else str(c)
            dets.append(Detection(name, float(s), (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
        return dets


_DETECTOR_CACHE: dict[tuple[Any, ...], OnnxDetector | TorchDetector] = {}
_DETECTOR_CACHE_LOCK = threading.Lock()
_WARMED_IDS: set[int] = set()


def _cache_key(model_path: str | Path, class_names: list[str] | str | Path | None,
               imgsz: int | tuple[int, int] | None, conf: float, iou: float,
               agnostic_nms: bool, providers: list[str] | None, device: str | None) -> tuple[Any, ...]:
    p = Path(model_path).resolve()
    try:
        st = p.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = (0, 0)
    if class_names is None:
        cls_sig: Any = None
    elif isinstance(class_names, (str, Path)):
        cp = Path(class_names).resolve()
        try:
            cst = cp.stat()
            cls_sig = (str(cp), cst.st_mtime_ns, cst.st_size)
        except OSError:
            cls_sig = (str(cp), 0, 0)
    else:
        cls_sig = tuple(str(x) for x in class_names)
    sz_sig = _hw(imgsz) if imgsz is not None else None
    prov_sig = tuple(providers) if providers is not None else None
    return (str(p), sig, cls_sig, sz_sig, float(conf), float(iou), bool(agnostic_nms), prov_sig, device)


def load_detector(model_path: str | Path, class_names: list[str] | str | Path | None = None, *,
                  imgsz: int | tuple[int, int] | None = None, conf: float = 0.35, iou: float = 0.5,
                  agnostic_nms: bool = False, providers: list[str] | None = None,
                  device: str | None = None, cache: bool = True) -> OnnxDetector | TorchDetector:
    """Load an ONNX (``.onnx``) or PyTorch Ultralytics (``.pt``) detector, caching weights by file signature."""
    key = _cache_key(model_path, class_names, imgsz, conf, iou, agnostic_nms, providers, device) if cache else None
    if key is not None:
        with _DETECTOR_CACHE_LOCK:
            cached = _DETECTOR_CACHE.get(key)
            if cached is not None:
                return cached
    if Path(model_path).suffix.lower() == ".pt":
        det: OnnxDetector | TorchDetector = TorchDetector(
            model_path, class_names, imgsz=(640, 384) if imgsz is None else imgsz,
            conf=conf, iou=iou, agnostic_nms=agnostic_nms, device=device,
        )
    else:
        det = OnnxDetector(model_path, class_names, imgsz=imgsz, conf=conf, iou=iou,
                           agnostic_nms=agnostic_nms, providers=providers)
    if key is not None:
        with _DETECTOR_CACHE_LOCK:
            _DETECTOR_CACHE[key] = det
    return det


def warmup_once(detector: Any, n: int = 3) -> bool:
    """Run ``detector.warmup(n)`` only on first use of a cached detector instance."""
    did = id(detector)
    with _DETECTOR_CACHE_LOCK:
        if did in _WARMED_IDS:
            return False
        _WARMED_IDS.add(did)
    if hasattr(detector, "warmup"):
        detector.warmup(n)
        return True
    return False


class AsyncDetector:
    """Runs a detector on its own thread against the newest submitted image.

    The brain loop never blocks on inference: it submits arena crops when the worker is
    idle and picks up finished results with ``poll()``. Each result carries the caller's
    tag (frame time, frame, crop origin) so detections are mapped with the frame they
    came from, not the frame that happens to be current.
    """

    def __init__(self, detector: Detector) -> None:
        self.detector = detector
        self._cv = threading.Condition()
        self._job: tuple[np.ndarray, Any] | None = None
        self._result: tuple[Any, list[Detection], float] | None = None
        self._busy = False
        self._stop = False
        self.errors = 0
        self.infer_ms: list[float] = []
        self._thread = threading.Thread(target=self._run, name="T2-detector", daemon=True)
        self._thread.start()

    @property
    def idle(self) -> bool:
        with self._cv:
            return not self._busy and self._job is None

    def submit(self, img: np.ndarray, tag: Any) -> None:
        with self._cv:
            self._job = (img, tag)  # replaces any unstarted job: newest frame wins
            self._cv.notify_all()

    def poll(self) -> tuple[Any, list[Detection], float] | None:
        with self._cv:
            r, self._result = self._result, None
            return r

    def wait_idle(self, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._busy or self._job is not None:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                img, tag = self._job
                self._job, self._busy = None, True
            t0 = time.perf_counter()
            try:
                dets = self.detector(img)
            except Exception:
                log.exception("detector failed")
                self.errors += 1
                dets = []
            ms = (time.perf_counter() - t0) * 1000
            with self._cv:
                self.infer_ms.append(ms)
                if len(self.infer_ms) > 2000:
                    del self.infer_ms[:1000]
                self._result = (tag, dets, ms)
                self._busy = False
                self._cv.notify_all()
