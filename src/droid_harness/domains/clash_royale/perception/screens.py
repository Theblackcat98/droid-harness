"""Screen identification by difference hashes of fixed regions."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .grid import Roi


def dhash(img: np.ndarray, size: int = 8) -> int:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    small = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class ScreenClassifier:
    regions: list[Roi]
    references: dict[str, list[list[int]]] = field(default_factory=dict)
    max_distance: int = 10  # summed over regions

    def signature(self, img: np.ndarray) -> list[int]:
        return [dhash(r.crop(img)) for r in self.regions]

    def register(self, name: str, img: np.ndarray) -> None:
        self.references.setdefault(name, []).append(self.signature(img))

    def classify(self, img: np.ndarray) -> tuple[str, int]:
        sig = self.signature(img)
        best, dist = "UNKNOWN", 10**9
        for name, refs in self.references.items():
            for ref in refs:
                d = sum(hamming(a, b) for a, b in zip(sig, ref, strict=False))
                if d < dist:
                    best, dist = name, d
        return (best, dist) if dist <= self.max_distance else ("UNKNOWN", dist)
