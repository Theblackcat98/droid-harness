"""Set-of-Mark overlay: number candidate elements so a GUI model answers with an ID.

Answering "tap 7" is far more reliable than predicting raw coordinates; the harness maps
the ID back to the element's bounds and taps its center.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Element:
    bounds: tuple[int, int, int, int]  # x1, y1, x2, y2 in frame pixels
    text: str = ""
    resource_id: str = ""
    clickable: bool = True

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bounds
        return ((x1 + x2) // 2, (y1 + y2) // 2)


def overlay(img: np.ndarray, elements: list[Element], *, only_clickable: bool = True,
            min_area: int = 64) -> tuple[np.ndarray, dict[int, Element]]:
    out = img.copy()
    marks: dict[int, Element] = {}
    for e in elements:
        x1, y1, x2, y2 = e.bounds
        if (only_clickable and not e.clickable) or (x2 - x1) * (y2 - y1) < min_area:
            continue
        i = len(marks) + 1
        marks[i] = e
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1), (x1 + tw + 4, y1 + th + 4), (0, 0, 255), -1)
        cv2.putText(out, label, (x1 + 2, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out, marks


def legend(marks: dict[int, Element]) -> str:
    """Text listing to send alongside the marked screenshot."""
    lines = []
    for i, e in marks.items():
        desc = e.text or e.resource_id or "(no label)"
        lines.append(f"[{i}] {desc}")
    return "\n".join(lines)
