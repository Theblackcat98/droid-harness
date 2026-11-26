"""Accessibility-tree access for general Android apps.

Primary: DroidRun Portal (an accessibility service exposing the tree via a content provider).
Fallback: ``uiautomator dump`` (slower, no service needed). Both sit behind one interface
because some Android skins restrict accessibility grants or background services — test
Portal on your phones early (plan §13).
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from typing import Protocol

from ...core.transport.adb import Adb
from .som import Element

PORTAL_URI = "content://com.droidrun.portal/a11y_tree"


class TreeSource(Protocol):
    def elements(self) -> list[Element]: ...
    def tree_hash(self) -> str: ...


def _hash_elements(els: list[Element]) -> str:
    h = hashlib.sha256()
    for e in els:
        h.update(f"{e.bounds}|{e.text}|{e.resource_id}|{e.clickable}".encode())
    return h.hexdigest()[:16]


def parse_portal_output(out: str) -> list[Element]:
    """``content query`` prints ``Row: 0 result={...}``; take the JSON payload."""
    start = min((i for i in (out.find("{"), out.find("[")) if i >= 0), default=-1)
    if start < 0:
        return []
    data = json.loads(out[start:].strip())
    if isinstance(data, dict):
        data = data.get("result", data.get("elements", data))
        if isinstance(data, str):
            data = json.loads(data)
    els: list[Element] = []

    def walk(node):
        if isinstance(node, list):
            for n in node:
                walk(n)
            return
        if not isinstance(node, dict):
            return
        b = node.get("bounds")
        if isinstance(b, str):
            nums = [int(x) for x in re.findall(r"-?\d+", b)]
            b = nums if len(nums) == 4 else None
        if b:
            els.append(Element(tuple(int(v) for v in b), node.get("text") or "", node.get("resourceId") or
                               node.get("resource_id") or "", bool(node.get("clickable", True))))
        walk(node.get("children", []))

    walk(data)
    return els


def parse_uiautomator_xml(xml: str) -> list[Element]:
    root = ET.fromstring(xml[xml.find("<"):])
    els = []
    for n in root.iter("node"):
        nums = [int(x) for x in re.findall(r"\d+", n.get("bounds", ""))]
        if len(nums) != 4:
            continue
        els.append(Element(tuple(nums), n.get("text", ""), n.get("resource-id", ""),
                           n.get("clickable") == "true"))
    return els


class PortalTree:
    def __init__(self, adb: Adb) -> None:
        self.adb = adb

    def elements(self) -> list[Element]:
        return parse_portal_output(self.adb.shell(f"content query --uri {PORTAL_URI}"))

    def tree_hash(self) -> str:
        return _hash_elements(self.elements())


class UiAutomatorTree:
    def __init__(self, adb: Adb) -> None:
        self.adb = adb

    def elements(self) -> list[Element]:
        self.adb.shell("uiautomator dump /sdcard/dh_ui.xml", check=False)
        return parse_uiautomator_xml(self.adb.shell("cat /sdcard/dh_ui.xml"))

    def tree_hash(self) -> str:
        return _hash_elements(self.elements())
