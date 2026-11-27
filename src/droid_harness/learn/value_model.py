"""Win-probability model: P(win | logged state), trained on finished matches (plan step 6).

Why: the offline tools need to know whether a moment went well or badly without waiting for
the end of the match. A model of P(win) from the state gives a dense signal: the change in
win probability around a play is that play's credit, the biggest drops in a loss are where
to look, and the Studio draws it under the elixir tape.

Deliberately small and dependency-free: hand-made features from the logged state digest,
L2-regularized logistic regression in numpy, a split by match (states of one match are
never on both sides), and calibration numbers so you can see whether it's any good. It gets
useful past ~200 finished matches; below that, read its metrics before trusting it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = [
    "t_frac", "phase_x2", "phase_ot", "my_elixir", "opp_elixir_mid", "opp_elixir_width", "elixir_adv",
    "my_left_frac", "my_right_frac", "my_king_frac", "opp_left_frac", "opp_right_frac", "opp_king_frac",
    "crown_diff", "my_units", "opp_units", "opp_units_my_half", "my_units_their_half", "opp_hp_my_half",
    "my_hp_their_half", "opp_deck_known", "level_gap", "hand_fresh_frac",
]
RESULT_LABEL = {"win": 1.0, "loss": 0.0, "draw": 0.5}
VERSION = "droid-harness-value/1"


def features(st: dict, tower_max: dict[str, int] | None = None) -> list[float]:
    """Feature vector from a ``decisions.jsonl`` state digest (see ``FEATURES``)."""
    t = float(st.get("t_ms") or 0) / 180_000.0
    phase = st.get("phase") or "x1"
    my_e = float(st.get("my_elixir") or 0.0)
    lo, hi = (st.get("opp_elixir") or [5.0, 5.0])[:2]
    towers = st.get("towers") or {}
    tmax = tower_max or {}

    def frac(k: str, default_max: int) -> float:
        hp = towers.get(k)
        if hp is None:
            return 1.0  # unread: assume intact (the start-of-match case)
        return max(0.0, min(1.0, hp / max(1, tmax.get(k) or default_max)))

    crowns = st.get("crowns") or [0, 0]
    units = st.get("units") or []
    river = 16.0
    my_u = [u for u in units if len(u) >= 8 and u[2] == "me"]
    opp_u = [u for u in units if len(u) >= 8 and u[2] == "opp"]
    my_k, opp_k = st.get("my_king_level"), st.get("opp_king_level")
    fresh = st.get("hand_fresh") or []
    return [
        min(t, 1.7), 1.0 if phase == "x2" else 0.0, 1.0 if str(phase).startswith("ot") else 0.0,
        my_e / 10, (lo + hi) / 20, (hi - lo) / 10, (my_e - (lo + hi) / 2) / 10,
        frac("me_left", 1400), frac("me_right", 1400), frac("me_king", 2400),
        frac("opp_left", 1400), frac("opp_right", 1400), frac("opp_king", 2400),
        float(crowns[0] - crowns[1]) / 3,
        len(my_u) / 5, len(opp_u) / 5,
        sum(1 for u in opp_u if u[4] < river) / 5, sum(1 for u in my_u if u[4] > river) / 5,
        sum(float(u[7]) for u in opp_u if u[4] < river) / 3, sum(float(u[7]) for u in my_u if u[4] > river) / 3,
        1.0 if len(st.get("opp_deck_seen") or []) >= 8 else 0.0,
        float(max(0, min(4, (opp_k or 0) - (my_k or 0)))) / 4 if my_k and opp_k else 0.0,
        (sum(1 for f in fresh if f) / len(fresh)) if fresh else 1.0,
    ]


def _match_dirs(roots: list[Path]) -> list[Path]:
    return sorted({p.parent for r in roots for p in Path(r).rglob("decisions.jsonl")})


def load_dataset(roots: list[Path], *, every: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(X, y, match_index, match_ids) from finished matches; every ``every``-th logged state."""
    X, y, g, ids = [], [], [], []
    for d in _match_dirs(roots):
        try:
            s = json.loads((d / "summary.json").read_text())
        except (OSError, ValueError):
            continue
        if s.get("status") != "completed" or s.get("result") not in RESULT_LABEL:
            continue
        label = RESULT_LABEL[s["result"]]
        tmax: dict[str, int] = {}
        mi = len(ids)
        ids.append(str(s.get("match_id") or d.name))
        for i, line in enumerate((d / "decisions.jsonl").read_text().splitlines()):
            if not line.strip():
                continue
            try:
                st = json.loads(line).get("state")
            except ValueError:
                continue
            if not isinstance(st, dict):
                continue
            for k, v in (st.get("towers") or {}).items():
                if isinstance(v, (int, float)):
                    tmax[k] = max(tmax.get(k, 0), int(v))
            if i % every:
                continue
            X.append(features(st, tmax))
            y.append(label)
            g.append(mi)
    return np.asarray(X, np.float64), np.asarray(y, np.float64), np.asarray(g, np.int64), ids


def _split(ids: list[str], val_frac: float) -> set[int]:
    return {i for i, m in enumerate(ids) if int(hashlib.sha256(m.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < val_frac}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = p[y > 0.5], p[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if len(y) == 0:
        return {"states": 0}
    eps = 1e-7
    ll = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    bins = []
    for lo in np.arange(0, 1, 0.2):
        m = (p >= lo) & (p < lo + 0.2 + (1e-9 if lo >= 0.8 else 0))
        if m.any():
            bins.append({"pred": round(float(p[m].mean()), 3), "actual": round(float(y[m].mean()), 3), "n": int(m.sum())})
    return {"states": int(len(y)), "log_loss": round(ll, 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "accuracy": round(float(np.mean((p > 0.5) == (y > 0.5))), 3),
            "auc": None if (a := _auc(y, p)) is None else round(a, 3), "calibration": bins}


def train(roots: list[Path], *, l2: float = 1.0, epochs: int = 600, lr: float = 0.3, val_frac: float = 0.25,
          every: int = 4) -> dict[str, Any]:
    X, y, g, ids = load_dataset(roots, every=every)
    if len(ids) < 4 or len(X) == 0:
        raise ValueError(f"need finished matches with results; found {len(ids)}")
    val = _split(ids, val_frac)
    is_val = np.isin(g, list(val))
    if is_val.all() or not is_val.any():  # tiny data: hold out every 4th match instead
        val = set(range(0, len(ids), 4))
        is_val = np.isin(g, list(val))
    mean, std = X[~is_val].mean(0), X[~is_val].std(0) + 1e-6
    Z = (X - mean) / std
    # weight each match equally (long matches log more states)
    counts = np.bincount(g, minlength=len(ids)).astype(float)
    sw = 1.0 / counts[g]
    w, b = np.zeros(Z.shape[1]), 0.0
    tr = ~is_val
    sw_tr = sw[tr] / sw[tr].sum()
    for _ in range(epochs):
        p = _sigmoid(Z[tr] @ w + b)
        err = (p - y[tr]) * sw_tr
        w -= lr * (Z[tr].T @ err + l2 * w / max(1, tr.sum()))
        b -= lr * err.sum()
    p_all = _sigmoid(Z @ w + b)
    return {
        "version": VERSION, "features": FEATURES, "mean": mean.tolist(), "std": std.tolist(),
        "weights": w.tolist(), "bias": float(b),
        "matches": {"train": int(len(ids) - len(val)), "val": int(len(val))},
        "metrics": {"train": _metrics(y[tr], p_all[tr]), "val": _metrics(y[is_val], p_all[is_val])},
        "base_rate": round(float(np.mean([y[g == i][0] for i in range(len(ids)) if (g == i).any()])), 3),
        "top_weights": sorted(zip(FEATURES, (round(v, 3) for v in w.tolist()), strict=True), key=lambda kv: -abs(kv[1]))[:8],
    }


class ValueModel:
    def __init__(self, d: dict) -> None:
        if d.get("version") != VERSION or d.get("features") != FEATURES:
            raise ValueError("value model was trained with a different feature set; retrain it")
        self.d = d
        self.mean, self.std = np.asarray(d["mean"]), np.asarray(d["std"])
        self.w, self.b = np.asarray(d["weights"]), float(d["bias"])

    @classmethod
    def load(cls, path: str | Path) -> ValueModel:
        return cls(json.loads(Path(path).read_text()))

    def predict(self, st: dict, tower_max: dict[str, int] | None = None) -> float:
        z = (np.asarray(features(st, tower_max)) - self.mean) / self.std
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, float(z @ self.w + self.b))))))

    def predict_many(self, states: list[dict]) -> list[float]:
        tmax: dict[str, int] = {}
        out = []
        for st in states:
            for k, v in (st.get("towers") or {}).items():
                if isinstance(v, (int, float)):
                    tmax[k] = max(tmax.get(k, 0), int(v))
            out.append(self.predict(st, tmax))
        return out


def swings(states: list[dict], probs: list[float], *, window_ms: int = 5000, top: int = 5) -> list[dict]:
    """The biggest win-probability drops within ``window_ms``: where a loss turned."""
    out = []
    j = 0
    for i, st in enumerate(states):
        t = int(st.get("t_ms") or 0)
        while j < i and t - int(states[j].get("t_ms") or 0) > window_ms:
            j += 1
        k = max(range(j, i + 1), key=lambda m: probs[m])
        drop = probs[k] - probs[i]
        if drop > 0.05:
            out.append({"t_ms": t, "from_t_ms": int(states[k].get("t_ms") or 0), "drop": round(drop, 3),
                        "p": round(probs[i], 3)})
    out.sort(key=lambda r: -r["drop"])
    picked: list[dict] = []
    for r in out:  # one per window
        if all(abs(r["t_ms"] - p["t_ms"]) > window_ms for p in picked):
            picked.append(r)
        if len(picked) >= top:
            break
    return sorted(picked, key=lambda r: r["t_ms"])
