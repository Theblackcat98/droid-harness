"""Behavior-cloning ranker for the Clash Royale utility policy (M8).

``CRPolicy`` calls ``ranker.rescore(state, candidates)`` after the rules propose and the
constraints filter (``policy/utility.py``). ``rescore`` must return *the same* candidates,
in the same order, with a new 0..1 ``score`` so the policy's ``act_threshold`` keeps its
meaning.

One encoder is shared by the offline labeler (``label_from_video``) and the live ONNX
loader below. The feature order and the deck order are written to a sidecar ``.json`` next
to the model; the loader refuses a mismatch (a shifted one-hot would silently score the
wrong card). Training is plain numpy so it runs on the CPU-only host; only ONNX export and
inference need ``onnx`` / ``onnxruntime``.
"""

from __future__ import annotations

import argparse
import json
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

# Layout constants. They are part of the on-disk contract (see model_metadata()).
TOWER_KEYS = ("me_left", "me_right", "me_king", "opp_left", "opp_right", "opp_king")
TOWER_REF_HP = 4000.0  # only a normalizer; relative tower HP is what the model needs
POSTURES = ("OPENING", "BALANCED", "DEFEND_LEAD", "PRESS", "SPELL_CYCLE")
PHASES = ("x1", "x2", "ot_x2", "ot_x3")
HASH_BUCKETS = 16
ARENA_W, ARENA_H = 18.0, 32.0
MAX_ELIXIR = 10.0
FORMAT = "droid-harness-ranker/1"
DEFAULT_MAX_CANDIDATES = 32


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a ``GameState``/``Candidate`` or a JSON digest dict."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    try:
        a, b = value
        return float(a), float(b)
    except (TypeError, ValueError):
        return default


def state_dim(deck: Sequence[str]) -> int:
    return 3 + len(deck) + 2 + len(TOWER_KEYS) + len(POSTURES) + len(PHASES)


def candidate_dim(deck: Sequence[str]) -> int:
    return len(deck) + 1 + 2 + HASH_BUCKETS


def encode_state(state: Any, deck: Sequence[str]) -> np.ndarray:
    """Compact state vector: elixir, hand multi-hot, crowns, towers, posture, phase."""
    deck = list(deck)
    hand = _get(state, "hand", []) or []
    if not isinstance(hand, (list, tuple)):
        hand = []
    opp_lo, opp_hi = _as_pair(_get(state, "opp_elixir", (5.0, 5.0)), (5.0, 5.0))
    crowns = _as_pair(_get(state, "crowns", (0, 0)), (0.0, 0.0))
    towers = _get(state, "towers", {}) or {}

    vec: list[float] = [
        float(_get(state, "my_elixir", 0.0) or 0.0) / MAX_ELIXIR,
        opp_lo / MAX_ELIXIR,
        opp_hi / MAX_ELIXIR,
    ]
    vec += [1.0 if card in hand else 0.0 for card in deck]
    vec += [crowns[0] / 3.0, crowns[1] / 3.0]
    for key in TOWER_KEYS:
        hp = float(towers.get(key, 0.0) or 0.0)
        vec.append(min(max(hp / TOWER_REF_HP, 0.0), 1.0))
    posture = _get(state, "posture", "BALANCED") or ""
    vec += [1.0 if posture == p else 0.0 for p in POSTURES]
    phase = _get(state, "phase", "x1") or ""
    vec += [1.0 if phase == p else 0.0 for p in PHASES]
    return np.asarray(vec, np.float32)


def encode_candidate(candidate: Any, deck: Sequence[str]) -> np.ndarray:
    """Candidate vector: card one-hot, rule score, tile x/y, hashed rule id."""
    deck = list(deck)
    card = _get(candidate, "card", None)
    score = _get(candidate, "score", 0.0) or 0.0
    tx, ty = _as_pair(_get(candidate, "tile", (0.0, 0.0)), (0.0, 0.0))
    rule_id = str(_get(candidate, "rule_id", "") or "")

    vec: list[float] = [1.0 if card == c else 0.0 for c in deck]
    vec.append(float(score))
    vec += [tx / ARENA_W, ty / ARENA_H]
    bucket = zlib.crc32(rule_id.encode()) % HASH_BUCKETS
    vec += [1.0 if i == bucket else 0.0 for i in range(HASH_BUCKETS)]
    return np.asarray(vec, np.float32)


def model_metadata(deck: Sequence[str], *, hidden: Sequence[int], max_candidates: int,
                   trained_rows: int, epochs: int) -> dict:
    deck = list(deck)
    return {
        "format": FORMAT,
        "deck": deck,
        "state_dim": state_dim(deck),
        "candidate_dim": candidate_dim(deck),
        "input_dim": state_dim(deck) + candidate_dim(deck),
        "state_layout": [
            {"name": "my_elixir", "size": 1},
            {"name": "opp_elixir_bounds", "size": 2},
            {"name": "hand_multi_hot", "size": len(deck), "classes": deck},
            {"name": "crowns", "size": 2},
            {"name": "towers_hp", "size": len(TOWER_KEYS), "classes": list(TOWER_KEYS),
             "normalizer": TOWER_REF_HP},
            {"name": "posture_one_hot", "size": len(POSTURES), "classes": list(POSTURES)},
            {"name": "phase_one_hot", "size": len(PHASES), "classes": list(PHASES)},
        ],
        "candidate_layout": [
            {"name": "card_one_hot", "size": len(deck), "classes": deck},
            {"name": "rule_score", "size": 1},
            {"name": "tile_xy", "size": 2, "normalizer": [ARENA_W, ARENA_H]},
            {"name": "rule_id_hash", "size": HASH_BUCKETS, "buckets": HASH_BUCKETS},
        ],
        "hidden": list(hidden),
        "max_candidates": int(max_candidates),
        "output": "sigmoid_probability",
        "trained_rows": int(trained_rows),
        "epochs": int(epochs),
    }


# --- tiny numpy MLP -----------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _init_params(dim: int, hidden: Sequence[int], rng: np.random.Generator) -> list[list[np.ndarray]]:
    sizes = [dim, *hidden, 1]
    params: list[list[np.ndarray]] = []
    for i, (n_in, n_out) in enumerate(zip(sizes, sizes[1:], strict=False)):
        scale = np.sqrt(2.0 / n_in) if i < len(sizes) - 2 else np.sqrt(1.0 / n_in)
        w = (rng.standard_normal((n_in, n_out)) * scale).astype(np.float32)
        params.append([w, np.zeros(n_out, np.float32)])
    return params


def _forward(x: np.ndarray, params: list[list[np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray]]:
    acts = [x]
    h = x
    for i, (w, b) in enumerate(params):
        z = h @ w + b
        if i < len(params) - 1:
            h = np.maximum(z, 0.0)
            acts.append(h)
        else:
            h = z
    return _sigmoid(h), acts


def _grads(x: np.ndarray, y: np.ndarray, weight: np.ndarray,
           params: list[list[np.ndarray]], probs: np.ndarray,
           acts: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    # Weighted BCE: dL/dlogit = w * (p - y) / sum(w).
    d = (weight.reshape(-1, 1) * (probs - y)) / max(float(weight.sum()), 1e-9)
    grads: list[tuple[np.ndarray, np.ndarray]] = []
    for i in reversed(range(len(params))):
        w, _ = params[i]
        grads.append((acts[i].T @ d, d.sum(axis=0)))
        if i > 0:
            d = (d @ w.T) * (acts[i] > 0.0)
    return list(reversed(grads))


def _adam_step(params: list[list[np.ndarray]], grads: list[tuple[np.ndarray, np.ndarray]],
               m: list[list[np.ndarray]], v: list[list[np.ndarray]], t: int,
               lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
    for i, (gw, gb) in enumerate(grads):
        for j, g in enumerate((gw, gb)):
            m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g
            v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * (g * g)
            mhat = m[i][j] / (1.0 - beta1 ** t)
            vhat = v[i][j] / (1.0 - beta2 ** t)
            params[i][j] -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)


def _state_vector(value: Any, deck: Sequence[str]) -> np.ndarray:
    if isinstance(value, Mapping):
        return encode_state(value, deck)
    return np.asarray(value, np.float32)


def build_examples(rows: Sequence[dict], deck: Sequence[str] | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """Flatten labeled decisions into (candidate row, played?, outcome weight) examples."""
    rows = list(rows)
    if not rows:
        raise ValueError("no labeled rows")
    deck = list(deck or rows[0].get("deck") or [])
    if not deck:
        raise ValueError("labeled rows carry no deck order; pass deck=...")
    sdim, cdim = state_dim(deck), candidate_dim(deck)

    xs, ys, ws = [], [], []
    used_rows = skipped = positives = 0
    for row in rows:
        row_deck = row.get("deck")
        if row_deck is not None and list(row_deck) != deck:
            raise ValueError(f"row deck order differs from {deck}: {row_deck}")
        candidates = row.get("candidates") or []
        chosen = int(row.get("chosen_index", -1))
        if not candidates or not (0 <= chosen < len(candidates)):
            skipped += 1
            continue
        state = _state_vector(row.get("state"), deck)
        if state.shape != (sdim,):
            raise ValueError(f"state has {state.shape[0]} values, model expects {sdim}")
        weight = float(row.get("weight", 1.0))
        for i, candidate in enumerate(candidates):
            cand = encode_candidate(candidate, deck)
            if cand.shape != (cdim,):
                raise ValueError(f"candidate has {cand.shape[0]} values, model expects {cdim}")
            xs.append(np.concatenate([state, cand]))
            ys.append(1.0 if i == chosen else 0.0)
            ws.append(weight)
            positives += int(i == chosen)
        used_rows += 1
    if not xs:
        raise ValueError("no usable rows (need candidates + a valid chosen_index)")
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, np.float32).reshape(-1, 1)
    w = np.asarray(ws, np.float32)
    return x, y, w, deck, {"rows": used_rows, "candidate_rows": len(x), "positives": positives,
                           "skipped": skipped}


def export_onnx(params: list[list[np.ndarray]], out: Path, input_dim: int, max_candidates: int,
                meta: dict) -> Path:
    """Static-shape opset-17 ONNX: (max_candidates, input_dim) -> sigmoid score."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as e:  # pragma: no cover - depends on the host
        raise RuntimeError("onnx is required to export a ranker (pip install onnx)") from e

    nodes, inits = [], []
    x = "features"
    for i, (w, b) in enumerate(params):
        prefix = f"fc{i}"
        wn, bn = f"{prefix}.weight", f"{prefix}.bias"
        inits.append(numpy_helper.from_array(w, wn))
        inits.append(numpy_helper.from_array(b, bn))
        mm, add = f"{prefix}.mm", f"{prefix}.add"
        nodes.append(helper.make_node("MatMul", [x, wn], [mm]))
        nodes.append(helper.make_node("Add", [mm, bn], [add]))
        if i < len(params) - 1:
            relu = f"{prefix}.relu"
            nodes.append(helper.make_node("Relu", [add], [relu]))
            x = relu
        else:
            nodes.append(helper.make_node("Sigmoid", [add], ["score"]))
            x = "score"

    graph = helper.make_graph(
        nodes, "cr_ranker",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [max_candidates, input_dim])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, [max_candidates, 1])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="droid-harness")
    model.ir_version = 9
    for key in ("format", "deck", "state_dim", "candidate_dim", "input_dim", "max_candidates", "output"):
        value = meta[key]
        model.metadata_props.add(key=key, value=value if isinstance(value, str) else json.dumps(value))
    onnx.checker.check_model(model)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    return out


def train_ranker(rows: Sequence[dict], out_onnx: str | Path, meta_out: str | Path | None = None, *,
                 epochs: int = 60, hidden: Sequence[int] = (64, 32), lr: float = 0.01,
                 batch_size: int = 16, seed: int = 0, max_candidates: int = DEFAULT_MAX_CANDIDATES,
                 deck: Sequence[str] | None = None) -> dict:
    """Behavior cloning on labeled decisions; writes ONNX + sidecar order metadata."""
    x, y, weight, deck, stats = build_examples(rows, deck)
    input_dim = state_dim(deck) + candidate_dim(deck)
    if x.shape[1] != input_dim:
        raise ValueError(f"feature width {x.shape[1]} != state+candidate {input_dim}")

    rng = np.random.default_rng(seed)
    params = _init_params(input_dim, hidden, rng)
    m = [[np.zeros_like(p) for p in layer] for layer in params]
    v = [[np.zeros_like(p) for p in layer] for layer in params]
    n, t = len(x), 0
    order = np.arange(n)
    for _ in range(int(epochs)):
        rng.shuffle(order)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            probs, acts = _forward(x[idx], params)
            grads = _grads(x[idx], y[idx], weight[idx], params, probs, acts)
            t += 1
            _adam_step(params, grads, m, v, t, lr)

    meta = model_metadata(deck, hidden=hidden, max_candidates=max_candidates,
                          trained_rows=stats["rows"], epochs=int(epochs))
    meta.update({k: stats[k] for k in ("rows", "candidate_rows", "positives", "skipped")})

    probs, _ = _forward(x, params)
    w = weight.reshape(-1, 1)
    meta["train_loss"] = float(-(w * (y * np.log(probs + 1e-7)
                                       + (1 - y) * np.log(1 - probs + 1e-7))).sum() / w.sum())
    meta["train_accuracy"] = float(((probs >= 0.5) == (y >= 0.5)).mean())

    out_onnx = Path(out_onnx)
    export_onnx(params, out_onnx, input_dim, int(max_candidates), meta)
    meta_out = Path(meta_out) if meta_out else Path(str(out_onnx) + ".json")
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2) + "\n")
    meta["onnx"] = str(out_onnx)
    meta["meta"] = str(meta_out)
    return meta


def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"labeled rows not found: {p}")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# --- live loader --------------------------------------------------------------------

def _metadata_path(model_path: Path, meta_path: str | Path | None) -> Path:
    if meta_path is not None:
        return Path(meta_path)
    candidates = (Path(str(model_path) + ".json"), model_path.with_suffix(".json"))
    for p in candidates:
        if p.exists():
            return p
    raise ValueError(f"ranker metadata missing next to {model_path} "
                     f"(expected {candidates[0].name} or {candidates[1].name})")


class OnnxRanker:
    """Drop-in for ``CRPolicy(ranker=...)``; scores candidates on the 0..1 utility scale."""

    def __init__(self, model_path: str | Path, meta_path: str | Path | None = None, *,
                 deck: Sequence[str] | None = None,
                 providers: Sequence[str] | None = None) -> None:
        model_path = Path(model_path)
        try:
            model_path.read_bytes()
        except OSError as e:
            raise ValueError(f"ranker model missing or unreadable: {model_path} ({e})") from e
        mp = _metadata_path(model_path, meta_path)
        try:
            meta = json.loads(mp.read_text())
        except (OSError, ValueError) as e:
            raise ValueError(f"ranker metadata missing or unreadable: {mp} ({e})") from e
        if meta.get("format") != FORMAT:
            raise ValueError(f"ranker metadata {mp} has format {meta.get('format')!r}, expected {FORMAT!r}")
        if deck is not None and list(deck) != list(meta.get("deck") or []):
            raise ValueError(f"ranker deck order {meta.get('deck')} does not match the strategy deck "
                             f"{list(deck)}; retrain or fix learned_ranker")

        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(model_path),
                                                 providers=list(providers or ["CPUExecutionProvider"]))
        except Exception as e:  # ORT raises several unrelated exception types
            raise ValueError(f"ranker ONNX unreadable: {model_path} ({e})") from e

        inp = self._session.get_inputs()[0]
        shape = inp.shape
        if len(shape) != 2 or not all(isinstance(s, int) and s > 0 for s in shape):
            raise ValueError(f"ranker ONNX input must be a static (candidates, features) shape, got {shape}")
        self.input_name = inp.name
        self.input_dim = int(shape[1])
        self.max_candidates = int(shape[0])
        if self.input_dim != int(meta.get("input_dim", -1)):
            raise ValueError(f"ranker ONNX input width {self.input_dim} != metadata "
                             f"input_dim {meta.get('input_dim')}")
        self.meta = meta
        self.deck = list(meta["deck"])
        self.state_dim = int(meta["state_dim"])
        self.candidate_dim = int(meta["candidate_dim"])
        self.model_path = model_path

    def rescore(self, state: Any, candidates: Sequence[Any]) -> list[Any]:
        """Return the same candidates in order with model scores (utility-scale floats)."""
        candidates = list(candidates)
        if not candidates:
            return candidates
        sv = encode_state(state, self.deck)
        rows = np.stack([np.concatenate([sv, encode_candidate(c, self.deck)]) for c in candidates])
        scores: list[float] = []
        for start in range(0, len(rows), self.max_candidates):
            chunk = rows[start:start + self.max_candidates].astype(np.float32)
            batch = np.zeros((self.max_candidates, self.input_dim), np.float32)
            batch[:len(chunk)] = chunk
            out = self._session.run(None, {self.input_name: batch})[0]
            scores.extend(float(s) for s in np.asarray(out).reshape(-1)[:len(chunk)])
        return [replace(c, score=s) for c, s in zip(candidates, scores, strict=False)]


def load_ranker(model_path: str | Path, meta_path: str | Path | None = None, *,
                deck: Sequence[str] | None = None,
                providers: Sequence[str] | None = None) -> OnnxRanker:
    return OnnxRanker(model_path, meta_path, deck=deck, providers=providers)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="train_ranker")
    ap.add_argument("--labels", required=True, help="JSONL from learn.label_from_video")
    ap.add_argument("--out", default="models/ranker.onnx")
    ap.add_argument("--meta", default=None, help="sidecar JSON (default: <out>.json)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    a = ap.parse_args(argv)
    meta = train_ranker(load_rows(a.labels), a.out, a.meta, epochs=a.epochs, hidden=tuple(a.hidden),
                        lr=a.lr, batch_size=a.batch, seed=a.seed, max_candidates=a.max_candidates)
    print(f"exported {meta['onnx']} (+ {meta['meta']}); rows={meta['rows']} "
          f"loss={meta['train_loss']:.4f} acc={meta['train_accuracy']:.3f}")


if __name__ == "__main__":
    main()
