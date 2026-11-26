# plan.md — Low-Latency Android Agent Harness (v2, unified)

**Reference game:** Clash Royale · **Host:** Linux + AMD RX 7900 XT (20 GB) + 64 GB RAM · **Devices:** two dedicated Android phones over USB, two throwaway accounts
**Status:** blueprint v2 · Nov 2026 · merges plan v1, the "Aegis-Droid" plan, and the critique of v1 (see `review.md` for what changed and why)


> **Build notes (v2.1, from implementing this plan).** Five things changed once the code ran:
> 1. **Opponent elixir interval (§7):** the 10 cap *collapses* the interval (a full bar is known); uncertainty comes from unknown card costs and missed plays. The earlier wording had this backwards.
> 2. **Recording (§9.7):** writing SPS/PPS first isn't enough. A file that starts mid-stream holds P-frames whose key frame it lacks, so the recorder keeps a pre-roll back to the last key frame.
> 3. **Card matching (§6.2):** masked normalized cross-correlation on lightly blurred luma replaced equalized lightness + `TM_CCORR_NORMED`, which failed on H.264-compressed frames.
> 4. **Offense yields to defense:** the closed-loop test caught the bot counter-attacking the other lane while a Hog walked in undefended. Offense rules now stand down while an unanswered threat is inbound (`attack_blocked_by_threat_eta_s`).
> 5. **Engagements (§9.3):** a lane stays "live" while a unit is still advancing toward the river, not only once it has crossed, otherwise one exchange splits in two.

---

## 0. Decisions at a glance

1. **No LLM in the reflex loop.** Capture → perceive → decide → touch runs on deterministic code plus small ONNX models, targeting **≤ 60 ms p50 / ≤ 100 ms p99**. LLMs work in two slower loops: an optional, non-blocking in-match advisor and an offline coach that proposes one strategy change at a time.
2. **One persistent scrcpy session per phone is the device bus.** Video and touches ride `scrcpy-server`'s sockets through your own small Python client, with a per-device port. No `screencap`, no `adb shell input` in the hot path.
3. **Perception = fixed-region HUD parsing + one YOLO26 detector + a tracker.** Card matching works on lightness with masks, so greyed-out (unaffordable) and selected cards still classify.
4. **Decisions in three layers:** Tier 0 reflex invariants (lethal spell, elixir cap, emergency defense) → a posture (OPENING, BALANCED, DEFEND_LEAD, PRESS, SPELL_CYCLE) → utility-scored rules. Placements lead moving targets by deploy time, spell travel time, and measured action latency.
5. **A strategy is versioned code + YAML params + optional model weights, content-hashed.** Every decision is logged with the candidates it beat; every lane engagement gets a value and a root-cause tag.
6. **Win rate is the gate, not the guide.** Cheap filters first (replay diff, shadow mode, engagement value), then an SPRT with an outcome-blind inclusion rule, device/account swapping, and game-version pinning.
7. **Only strategy problems reach the coach.** Perception, execution, and latency problems go to their own fix queues.
8. **The same harness drives any Android app:** accessibility tree first, then templates/OCR, then a GUI-agent model with a Set-of-Mark overlay. Successful action sequences are cached and replayed.

---

## 1. Goals, non-goals, scope

| ID | Goal | Measured by |
|---|---|---|
| G1 | Fastest decision-to-action | Per-stage timestamps on every frame; loopback round-trip test (§5.4) |
| G2 | Highest win rate | Win rate vs a fixed opponent pool under a pinned game version (§10) |
| G3 | Calculated, iterable strategies | Every change is a hypothesis with an ID, a test, and a recorded result (§9–§11) |
| G4 | Reusable for general Android use | Core harness + domain plugins; same logging and experiment tables (§13–§14) |

**Non-goals (v1 of the build):** reading game memory, sniffing traffic, modifying the APK, emulators, champion-heavy decks, 2v2.

### Scope and account risk
Supercell's support pages call bots cheating, against its terms of service, and grounds for a heavy account penalty; its Safe & Fair Play policy says prohibited third-party software leads to a permanent ban. So:
- Use **throwaway accounts** only, never your main Supercell ID.
- Run only in **Training Camp** (a skilled built-in bot since Oct 2023, no resources won or lost), **Practice Mode** (a bot with the exact deck and levels of a non-champion opponent from your battle log), and **friendly battles between your own two accounts**. Never ladder/ranked against real people.
- The general-Android part (§13) carries none of this risk.

### License note
Ultralytics YOLO is AGPL-3.0 (or a paid enterprise license). That's fine for a private project. If you ever publish the harness or run it as a service, switch the detector to an Apache-2.0 model such as RF-DETR.

---

## 2. Architecture

```
 PHONE (Android 12+)                          HOST (Linux, RX 7900 XT)
┌────────────────────────────┐  USB 3  ┌───────────────────────────────────────────────────────────┐
│ scrcpy-server (app_process) │        │ T1 video:   socket → tee(T4) → H.264 decode → [latest slot]│
│  ├ MediaCodec H.264 encoder ├─video─▶│                                          │                 │
│  ├ InputManager injection   │◀─ctrl──┤ T3 executor ◀── timed actions ◀── T2 brain:                │
│  └ display power control    │        │   (TCP_NODELAY)        HUD → detector → tracker → state    │
└────────────────────────────┘        │                        → Tier 0 → posture → utility rules  │
 (general mode: a11y Portal app)       │                        → lead-targeted placement            │
                                       │ T4 recorder: raw .h264 (config first) + JSONL, drops if full │
                                       │ T5 liveness: socket/process/ADB ping, stuck-screen detector  │
                                       │ P2 advisor (optional, separate process): LLM → intent        │
                                       └───────────────────────────────────────────────────────────┘
 OFFLINE (L3): DuckDB/Parquet → engagements → triage → { STRATEGY → coach → Proposal → funnel → SPRT }
                                                       { PERCEPTION → labeling/calibration queue    }
                                                       { EXECUTION / LATENCY → bug & profiling queue }
```

### Three loops

| Loop | Cadence | What runs | May block L1? |
|---|---|---|---|
| **L1 Reflex** | every frame (30–60 Hz); decisions are event-driven | capture, HUD, detector, tracker, Tier 0, posture, rules, executor | — |
| **L2 Intent** (optional) | every 3–5 s, async | LLM reads a compact *text* state, sets intent knobs and may suggest a posture | **Never.** L1 reads the latest intent; stale intent expires |
| **L3 Learning** | between batches / overnight | engagements, triage, coach, Optuna, model training, SPRT | n/a |

### Threads and processes
- **T1 `video`** — reads the socket, tees every packet to T4, decodes, overwrites a size-1 slot.
- **T2 `brain`** — wakes on a new frame; perception → state → policy → executor.
- **T3 `executor`** — time-ordered action queue; sends at `execute_at`.
- **T4 `recorder`** — bounded queue to disk; drops and counts if full; never back-pressures T2.
- **T5 `liveness`** — socket and process health, periodic ADB ping, stuck-screen detection. Not in the hot path.
- **P2 `advisor`** — separate process so its network/GPU work can't stall the brain.

A `threading.Condition` guards the latest-frame slot; that's enough with threads (Python has no lock-free atomics to lean on). If profiling shows GIL contention, move T2 into its own process and pass frames via `multiprocessing.shared_memory`. Port T1/T3 to Rust only if Python overhead exceeds ~5 ms at p99.

---

## 3. Latency budget

Targets to validate in Phase 0 — engineering estimates, not measurements. scrcpy itself quotes 35–70 ms end-to-end latency for its mirroring.

| Stage | Target p50 | Notes |
|---|---|---|
| Game renders frame → encoder output | 10–25 ms | Device-dependent, largest term |
| USB transfer | < 2 ms | USB 3, not Wi-Fi |
| Decode (~460×1024 H.264, software, 1 thread) | 2–5 ms | Measure VA-API too |
| Color convert + crops | ~1 ms | Use the Y plane where grayscale is enough |
| HUD parse | < 1 ms | Fixed ROIs |
| Arena detector (YOLO26-n/s, FP16) | 3–8 ms | Benchmark MIGraphX vs PyTorch-ROCm |
| Tracker + state + Tier 0 + posture + rules | < 2 ms | |
| Control message → injected | 1–3 ms | `TCP_NODELAY` |
| Game picks up the touch | 0–17 ms | Next game frame at 60 Hz |
| **Frame-out → touch-injected** | **≈ 35–60 ms** | |

### Design rules for speed
1. **Drop decoded frames, never packets.** H.264 frames reference earlier ones, so every packet must be decoded; the "latest frame wins" slot applies after decode. The decoder must therefore keep up with the stream at your chosen resolution.
2. **No bitstream parser.** Feed each scrcpy packet to the decoder whole, prepending the config packet (SPS/PPS) to the next one. A parser waits for the next start code before emitting a frame.
3. **Single-threaded, low-delay decoder.** Frame threading adds (threads − 1) frames of delay.
4. **`TCP_NODELAY`** on the control socket.
5. **Static ONNX input shapes + warm-up** (MIGraphX compiles per shape).
6. **Adaptive detector cadence:** every frame if it fits, else every 2nd frame with tracker extrapolation.
7. **Event-driven decisions** (new threat, elixir threshold, card cycled, phase change, posture change) plus a 250 ms heartbeat.
8. **Timed actions** via `execute_at_ms`: deciding fast and placing at the right moment are separate.
9. **Resolution is a sweep, not a guess.** Test `max_size` 720 / 1024 / 1280. On a portrait phone, 720 on the long edge is only ~324 px wide, so check small-unit recall, not just latency. Start at 1024.
10. **Phone hygiene:** USB 3, cooling, fixed high refresh rate, battery saver off, stay-awake on, **display panel off during unattended runs** (§5.2).

> **Where speed stops paying:** once p99 is under ~100 ms you're several times faster than human reaction (~200–250 ms). After that, perception accuracy and decision quality bound the win rate.

---

## 4. Recommended stack

| Layer | Pick | Alternatives | Avoid (why) |
|---|---|---|---|
| Transport | ADB over USB 3 | Wi-Fi ADB | Wi-Fi for the hot path (jitter) |
| Capture + input | **scrcpy-server v5.x**, own Python client (§5) | `scrcpy --v4l2-sink` (prototype); MaaFramework (MaaTouch + minicap) | `adb exec-out screencap` (hundreds of ms/frame); `adb shell input` (starts a new process per command) |
| Decode | **PyAV** software decode, 1 thread, low-delay | VA-API via PyAV | A parser in front of the decoder; frame threading |
| Image ops | OpenCV + NumPy | — | — |
| Detector | **Ultralytics YOLO26-n/s** → ONNX (NMS-free head) | RF-DETR (Apache-2.0), YOLO11 | Prompting a VLM per frame |
| Inference on your GPU | **ONNX Runtime + MIGraphX EP**, or **PyTorch-ROCm** — benchmark both | ORT-CPU / OpenVINO for tiny models | The ONNX Runtime ROCm EP (deleted upstream); TensorRT/NVDEC (NVIDIA-only) |
| Tracking | ByteTrack (`supervision` / `ultralytics`) | OC-SORT | — |
| HUD digits | Templates / tiny CNN | RapidOCR (ONNX) for free text | Full OCR pipelines on full frames; Tesseract on game fonts |
| Policy | Tier 0 invariants + postures + **utility-scored rules** + ONNX ranker | Behavior trees | String-eval rules in YAML (untestable) |
| Learned policy | PyTorch-ROCm: BC → offline RL (IQL / Decision Transformer) | — | Online RL from scratch (~12 matches/hour) |
| Hot-path logging | `orjson` JSONL + raw `.h264` | — | Synchronous DB writes in the loop |
| Warehouse | **DuckDB** over Parquet | SQLite, Polars | Unindexed text logs |
| Model-run tracking | MLflow (local) | W&B | — |
| Tuning | **Optuna** | Nevergrad / CMA-ES | Running several optimizer families at once |
| Statistics | SciPy + own SPRT (§10.5) | — | Eyeballing win rates under ~200 games |
| Offline coach | Claude via API | Local coder model via llama.cpp | Un-versioned prompt/rule edits |
| GUI agent (general apps) | GUI-Owl-1.5 (2B–8B) or UI-TARS-1.5-7B | Qwen3-VL | A VLM call per step when the tree or cache suffices |
| App UI tree | **DroidRun Portal** | `uiautomator2` | UI-tree-only perception for games, canvases, custom WebViews |
| Labeling | CVAT or Label Studio | FiftyOne | — |
| Game data | Official Clash Royale API + RoyaleAPI `cr-api-data` | — | Hard-coded card stats |

---

## 5. Device I/O layer

### 5.1 Per-device configuration

```yaml
# configs/device_profiles/phone-a.yaml
serial: R5CT1234ABC
local_port: 27183            # unique per phone
account: throwaway-1
scrcpy_server_version: "5.0" # must equal the pushed jar
max_size: 1024
max_fps: 60
video_bit_rate: 8000000
video_encoder: null          # set from `scrcpy --list-encoders`
display_power_on_byte: 1     # payload for "on" in your pinned version (see §5.2)
rois: configs/games/clash_royale/rois/phone-a.yaml
```

Server parameters: `video_codec=h264`, `max_size`, `max_fps`, `video_bit_rate`, `video_codec_options=priority:int=0,latency:int=0` (realtime/low-latency encoder hints; some encoders ignore them — A/B with the loopback test), `audio=false`, `control=true`, `tunnel_forward=true`, `send_frame_meta=true`, optional `crop=W:H:X:Y`, `stay_awake=true`.

Protocol details change between versions, and the server refuses a client of a different version. **Pin one version and verify every struct against that release's `doc/develop.md`.**

### 5.2 Client skeleton

```python
# src/core/transport/scrcpy_bridge.py
import random, socket, struct, subprocess, threading, time
from dataclasses import dataclass
import av, numpy as np

SERVER_PATH = "/data/local/tmp/scrcpy-server.jar"
PTS_MASK = (1 << 62) - 1
FLAG_CONFIG = 1 << 63
HDR = struct.Struct(">QI")                    # pts_and_flags, packet size
TOUCH = struct.Struct(">BBqiiHHHII")          # 32 bytes — verify for your version
TYPE_INJECT_TOUCH, TYPE_SET_DISPLAY_POWER = 2, 10
DOWN, UP, MOVE = 0, 1, 2


@dataclass
class Frame:
    seq: int
    img: np.ndarray
    pts_us: int
    t_recv_ns: int
    t_decoded_ns: int


class LatestFrame:
    def __init__(self):
        self._f, self._cv = None, threading.Condition()

    def put(self, f: Frame):
        with self._cv:
            self._f = f
            self._cv.notify_all()

    def wait_newer(self, seq: int, timeout: float = 0.5) -> Frame | None:
        with self._cv:
            self._cv.wait_for(lambda: self._f is not None and self._f.seq > seq, timeout)
            return self._f if self._f and self._f.seq > seq else None


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return bytes(buf)


class ScrcpyClient:
    def __init__(self, profile: dict, on_packet=None):
        self.p = profile
        self.serial, self.port = profile["serial"], profile["local_port"]
        self.scid = f"{random.getrandbits(31):08x}"          # unique per session
        self.on_packet = on_packet or (lambda flags, data: None)   # T4 recorder tee
        self.slot = LatestFrame()
        self.size = (0, 0)
        self.running = False
        self.error: Exception | None = None
        self.last_packet_ns = 0

    def _server_args(self):
        p = self.p
        args = [f"scid={self.scid}", "log_level=warn", "video=true", "audio=false",
                "control=true", "video_codec=h264", f"max_size={p['max_size']}",
                f"max_fps={p['max_fps']}", f"video_bit_rate={p['video_bit_rate']}",
                "video_codec_options=priority:int=0,latency:int=0",
                "tunnel_forward=true", "send_frame_meta=true", "send_device_meta=true",
                "send_dummy_byte=true", "send_codec_meta=true", "stay_awake=true"]
        if p.get("video_encoder"):
            args.append(f"video_encoder={p['video_encoder']}")
        return args

    def _adb(self, *a, **kw):
        return subprocess.run(["adb", "-s", self.serial, *a], check=True, **kw)

    def _connect(self) -> socket.socket:
        s = socket.create_connection(("127.0.0.1", self.port), timeout=2)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def _connect_first(self, attempts=50) -> socket.socket:
        # adb forward accepts connect() before the server listens; the dummy byte is the ready signal.
        for _ in range(attempts):
            s = self._connect()
            try:
                recv_exact(s, 1)
                return s
            except (ConnectionError, OSError):
                s.close()
                time.sleep(0.1)
        raise TimeoutError("scrcpy-server never became ready")

    def start(self):
        self._adb("push", "scrcpy-server", SERVER_PATH, capture_output=True)
        self._adb("forward", f"tcp:{self.port}", f"localabstract:scrcpy_{self.scid}")
        self.proc = subprocess.Popen(
            ["adb", "-s", self.serial, "shell", f"CLASSPATH={SERVER_PATH}", "app_process", "/",
             "com.genymobile.scrcpy.Server", self.p["scrcpy_server_version"], *self._server_args()])
        self.video = self._connect_first()          # 1st socket: video
        self.control = self._connect()              # 2nd socket: control
        self.device_name = recv_exact(self.video, 64).rstrip(b"\0").decode()  # sent after all sockets connect
        _codec, w, h = struct.unpack(">III", recv_exact(self.video, 12))
        self.size = (w, h)
        self.video.settimeout(None)
        self.running = True
        threading.Thread(target=self._video_loop, name="T1-video", daemon=True).start()

    def _video_loop(self):
        ctx = av.CodecContext.create("h264", "r")
        ctx.thread_count = 1
        ctx.options = {"flags": "low_delay"}        # option API differs across PyAV versions — verify
        config, seq = b"", 0
        try:
            while self.running:
                flags, n = HDR.unpack(recv_exact(self.video, HDR.size))
                data = recv_exact(self.video, n)
                self.last_packet_ns = t_recv = time.perf_counter_ns()
                self.on_packet(flags, data)          # recorder sees config packets too
                if flags & FLAG_CONFIG:
                    config = data
                    continue
                pkt = av.Packet(config + data)       # whole packet, no parser
                config = b""
                for fr in ctx.decode(pkt):           # every packet decoded; only frames are dropped
                    seq += 1
                    self.size = (fr.width, fr.height)
                    self.slot.put(Frame(seq, fr.to_ndarray(format="bgr24"),
                                        flags & PTS_MASK, t_recv, time.perf_counter_ns()))
        except Exception as e:                       # surfaced to T5 liveness
            self.error = e
            self.running = False

    # --- input --------------------------------------------------------------
    def touch(self, action: int, x: int, y: int, pointer_id: int = 0):
        """x, y in video-frame pixels; w, h must equal the current frame size
        or the server ignores the event."""
        w, h = self.size
        pressure = 0 if action == UP else 0xFFFF
        self.control.sendall(TOUCH.pack(TYPE_INJECT_TOUCH, action, pointer_id,
                                        x, y, w, h, pressure, 0, 0))

    def set_display_power(self, on: bool):
        """Panel off while capture continues (cooler phone, no OLED burn-in).
        Older protocol: power-mode byte (0 = off, 2 = normal); newer: boolean.
        0 means off in both; the 'on' byte comes from the device profile."""
        self.control.sendall(struct.pack(">BB", TYPE_SET_DISPLAY_POWER,
                                         self.p["display_power_on_byte"] if on else 0))
```

Display-off caveats: on some LCD phones only the backlight turns off, and in-display fingerprint readers stop working. Android 15+ also offers `adb shell cmd display power-off 0` as a fallback.

### 5.3 Gestures and the executor
- **Default card play: tap-tap.** Tap the card slot (DOWN, UP ~15–30 ms later), then tap the tile. Hold times are scheduled by the executor, never `sleep` in T2. Tune on-device.
- **Pre-selection:** when the policy commits to a card that's waiting on elixir, select it early; placement is then a single tap at `execute_at`.
- **Drag** stays available for cards or devices where tap-tap is unreliable; measure both.
- **Execution check:** after a placement, the card must leave the hand within ~300 ms and a matching spawn must appear near the target tile within the card's deploy time + 0.5 s. Otherwise, log an `EXECUTION` anomaly (§9.5).
- Debug aids: Developer options → *Show taps* and *Pointer location*.

### 5.4 Loopback latency test
Enable *Show taps*; inject `DOWN` at a quiet spot and record `t_send`; watch decoded frames for the indicator and record `t_seen`; `RTT = t_seen − t_send`. Repeat 200×; store p50/p95/p99 per phone in `latency_runs`. Re-run after any capture, encoder, resolution, or thermal change. Your reaction time ≈ RTT + perception + policy.

### 5.5 Liveness and watchdogs
**scrcpy produces new frames only when the screen changes**, so a frame gap on a static menu or a settled app is normal, not a failure.

| Signal | Meaning | Action |
|---|---|---|
| Socket EOF/error, `client.error` set | Session broken | Reconnect |
| Server process exited | Session broken | Restart server |
| `adb shell true` ping fails for > 5 s (run every 2 s off the hot path) | Device/ADB problem | Reconnect; then `adb reconnect` |
| Frame gap > 250 ms **during battle** | Anomaly (battles always animate) | Log `LATENCY` anomaly; reconnect if > 2 s |
| Frame gap outside battle | Static screen | Nothing — this is the quiescence signal (§13) |
| Same screen ID for > 30 s outside battle | Stuck UI | Back key → re-navigate → force-stop and relaunch the app |

### 5.6 Prototyping shortcuts
- `scrcpy --v4l2-sink=/dev/videoN --no-playback` (Linux, `v4l2loopback`) exposes the stream to OpenCV; use a second control-only server for touches. Day-1 only.
- **MaaFramework** has ready ADB controllers and JSON "recognize → act" pipelines — useful for menu glue and general apps; keep the custom client for the reflex loop.

---

## 6. Perception (Clash Royale pack)

### 6.1 Layout calibration
- Normalized ROIs per phone (`rois/phone-a.yaml`): hand slots ×4, next card, elixir bar, clock, 6 tower-HP fields, arena rectangle.
- Arena pixels ↔ tiles via a homography from ≥ 4 landmarks (bridges, tower centers). The arena is commonly modeled as an 18 × 32 tile grid — confirm against your calibration.
- `calibrate.py` captures a frame, takes clicked landmarks, writes the YAML.

### 6.2 HUD (every frame, < 1 ms)

| Field | Method | Rate |
|---|---|---|
| Hand cards + next card | **Masked lightness matching** against your 8 deck cards (below), or a tiny CNN trained on the same crops | every frame |
| Card affordability | Mean saturation of the slot (greyed = unaffordable); cross-checked against elixir and card cost — disagreement → perception warning | every frame |
| Elixir | HSV threshold on the bar → filled fraction × 10 | every frame |
| Clock | Digit templates | 1 Hz, interpolate |
| Tower HP | Digit templates | 2 Hz |
| Pocket unlocked | Tower-destroyed event | on event |

Card matching that survives greyed-out and selected states:

```python
def match_hand_slot(crop_bgr, templates_L: dict[str, np.ndarray], mask: np.ndarray):
    """templates_L: lightness templates captured on *your* device, several per card
    (affordable, greyed, selected). mask: 255 everywhere except the elixir-cost badge."""
    L = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.equalizeHist(L)                                  # normalize the darker greyed state
    best, best_s = None, -1.0
    for card, tpl in templates_L.items():
        s = float(cv2.matchTemplate(L, tpl, cv2.TM_CCORR_NORMED, mask=mask).max())
        if s > best_s:
            best, best_s = card, s
    return best, best_s                                       # low best_s → perception warning
```

Accept only if `best_s` clears a calibrated threshold *and* beats the runner-up by a margin; otherwise keep the previous reading and flag it. Target ≥ 99.5% on the golden set in every card state.

### 6.3 Arena detector
- **Model:** YOLO26-n first; YOLO26-s if small-unit recall is poor. Fixed input on the arena crop, FP16, ONNX, static shape, warmed up.
- **Classes:** unit, building, and spell-effect types. Ownership comes from the HP bar (§6.4), not doubled classes.
- **Data, in priority order:** (1) synthetic sprite compositing onto empty arenas — the approach of the KataCR paper's generative Clash Royale dataset; (2) KataCR's open detection and replay datasets as a seed (check license and coverage — its stats date from May 2024, so newer cards are missing); (3) active learning: low-confidence and track-inconsistent frames are exported as keyframes to CVAT/Label Studio, retrained weekly.
- **Runtime on the 7900 XT:** benchmark ONNX Runtime + MIGraphX EP vs PyTorch-ROCm; keep the faster. Train on PyTorch-ROCm.

### 6.4 Ownership, health, tracking, events
- For each box, sample the thin strip above it: **blue vs red HP bar → owner**; bar fill ratio → `hp_pct`. When no bar is readable, fall back to the spawn-side prior and track history, and treat `hp_pct = 1.0` while alive.
- ByteTrack → `Unit(track_id, kind, side, tile, vel, hp_pct, first_seen)`.
- Events: `spawn(side, kind, tile, t)` (drives opponent elixir and cycle tracking), `threat(lane, kind, eta_to_tower_s)`, `tower_damaged`, `tower_destroyed` (unlocks pocket placement).

### 6.5 Menu / screen state machine
`HOME → TRAINING_OR_PRACTICE → LOADING → IN_BATTLE → RESULT → HOME`, plus `POPUP`, `UNKNOWN`. Screen ID by perceptual hash of fixed regions + a few templates. Recovery rules are in §5.5.

### 6.6 Perception QA
- **Golden frames:** 500+ labeled frames across arenas, effects, and all card states; CI runs on every model/ROI change.
- **Replay regression:** re-run recorded `.h264` matches through new perception and diff the event logs (spawns, plays, tower HP) against the previous model.
- Metrics: per-class precision/recall, HUD accuracy, tracker ID switches, `perception_warnings` per match.

---

## 7. Game state and opponent model

```python
@dataclass
class Unit:
    track_id: int
    kind: str
    side: Literal["me", "opp"]
    tile: tuple[float, float]
    vel: tuple[float, float]              # tiles / s
    hp_pct: float
    first_seen_ms: int
    conf: float

@dataclass
class GameState:
    t_ms: int
    phase: Literal["x1", "x2", "ot_x2", "ot_x3"]
    my_elixir: float                      # HUD reading fused with the model
    opp_elixir: tuple[float, float]       # (low, high)
    hand: list[str]
    hand_affordable: list[bool]
    next_card: str | None
    units: list[Unit]
    towers: dict[str, int]
    crowns: tuple[int, int]
    opp_deck_seen: list[str]
    opp_hand_possible: set[str]
    pocket_open: dict[str, bool]
    posture: str = "BALANCED"
```

- **Elixir phases (official):** x1 for the first two minutes, x2 in the third; in overtime, x2 in the fourth minute and x3 in the fifth. At x1, about 1 elixir per 2.8 s, capped at 10.
- **Opponent elixir** = generation − costs of detected spawns, kept as an interval; the upper bound clamps at 10 because leaks are invisible.
- **Opponent cycle:** 8 cards, 4 in hand; a played card returns only after four more plays. Once their deck is known, their hand ≈ deck − last 4 played.
- **Card database** (dated snapshot): cost, type, targets, range, speed, **deploy time** (1 s for most troops but not all — e.g., Spear Goblin's deploy delay was cut to 0.5 s in 2025; multi-unit cards deploy units 0.1 s apart), **spell travel speed**, and **crown-tower damage** for spells. Sources: official API `/cards`, RoyaleAPI `cr-api-data`, balance notes.

---

## 8. Decision layer — codified strategy

### 8.1 Layers

| Layer | What | Latency |
|---|---|---|
| **Tier 0 — reflex invariants** | Deterministic overrides that fire before scoring | < 1 ms |
| **Posture** | Deterministic regime classifier that rescales rule weights | < 0.1 ms |
| **Tier 1 — utility rules** | Typed Python rules propose scored candidates; best one wins | < 1 ms |
| **Tier 1b — learned ranker** (later) | ONNX model adds a score bonus to candidates | 1–2 ms |
| **Tier 2 — LLM advisor** (optional) | Async intent + posture suggestion; a strategy variant like any other | seconds, never blocking |
| **Hard constraints** | Legality filters; never learned, never LLM-editable | < 0.1 ms |

Every layer emits logged candidates with a `rule_id` (`t0.*`, `posture`, rule IDs), so all of it is measurable and tunable.

### 8.2 Tier 0 reflex invariants
- `t0.lethal_spell` — a spell in hand whose crown-tower damage ≥ an enemy tower's HP (minus a margin) and that's affordable: cast now.
- `t0.leak_guard` — elixir ≥ threshold for longer than N ms: play the lowest-regret card.
- `t0.emergency_defend` — threat ETA to a low-HP tower < threshold: cheapest effective counter.

Their thresholds live in the strategy YAML and go through the same evaluation funnel.

### 8.3 Postures

```python
Posture = Literal["OPENING", "BALANCED", "DEFEND_LEAD", "PRESS", "SPELL_CYCLE"]

def classify_posture(s: GameState, p: Params, intent: Intent | None) -> Posture:
    if intent and intent.posture and intent.fresh() and intent.posture in p.advisor_may_set:
        return intent.posture                                  # advisor override, allowlisted
    if s.phase in ("ot_x2", "ot_x3") and s.spell_cycles_to_finish(p) <= p.spell_cycle_max_cycles:
        return "SPELL_CYCLE"                                    # chip a tower with spells
    if s.crowns[0] > s.crowns[1] and s.phase != "x1":
        return "DEFEND_LEAD"
    if s.t_ms < p.opening_ms and not s.enemy_units():
        return "OPENING"
    if s.opp_elixir[1] <= p.press_opp_elixir_max:
        return "PRESS"
    return "BALANCED"
```

Posture multipliers are in the strategy YAML (§8.6). The posture is logged on every decision.

### 8.4 Utility rule engine

```python
@dataclass(frozen=True)
class Candidate:
    rule_id: str
    card: str
    slot: int
    tile: tuple[int, int]
    score: float
    execute_at_ms: int | None = None
    reason: str = ""

class DefendLaneThreat:
    id = "defend.lane_threat"
    family = "defend"

    def propose(self, s: GameState, p: Params, intent: Intent):
        for threat in s.threats(max_tiles_to_tower=p.defend_trigger_tiles):
            for slot, card in s.affordable_hand():
                eff = p.counters.get((card, threat.kind), 0.0)
                if eff <= 0:
                    continue
                urgency = max(0.0, 1.0 - threat.eta_to_tower_s / p.defend_horizon_s)
                t_exec = placement.timing(card, threat, s)
                yield Candidate(
                    self.id, card, slot,
                    placement.lead_target(card, threat, s, t_exec),
                    score=eff * urgency - p.elixir_weight * CARD[card].cost / 10,
                    execute_at_ms=t_exec,
                    reason=f"{card} vs {threat.kind} eta={threat.eta_to_tower_s:.1f}s")

class Policy:
    def decide(self, s, intent):
        if (c := self.tier0.check(s, self.p)) is not None:            # reflex invariants first
            return c, [c]
        s.posture = classify_posture(s, self.p, intent)
        mult = self.p.posture_multipliers.get(s.posture, {})
        cands = [c for r in self.rules for c in r.propose(s, self.p, intent)]
        cands = [c for c in cands if self.constraints.ok(s, c)]
        if self.ranker:
            cands = self.ranker.rescore(s, cands)                     # Tier 1b, optional
        if not cands:
            return None, []
        def total(c):
            return c.score * self.p.weights[c.rule_id] * mult.get(family(c.rule_id), 1.0)
        best = max(cands, key=total)
        return (best if total(best) >= self.p.act_threshold else None), cands
```

Starter rules: `defend.lane_threat`, `defend.air_threat`, `offense.win_condition_when_opp_low`, `offense.counter_push`, `spell.value`, `spell.tower_chip`, `cycle.leak_guard`, `opening.safe_first_play`, `x2.pressure_split_lanes`.

### 8.5 Placement and lead targeting
A placed card takes effect after **action latency + deploy time (+ travel time for spells)**; moving targets keep moving meanwhile.

```python
def lead_target(card: Card, threat: Unit, s: GameState, t_exec_ms: int) -> tuple[int, int]:
    lag_s = (t_exec_ms - s.t_ms) / 1000 + EXEC_LATENCY_S        # measured p50 from §5.4, per phone
    t = lag_s + card.deploy_time_s
    target = path.advance(threat, t)                              # follows lane/bridge path, not a straight line
    if card.is_spell:
        for _ in range(2):                                        # travel time depends on the aim point
            t_total = t + card.travel_time_s(KING_TOWER_TILE, target)
            target = path.advance(threat, t_total)
    return grid.snap_legal(target, card, s)
```

- `path.advance` moves a unit along its lane toward its target (bridge, then tower), using its velocity; beyond ~1.5 s, prediction error grows, so rules prefer placements that don't depend on long horizons.
- Defensive buildings and pull placements use fixed, tuned tiles relative to the threat's lane, plus `execute_at` timing.
- Log `predicted_tile` vs where the threat actually was at effect time; that error feeds tuning.

### 8.6 Strategy file

```yaml
strategy: hog-cycle-v4
parent: hog-cycle-v3
deck: [hog_rider, musketeer, ice_golem, cannon, fireball, the_log, ice_spirit, skeletons]
params:
  act_threshold: 0.15
  elixir_weight: 0.6
  defend_trigger_tiles: 7
  defend_horizon_s: 6.0
  min_elixir_to_attack: 7
  spell_value_threshold: 4.0
  opening_ms: 30000
  press_opp_elixir_max: 3.0
  spell_cycle_max_cycles: 3
tier0:
  lethal_margin_hp: 30
  leak_threshold: 9.3
  leak_ms: 800
  emergency_eta_s: 1.5
weights:
  defend.lane_threat: 1.0
  offense.win_condition_when_opp_low: 1.0
  spell.value: 0.8
  spell.tower_chip: 0.6
  cycle.leak_guard: 1.0
posture_multipliers:
  OPENING:     {offense: 0.3, cycle: 1.2}
  DEFEND_LEAD: {offense: 0.5, defend: 1.3}
  PRESS:       {offense: 1.4}
  SPELL_CYCLE: {spell: 2.0, offense: 0.6}
counters: counters/hog-cycle-v4.csv
learned_ranker: null
intent_advisor: false
advisor_may_set: [DEFEND_LEAD, PRESS, SPELL_CYCLE]
```

Pick a v1 deck with no champion and simple, low-variance cards.

### 8.7 Hard constraints
Legal deploy zone (own half; pocket only when unlocked; spells anywhere) · enough elixir at `execute_at` · one placement per card per cycle · minimum spacing between actions · no input outside `IN_BATTLE`.

---

## 9. Strategy tracking and logging

### 9.1 Identity
`strategy_id = sha256(git_sha_of_rules + strategy.yaml + counters.csv + model_checksums)[:12]`. Immutable once used; edits create a child with `parent`.

### 9.2 Schema

```sql
CREATE TABLE strategies (
  strategy_id TEXT PRIMARY KEY, name TEXT, parent_id TEXT, git_sha TEXT,
  params_json JSON, deck TEXT[], created_at TIMESTAMP, created_by TEXT  -- human | coach | optuna
);

CREATE TABLE matches (
  match_id TEXT PRIMARY KEY, started_at TIMESTAMP,
  device TEXT, account TEXT, game_version TEXT, balance_epoch TEXT,
  strategy_id TEXT, opponent_kind TEXT, opponent_id TEXT, opp_deck_seen TEXT[],
  status TEXT,                     -- completed | infra_abort
  result TEXT, my_crowns INT, opp_crowns INT,
  my_tower_hp_end INT, opp_tower_hp_end INT, duration_s DOUBLE, overtime BOOL,
  elixir_leaked DOUBLE, n_actions INT,
  rtt_p50_ms DOUBLE, loop_p99_ms DOUBLE, perception_warnings INT,
  experiment_id TEXT, arm TEXT, video_path TEXT
);

CREATE TABLE decisions (
  match_id TEXT, seq INT, t_match_ms INT, frame_seq INT, eng_id INT,
  state_json JSON, candidates_json JSON, posture TEXT,
  chosen_rule TEXT, chosen_card TEXT, tile_x INT, tile_y INT, score DOUBLE,
  predicted_tile JSON, intent_json JSON,
  lat_decode_ms DOUBLE, lat_perceive_ms DOUBLE, lat_decide_ms DOUBLE, lat_inject_ms DOUBLE,
  exec_ok BOOL, play_value DOUBLE,
  shadow_strategy_id TEXT, shadow_choice_json JSON
);

CREATE TABLE engagements (
  match_id TEXT, eng_id INT, lane TEXT, t_start_ms INT, t_end_ms INT,
  spent_me DOUBLE, spent_opp DOUBLE,
  board_me_start DOUBLE, board_me_end DOUBLE, board_opp_start DOUBLE, board_opp_end DOUBLE,
  tower_dmg_dealt INT, tower_dmg_taken INT, value DOUBLE,
  triage TEXT,                     -- STRATEGY | PERCEPTION | EXECUTION | LATENCY
  triage_evidence JSON
);

CREATE TABLE experiments (
  experiment_id TEXT PRIMARY KEY, title TEXT, hypothesis TEXT, proposer TEXT,
  champion_id TEXT, candidate_id TEXT, opponent_pool TEXT, test TEXT,
  game_version TEXT, balance_epoch TEXT,
  p0 DOUBLE, p1 DOUBLE, alpha DOUBLE, beta DOUBLE,
  status TEXT, games INT, wins INT, losses INT, draws INT, aborts_by_arm JSON, llr DOUBLE,
  decided_at TIMESTAMP, notes TEXT
);
```

Per-frame telemetry goes to its own Parquet per match.

### 9.3 Engagements
Segment each lane into **engagements**: one opens when a unit enters contested ground in that lane or a card is placed there; it closes when neither side has units past the river in that lane for 2 s (cap 25 s). Spells cast into a lane join the open engagement there.

### 9.4 Engagement value and shared credit
Board value for a side = Σ over its units of (card cost ÷ units per card) × `hp_pct`.

```
net_me   = board_me_end  − board_me_start  − spent_me
net_opp  = board_opp_end − board_opp_start − spent_opp
value    = (net_me − net_opp) + λ · (tower_dmg_dealt − tower_dmg_taken)
```

Each of my decisions in the engagement gets `play_value = value × (its elixir ÷ spent_me)`; zero-cost decisions (none in standard play) would get equal shares. Every swing is counted once, everything is observable, and opponent plays are part of the same exchange rather than leaking into unrelated windows. Start with λ = 1 elixir per 100 tower HP and tune.

Play value is for **ranking rules and finding problems**, not for accepting changes — it's correlated with winning but not the same thing.

### 9.5 Root-cause triage (before anything reaches the coach)

| Tag | Detection rules (any) | Routed to |
|---|---|---|
| `PERCEPTION` | HUD elixir vs model disagree by > 0.7 for > 1 s · hand changed with no placement · affordability read contradicts elixir · spawn without a matching play (or vice versa) · tracker ID switches > k · tower HP rises | Labeling/calibration queue; keyframes exported |
| `EXECUTION` | Card didn't leave hand within 300 ms · spawn > 1.5 tiles from target · placement rejected (elixir race) | Executor bug queue |
| `LATENCY` | Loop p99 > 100 ms or frame gap > 250 ms during the engagement | Profiling queue |
| `STRATEGY` | None of the above and `value` < −threshold | **Coach** |

Matches aren't dropped from win-rate tests because of triage tags (§10.2); triage only controls what the coach and the diagnostics see.

### 9.6 Loss stories (STRATEGY engagements only)

```
[match 9f2c… | hog-cycle-v4 vs practice:golem-beatdown | L 0–1 | game 9.x | epoch 2026-11]
eng 7 · left · 1:12–1:31 · value −4.6 · STRATEGY
  1:12 opp Night Witch + Golem bridge left (opp elixir est 0–2) · posture BALANCED
  1:12 we: Cannon @ (8,22), predicted Golem tile (6,19), actual (6,18) · rule defend.lane_threat 0.72
  1:19 Cannon destroyed; −612 left princess HP
  excluded from this story: eng 9 (PERCEPTION: elixir read 8.9 vs model 9.6)
```

### 9.7 Storage
At 8 Mbps the raw stream is ~60 MB per minute, so ~0.2–0.3 GB per match and tens of GB per day of unattended play.
- Keep raw `.h264` (config packet first) for **7 days**.
- Nightly: transcode losses and flagged engagements to ~1 Mbps H.265 for review; export keyframes (PNG) for anomalies to the labeling queue; delete the rest.
- Keep `decisions`, `engagements`, `matches` forever.

### 9.8 Standard queries
Win rate by strategy × opponent × game version (Wilson interval) · engagement value by rule × posture × phase · triage mix over time · lead-target error by card · reaction time (threat appeared → action sent) · loop p99 per phone per day.

---

## 10. Evaluation and iteration

### 10.1 Opponent pools
1. **Training Camp** — first baseline.
2. **Practice gauntlet** — 10–20 archetypes from your battle log (non-champion decks). Keep opponents where the champion wins 15–85%; others don't separate strategies.
3. **Self-play** — friendlies between your two accounts on two phones. **Swap strategy ↔ (phone, account) every game** so phone speed, thermal state, and card levels cancel out. Log the assignment; check afterward whether phone or account predicts results.

### 10.2 Metrics and inclusion rules
- **Primary:** win rate (draws reported separately).
- **Dense secondary:** crown differential, tower-HP differential, engagement value per rule, leak, reaction time, lead-target error.
- **Inclusion rule for win-rate tests (decided before any results):** count **every completed match**. Exclude only *infrastructure aborts* (disconnect, server crash, app restart mid-battle), and compare abort rates between arms — an arm that aborts more is failing, not getting a free pass. Never exclude matches by triage or data quality; that can correlate with the outcome.
- **Game-version pinning:** every match records `game_version` (from `dumpsys package`) and `balance_epoch`. An experiment that spans an update is void and restarts.

### 10.3 Sample-size reality
~3–5 min per match plus menus → **~10–15 matches/hour per phone.**

95% Wilson interval at a 50% observed win rate:

| Games | Interval |
|---|---|
| 50 | 37–63% |
| 100 | 40–60% |
| 200 | 43–57% |
| 400 | 45–55% |

Average games for a Bernoulli SPRT to decide (H0 = 50%, α = β = 0.05):

| H1 win rate | ≈ Elo | Avg games | Hours at 12/h, one phone |
|---|---|---|---|
| 55% | +35 | ~530 | ~44 |
| 58% | +56 | ~205 | ~17 |
| 60% | +70 | ~130 | ~11 |
| 65% | +108 | ~57 | ~5 |

One serious test ≈ one night on two phones. Filter candidates before they reach the SPRT and target improvements worth having (H1 = 58–60%).

### 10.4 Promotion funnel

| Gate | Cost | Catches |
|---|---|---|
| 1. Unit + constraint tests | seconds | Bugs, illegal plays |
| 2. **Offline replay diff** on logged states | minutes | Changes larger than the hypothesis claims; crashes |
| 3. **Shadow mode** (~20 matches) | ~2 h, free | Where the candidate disagrees and whether those engagements were losing ones |
| 4. **Screening** — Thompson sampling across several candidates on crown/tower-HP differential | hours | Clear losers |
| 5. **SPRT** — self-play head-to-head (with swapping) or paired gauntlet | ~1 night | Accept / reject |
| 6. Promote + Strategy Ledger entry | — | Lineage |

**Paired gauntlet:** champion and candidate play the same opponent sequence alternately; run the SPRT on discordant pairs (one wins, one loses) with H0 = 0.5. It removes opponent-difficulty variance but needs more total games.

### 10.5 SPRT

```python
import math

def sprt(wins: int, losses: int, p0=0.50, p1=0.58, alpha=0.05, beta=0.05):
    """Bernoulli SPRT; draws excluded (or use a trinomial GSPRT as Fishtest does)."""
    llr = wins * math.log(p1 / p0) + losses * math.log((1 - p1) / (1 - p0))
    upper, lower = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    if llr >= upper:
        return "accept_H1", llr
    if llr <= lower:
        return "accept_H0", llr
    return "continue", llr
```

Non-regression changes (refactors, perception upgrades) use non-regression bounds, e.g. H0 = 47%, H1 = 50%.

### 10.6 Parameter tuning (Optuna)
Search space = `params`, `tier0`, and `posture_multipliers`. One trial = a fixed block of ~8 gauntlet matches; objective = crown differential + tower-HP differential. TPE sampler, median pruner, ≤ 30 trials per study. The best trial becomes a **candidate** and still goes through §10.4.

---

## 11. LLM roles

### 11.1 Offline coach
After each batch of ~50 matches:
1. Build engagements → triage → report (engagement value by rule/posture, worst matchups, lead-target error, triage mix).
2. Context pack: strategy YAML + rule source + counter table + report + 3–5 **STRATEGY-only** loss stories + the last 30 ledger entries + the share of engagements excluded by triage (so the coach knows data quality).
3. The LLM returns **one** `Proposal`; the harness applies it on a branch and runs the funnel; the result goes to the ledger.

```python
class Proposal(BaseModel):
    title: str
    hypothesis: str                          # cause → change → expected metric move
    evidence: list[str]                      # match/engagement ids, query outputs
    change_type: Literal["param", "tier0", "posture", "counter_table",
                         "rule_new", "rule_edit", "rule_weight"]
    patch: str                               # unified diff
    target_metric: str                       # e.g. "engagement_value[defend.lane_threat]"
    expected_direction: Literal["up", "down"]
    expected_decision_change_pct: float      # checked by the replay-diff gate
    risks: str
```

### 11.2 Guardrails
- Patches may touch only `configs/strategies/**`, `counters/**`, and `src/domains/clash_royale/policy/rules/**` — never constraints, perception, executor, triage, or eval code.
- A rule change ships with a unit test.
- If the replay diff changes far more decisions than claimed, reject before any match.
- One hypothesis per proposal; the ledger blocks near-duplicates of failed ideas unless the context changed (deck, perception model, game version).

```
## EXP-0042 · REJECTED · 2026-12-04 · game 9.x · epoch 2026-11
Hypothesis: Cannon placed late vs Golem pushes; move defend trigger 7→8 tiles.
Candidate 3a91c0e2 vs champion 77b1d4aa · self-play SPRT [0.50, 0.58] (swapped) · 231 games · 112–119 · LLR −2.95
Note: earlier trigger pulled the Golem to the king tower; try placement tile instead (EXP-0045).
```

### 11.3 In-match advisor (optional, off by default)
- Every 3–5 s: ~300-token text state → JSON `Intent {aggression, lane, hold: [cards], posture?}`; 1.5 s timeout; ignored after 6 s; posture overrides limited to `advisor_may_set`.
- Enabled only as a strategy variant that beats the champion in an SPRT.
- **If it wins, distill it:** train a small classifier on its logged (state → intent/posture) outputs, test that as the next candidate, and retire the runtime LLM if it holds.

### 11.4 Hosting and GPU contention
Coach: Claude via API (a local coder model works for cheap overnight runs). Advisor: via API or a small CPU model in system RAM — LLM decoding on the 7900 XT competes with YOLO and spikes p99. Measure p99 with it on.

---

## 12. Learned policy track
1. **Data:** your logs + **auto-labeled human demos** (a card leaves the hand and a matching unit spawns at a tile → a state/action pair). Play 50–100 matches yourself on a bot account to seed it.
2. **Model:** small transformer over unit tokens + hand + affordability + elixir + clock + posture + opponent-cycle estimate. Heads: {no-op, slot 1–4} + a legality-masked tile heat-map. 1–5 M parameters → ONNX → ~1–2 ms.
3. **Training:** behavior cloning → filtered BC (positive engagement value or won matches) → offline RL (IQL, or a return-conditioned Decision Transformer as KataCR used).
4. **Deployment:** shadow → score bonus inside the utility engine → standalone; each step SPRT-gated. The model checksum is part of `strategy_id`.

---

## 13. General Android use

Same core harness and tables (task success instead of wins); a different domain plugin.

**Observation ladder (cheapest first):**
1. **Accessibility tree** via DroidRun Portal (an accessibility service exposing the tree over a content provider and a local HTTP/WebSocket server), with `uiautomator2` behind the same interface. Test Portal on your phones early — some Android skins restrict accessibility grants or background sockets.
2. **Templates / small detectors / OCR** (RapidOCR) for known screens and text the tree can't see (game engines, Flutter canvases, custom WebViews).
3. **GUI-agent model** only when 1–2 fail, with a **Set-of-Mark overlay**: number the candidate elements on the screenshot so the model answers with an element ID instead of raw coordinates.

**Compiled skills (macro cache):** key = `(package, screen_signature, goal)`; value = action script with pre/post-condition checks. Hits replay in milliseconds per step; misses or failed post-conditions call the model and store the successful trace, promoted after k successful replays.

**Settle detection:**

```python
def wait_settled(client, a11y=None, quiet_ms=250, eps=0.002, n=5, timeout_s=5.0) -> bool:
    """Settled = (no new frame for quiet_ms — scrcpy only sends frames on change —
                  OR n consecutive frames differ by < eps, for blinking cursors/spinners)
                 AND the a11y tree hash is unchanged across that period (if available)."""
    t0 = time.monotonic()
    last_hash, low_diff, prev = None, 0, None
    while time.monotonic() - t0 < timeout_s:
        quiet = (time.perf_counter_ns() - client.last_packet_ns) / 1e6 >= quiet_ms
        f = client.slot.wait_newer(prev.seq if prev else 0, timeout=quiet_ms / 1000)
        if f is not None and prev is not None:
            low_diff = low_diff + 1 if frame_diff(prev.img, f.img) < eps else 0
        prev = f or prev
        h = a11y.tree_hash() if a11y else None
        if (quiet or low_diff >= n) and (a11y is None or h == last_hash):
            return True
        last_hash = h
    return False                                   # timeout → log, continue cautiously
```

**Model choices:** GUI-Owl-1.5 is open in 2B/4B/8B/32B/235B sizes and reports 71.6 on AndroidWorld; the small sizes fit beside YOLO on your GPU. AndroidWorld is near saturation (agents above 90%), while the harder MobileWorld shows best results of 51.7% (agentic framework) and 20.9% (end-to-end model) — real apps need the cache and fallbacks. Use Claude via API for hard, rare tasks.

**Text entry:** inject-text message for ASCII; clipboard + paste otherwise.

---

## 14. Repo layout

```
droid-harness/
├─ configs/
│  ├─ device_profiles/{phone-a.yaml, phone-b.yaml}
│  ├─ domains/clash_royale/{cards.json, rois/, counters/}
│  ├─ domains/android_os/defaults.yaml
│  └─ strategies/*.yaml
├─ src/
│  ├─ core/
│  │  ├─ transport/{scrcpy_bridge.py, frame_slot.py, a11y_portal.py, adb.py}
│  │  ├─ actuate/{executor.py, gestures.py}
│  │  ├─ runtime/{orchestrator.py, liveness.py, intent_bus.py}
│  │  ├─ telemetry/{recorder.py, schemas.py, loopback_test.py}
│  │  └─ interfaces.py                    # PerceptionProvider, Policy, ActionMapper
│  ├─ domains/
│  │  ├─ clash_royale/
│  │  │  ├─ perception/{hud.py, cards.py, digits.py, detector.py, ownership.py, tracker.py, grid.py, screens.py}
│  │  │  ├─ state.py · opponent_model.py · elixir.py
│  │  │  ├─ policy/{tier0.py, posture.py, utility.py, constraints.py, placement.py, path.py, learned.py, rules/}
│  │  │  └─ menus.py
│  │  └─ android_os/{perception.py, som.py, ocr.py, skills_cache.py, settle.py, planner.py}
│  └─ flywheel/
│     ├─ engagements.py · triage.py · report.py · loss_stories.py
│     ├─ coach/{proposal.py, apply.py, prompts/}
│     ├─ eval/{sprt.py, gauntlet.py, selfplay.py, shadow.py, replay_diff.py, stats.py}
│     ├─ tuning/optuna_study.py
│     └─ learn/{synth/, label_from_video.py, train_detector.py, train_bc.py, train_offline_rl.py}
├─ ledger/STRATEGY_LEDGER.md
├─ notebooks/dashboard.py
├─ tests/{golden_frames/, replays/, test_rules.py, test_constraints.py, test_protocol.py, test_triage.py}
└─ data/                                   # gitignored; retention per §9.7
```

---

## 15. Implementation phases

| Phase | Size | Build | Exit criteria |
|---|---|---|---|
| **P0 Foundations** | S | Pinned scrcpy server + client (per-device ports), recorder tee, executor, loopback test, liveness, display-off check, Portal check on both phones | Two phones streaming 30 min each with no false reconnects on static screens; RTT p50/p99 per phone; display-off keeps capture running; taps land correctly |
| **P1 Glue + HUD** | M | Menu FSM, Training Camp loop, ROI calibration, HUD with all card states, random-legal policy | 20 unattended matches; HUD ≥ 99.5% on golden frames in every card state |
| **P2 Arena perception** | L | Synthetic generator, YOLO26, MIGraphX vs PyTorch-ROCm, HP-bar ownership, tracker, tiles, resolution sweep | Per-class recall target met on real held-out frames; detector ≤ 10 ms p95; loop p99 < 100 ms |
| **P3 Baseline strategy + logging** | M | Tier 0, postures, utility rules, lead targeting, constraints, schema, engagements, dashboard | Beats Training Camp consistently; every decision logged with candidates, posture, predicted tile |
| **P4 Evaluation infra** | M | Practice gauntlet, swapped self-play, triage, replay diff, shadow, SPRT, ledger, version pinning, retention job | One SPRT run end-to-end overnight; triage tags on every engagement |
| **P5 Coach + tuning** | M | Reports, STRATEGY-only loss stories, Proposal flow + guardrails, Optuna | ≥ 1 coach proposal and ≥ 1 tuned candidate accepted by SPRT |
| **P6 Learned policy** | L | Video auto-labeling, BC → offline RL, ranker in the utility engine | Hybrid beats rules-only by SPRT |
| **P7 General Android** | M | Portal/uiautomator2, settle, SoM, OCR, macro cache, GUI-model fallback | Everyday-task suite; median step latency and cache-hit rate reported |

Don't start P5/P6 before P4 exists — without a trustworthy test, "improvements" are noise.

---

## 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Account ban | Throwaway accounts; Training Camp / Practice / own friendlies only |
| Game updates and balance patches | `game_version` + `balance_epoch` on every match; void tests across updates; dated card DB; golden-frame CI |
| scrcpy protocol changes | Pin the version; protocol unit test against a recorded handshake |
| False reconnects on static screens | Liveness by socket/process/ping, not frame gaps |
| Thermal throttling / burn-in | Display off during runs; cooling; RTT drift alerts |
| Perception errors steering strategy | Triage before the coach; perception queue; replay regression |
| Biased win-rate tests | Outcome-blind inclusion rule; strategy↔(phone, account) swapping; abort-rate comparison |
| GPU contention (LLM vs YOLO) | Advisor off-GPU; measure p99 with it on |
| Overfitting to one bot | Rotating gauntlet (15–85% band) + self-play |
| Too few matches | Funnel gates; two phones; dense secondary metrics; meaningful SPRT bounds |
| Disk growth | 7-day raw retention, nightly transcode, keyframe export |
| Coach churn | Path allowlist, replay-diff gate, one hypothesis per proposal, ledger dedupe |
| License (AGPL) | Private use is fine; swap to an Apache-2.0 detector before publishing |

---

## 17. Sources

- scrcpy: README (35–70 ms) https://github.com/Genymobile/scrcpy · v5.0 release https://github.com/Genymobile/scrcpy/releases/tag/v5.0 · frames only on change (`doc/develop.md`, `doc/video.md`) · display power / turn screen off (`doc/device.md`) · exact client/server version match (`doc/develop.md`)
- KataCR: https://arxiv.org/abs/2504.04783 · https://github.com/wty-yy/katacr · https://github.com/wty-yy/Clash-Royale-Detection-Dataset · https://github.com/wty-yy/Clash-Royale-Replay-Dataset
- YOLO26: https://arxiv.org/abs/2606.03748 · https://docs.ultralytics.com/models/yolo26/
- ONNX Runtime ROCm EP removal: https://github.com/microsoft/onnxruntime/pull/25181 · MIGraphX EP: https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html · AMD guide: https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/onnxruntime.html
- MaaFramework control methods: https://maafw.com/en/docs/2.4-ControlMethods
- DroidRun Portal: https://github.com/droidrun/droidrun-portal
- GUI-Owl-1.5: https://arxiv.org/abs/2602.16855 · MobileWorld: https://arxiv.org/abs/2512.19432
- Supercell: https://support.clashroyale.com/hc/en-us/articles/49484901806747-Don-t-Be-a-Cheat · https://supercell.com/en/safe-and-fair-play/ · Elixir https://support.clashroyale.com/hc/en-us/articles/49484936933403-Elixir · Training Camp bot https://supercell.com/en/games/clashroyale/blog/release-notes/app-store-update-october-23-2023/ · Practice Mode https://support.clashroyale.com/hc/nl/articles/49484905512219-Trainingskamp-en-oefenmodus · Deploy-delay change (Aug 2025) https://supercell.com/en/games/clashroyale/blog/release-notes/august-balance-changes-2
- Deploy timing of multi-unit cards: https://royaleapi.com/blog/season12
- Card data: https://developer.clashroyale.com · https://github.com/RoyaleAPI/cr-api-data
- SPRT: https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html · https://dannyhammer.github.io/engine-testing-guide/sprt.html
