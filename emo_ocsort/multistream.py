"""Multi-stream EMO tracking on one GPU (Phase 4).

N streams (Phase 4: copies of VNTraffic with staggered starts) share one YOLOv5x. Each stream has its own
OC-SORT tracker and EMO skip policy. A central loop works in "ticks":

  1. pick the frame to process for every stream that has a new one
  2. CPU: EMO decision per stream (skip or detect)
  3. GPU: ONE batched forward for all streams that need detection (batch 0..N)
  4. CPU: OC-SORT update (detected) / Kalman estimation (skipped) per stream

Modes
  offline   : every frame of every stream is processed (no drops); measures maximum aggregate throughput.
              Stream s starts at tick offset_frames[s].
  realtime  : frame i of stream s "arrives" at t0 + offset_s + i / fps (wall clock). Each tick takes the
              NEWEST arrived frame per stream; frames that arrived in between are DROPPED: never processed,
              no output (they count as missed in the evaluation), the tracker is moved over them
              (OCSortTracker.advance) and they count toward the policy's k-consecutive limit.

Frames are decoded once before the run and shared by all streams (same video), so decode time is
measured separately and not part of the tick (on Jetson decoding would go to the hardware decoder).
"""
import time

import numpy as np

from .emo_policy import SkipPolicy
from .tracker import OCSortTracker
from .video import VideoReader


def predecode(path, max_frames=None):
    frames, ms = [], []
    for _, f, dt in VideoReader(path, max_frames):
        frames.append(f)
        ms.append(dt)
    return frames, float(np.mean(ms))


class Stream:
    def __init__(self, sid, frames, fps, offset_s, tracker_cfg, policy_cfg):
        self.sid, self.frames, self.fps, self.offset_s = sid, frames, fps, offset_s
        self.n = len(frames)
        self.tracker = OCSortTracker(**tracker_cfg)
        self.policy = SkipPolicy(fps=fps, **(policy_cfg or {"name": "none"}))
        self.last = -1            # index of the last processed frame (0-based)
        self.rows = []            # MOT rows (frame ids 1-based)
        self.log = []             # per processed frame
        self.dropped = 0

    @property
    def done(self):
        return self.last >= self.n - 1

    def emit(self, idx, tracks):
        fid = idx + 1
        for x1, y1, x2, y2, tid in tracks:
            self.rows.append(f"{fid},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")


class MultiStreamRunner:
    def __init__(self, frames, fps, n_streams, offsets_s, detector, tracker_cfg, policy_cfg, mode="realtime",
                 async_gpu=False, closed_form_advance=True):
        if mode not in ("offline", "realtime"):
            raise ValueError("mode must be offline or realtime")
        self.mode, self.detector, self.fps = mode, detector, fps
        self.async_gpu, self.closed_form = async_gpu, closed_form_advance
        self.streams = [Stream(i, frames, fps, offsets_s[i], tracker_cfg, policy_cfg) for i in range(n_streams)]
        self.ticks = []

    # ------------------------------------------------------------------ one tick
    def process(self, items, now_fn):
        """items: list of (stream, idx, n_dropped_before). Returns the tick record."""
        t0 = time.perf_counter()
        rec = {"n_streams": len(items), "dropped": 0, "decision_ms": 0.0, "advance_ms": 0.0,
               "est_ms": 0.0, "det_ms": 0.0, "det_infer_ms": 0.0, "track_ms": 0.0, "batch": 0}
        to_detect, to_skip = [], []
        for s, idx, gap in items:
            if gap:
                ta = time.perf_counter()
                s.tracker.advance(gap, self.closed_form)
                s.policy.note_dropped(gap)
                s.dropped += gap
                rec["dropped"] += gap
                rec["advance_ms"] += 1e3 * (time.perf_counter() - ta)
            frame = s.frames[idx]
            skip, info = s.policy.decide(frame, idx + 1, s.tracker)
            rec["decision_ms"] += info["decision_ms"]
            (to_skip if skip else to_detect).append((s, idx, info))

        for s, idx, info in to_skip:
            te = time.perf_counter()
            tracks = s.tracker.skip()
            rec["est_ms"] += 1e3 * (time.perf_counter() - te)
            self._finish(s, idx, tracks, True, info, rec, now_fn)

        if to_detect:
            td = time.perf_counter()
            dets = self.detector([s.frames[idx] for s, idx, _ in to_detect])
            rec["det_ms"] = 1e3 * (time.perf_counter() - td)
            rec["det_infer_ms"] = self.detector.last_timing["infer_ms"]
            rec["batch"] = len(to_detect)
            for (s, idx, info), d in zip(to_detect, dets):
                tt = time.perf_counter()
                tracks = s.tracker.update(d, s.frames[idx].shape[:2])
                rec["track_ms"] += 1e3 * (time.perf_counter() - tt)
                self._finish(s, idx, tracks, False, info, rec, now_fn)
        rec["tick_ms"] = 1e3 * (time.perf_counter() - t0)
        return rec

    def _finish(self, s, idx, tracks, skipped, info, rec, now_fn):
        rec["decision_ms"] += s.policy.commit(s.frames[idx], skipped)
        s.last = idx
        s.emit(idx, tracks)
        arrival = s.offset_s + idx / s.fps
        s.log.append({"frame": idx + 1, "skipped": skipped, "sim": info["sim"],
                      "latency_ms": 1e3 * (now_fn() - arrival) if self.mode == "realtime" else None})

    # ------------------------------------------------------------------ asynchronous loop (O4)
    def _next_item(self, s, t, start_frame):
        """(idx, gap) of the frame stream s should process now, or None.
        offline: stream s starts once stream 0 has processed start_frame frames (staggered start), then runs
        free; realtime: newest arrived frame, the ones in between are dropped."""
        if s.done:
            return None
        if self.mode == "offline":
            started = s.last >= 0 or self.streams[0].last + 1 >= start_frame
            return (s.last + 1, 0) if started else None
        arrived = min(int(np.floor((t - s.offset_s) * s.fps)), s.n - 1)
        return (arrived, arrived - s.last - 1) if arrived > s.last else None

    def run_async(self, now):
        """GPU/CPU overlap (Phase 4 O4). A worker thread runs batched detections; this thread alone touches the
        trackers and policies. A stream waiting for its detection is busy: frames arriving meanwhile are dropped
        (real-time) or wait (offline). While the GPU works on one batch, this thread handles skipped frames,
        tracker updates of the previous batch and the decisions that form the next batch."""
        from concurrent.futures import ThreadPoolExecutor, wait

        pool = ThreadPoolExecutor(max_workers=1)
        pending, inflight, busy = [], None, set()
        start = {s.sid: round(s.offset_s * s.fps) for s in self.streams}  # offline staggered start (frames)

        def det_job(frames):
            t = time.perf_counter()
            d = self.detector(frames)
            return d, 1e3 * (time.perf_counter() - t), self.detector.last_timing["infer_ms"]

        while True:
            worked = False
            # 1. finished detection batch -> tracker updates (CPU)
            if inflight is not None and inflight[0].done():
                dets, det_ms, infer_ms = inflight[0].result()
                rec = {"n_streams": len(inflight[1]), "dropped": 0, "decision_ms": 0.0, "advance_ms": 0.0,
                       "est_ms": 0.0, "det_ms": det_ms, "det_infer_ms": infer_ms, "track_ms": 0.0,
                       "batch": len(inflight[1])}
                t0 = time.perf_counter()
                for (s, idx, info), d in zip(inflight[1], dets):
                    tt = time.perf_counter()
                    tracks = s.tracker.update(d, s.frames[idx].shape[:2])
                    rec["track_ms"] += 1e3 * (time.perf_counter() - tt)
                    self._finish(s, idx, tracks, False, info, rec, now)
                    busy.discard(s.sid)
                rec["tick_ms"] = 1e3 * (time.perf_counter() - t0)
                self.ticks.append(rec)
                inflight, worked = None, True
            # 2. free streams with a new frame -> decision; skipped frames are finished right away
            t = now()
            rec = {"n_streams": 0, "dropped": 0, "decision_ms": 0.0, "advance_ms": 0.0, "est_ms": 0.0,
                   "det_ms": 0.0, "det_infer_ms": 0.0, "track_ms": 0.0, "batch": 0}
            t0 = time.perf_counter()
            for s in self.streams:
                if s.sid in busy:
                    continue
                nxt = self._next_item(s, t, start[s.sid])
                if nxt is None:
                    continue
                idx, gap = nxt
                rec["n_streams"] += 1
                if gap:
                    ta = time.perf_counter()
                    s.tracker.advance(gap, self.closed_form)
                    s.policy.note_dropped(gap)
                    s.dropped += gap
                    rec["dropped"] += gap
                    rec["advance_ms"] += 1e3 * (time.perf_counter() - ta)
                skip, info = s.policy.decide(s.frames[idx], idx + 1, s.tracker)
                rec["decision_ms"] += info["decision_ms"]
                if skip:
                    te = time.perf_counter()
                    tracks = s.tracker.skip()
                    rec["est_ms"] += 1e3 * (time.perf_counter() - te)
                    self._finish(s, idx, tracks, True, info, rec, now)
                else:
                    pending.append((s, idx, info))
                    busy.add(s.sid)
            if rec["n_streams"]:
                rec["tick_ms"] = 1e3 * (time.perf_counter() - t0)
                self.ticks.append(rec)
                worked = True
            # 3. GPU idle -> submit everything that waits for detection as one batch
            if inflight is None and pending:
                inflight = (pool.submit(det_job, [s.frames[idx] for s, idx, _ in pending]), pending)
                pending, worked = [], True
            # 4. finished?
            if all(s.done for s in self.streams) and inflight is None and not pending:
                break
            # 5. nothing to do: wait for the GPU or the next frame arrival
            if not worked:
                timeout = 0.002
                if self.mode == "realtime":
                    nxt = [s.offset_s + (s.last + 1) / s.fps for s in self.streams if not s.done and s.sid not in busy]
                    if nxt:
                        timeout = max(0.0, min(nxt) - now())
                if inflight is not None:
                    wait([inflight[0]], timeout=timeout if self.mode == "realtime" else None)
                elif self.mode == "realtime":
                    time.sleep(timeout)
        pool.shutdown()

    # ------------------------------------------------------------------ main loops
    def run(self):
        self.t0 = time.perf_counter()
        now = lambda: time.perf_counter() - self.t0  # noqa: E731
        if self.async_gpu:
            self.run_async(now)
        elif self.mode == "offline":
            tick = 0
            start = [round(s.offset_s * s.fps) for s in self.streams]
            while not all(s.done for s in self.streams):
                items = [(s, s.last + 1, 0) for s, st in zip(self.streams, start) if tick >= st and not s.done]
                if items:
                    self.ticks.append(self.process(items, now))
                tick += 1
        else:
            while not all(s.done for s in self.streams):
                t = now()
                items = []
                for s in self.streams:
                    if s.done:
                        continue
                    arrived = min(int(np.floor((t - s.offset_s) * s.fps)), s.n - 1)
                    if arrived > s.last:
                        items.append((s, arrived, arrived - s.last - 1))
                if items:
                    self.ticks.append(self.process(items, now))
                else:  # nothing new: sleep until the next arrival
                    nxt = min(s.offset_s + (s.last + 1) / s.fps for s in self.streams if not s.done)
                    time.sleep(max(0.0, nxt - now()))
        self.wall_s = time.perf_counter() - self.t0
        return self

    # ------------------------------------------------------------------ report
    def summary(self):
        T = self.ticks
        n_proc = sum(len(s.log) for s in self.streams)
        n_total = sum(s.n for s in self.streams)
        n_skip = sum(r["skipped"] for s in self.streams for r in s.log)
        # async: CPU events have batch 0 and are not detection ticks; keep only real batches in the histogram
        batches = [r["batch"] for r in T if not (self.async_gpu and r["batch"] == 0)]
        sm = lambda k: round(float(np.sum([r[k] for r in T])) / max(n_proc, 1), 3)  # noqa: E731  ms per processed frame
        out = {
            "mode": self.mode, "async_gpu": self.async_gpu, "closed_form_advance": self.closed_form,
            # async: "tick" = main-thread CPU time only (the detection runs in the worker thread)
            "streams": len(self.streams), "wall_s": round(self.wall_s, 2), "ticks": len(T),
            "frames_total": n_total, "frames_processed": n_proc, "frames_dropped": n_total - n_proc,
            "drop_ratio": round((n_total - n_proc) / n_total, 4),
            "skip_ratio_of_processed": round(n_skip / max(n_proc, 1), 4),
            "detect_ratio_of_all": round((n_proc - n_skip) / n_total, 4),
            "throughput_fps": round(n_proc / self.wall_s, 1),
            "demand_fps": round(sum(s.fps for s in self.streams), 1),
            "batch_hist": {str(b): batches.count(b) for b in sorted(set(batches))},
            "ms_per_processed_frame": {k.replace("_ms", ""): sm(k) for k in
                                       ("decision_ms", "advance_ms", "est_ms", "det_ms", "track_ms", "tick_ms")},
            "det_ms_per_batch": {str(b): round(float(np.mean([r["det_ms"] for r in T if r["batch"] == b])), 2)
                                 for b in sorted(set(batches)) if b > 0},
            "det_infer_ms_per_batch": {str(b): round(float(np.mean([r["det_infer_ms"] for r in T if r["batch"] == b])), 2)
                                       for b in sorted(set(batches)) if b > 0},
            "per_stream": [],
        }
        for s in self.streams:
            lat = [r["latency_ms"] for r in s.log if r["latency_ms"] is not None]
            out["per_stream"].append({
                "sid": s.sid, "offset_s": s.offset_s, "processed": len(s.log), "dropped": s.dropped,
                "skipped": sum(r["skipped"] for r in s.log),
                "latency_ms_mean": round(float(np.mean(lat)), 1) if lat else None,
                "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None})
        return out
