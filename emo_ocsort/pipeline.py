"""Single-stream tracking loop: detection (live or cached) + OC-SORT, with an optional EMO skip policy."""
import time

import numpy as np

from .emo_policy import SkipPolicy
from .tracker import OCSortTracker
from .vis import VideoSink, draw_tracks


def run_sequence(reader, tracker_cfg, detector=None, cached_dets=None, policy_cfg=None,
                 vis_path=None, vis_width=960):
    """Run one video.

    policy_cfg: dict for SkipPolicy (name, k, sim_thresh, afd); None = detect every frame.
    Returns
      rows    : MOT rows "frame,id,x,y,w,h,1,-1,-1,-1"
      timing  : per-frame dicts (ms) incl. 'skipped'
      live    : {frame_id: (N, 6)} live detections of detected frames (cache material), or None if replayed
    """
    if (detector is None) == (cached_dets is None):
        raise ValueError("give exactly one of detector / cached_dets")
    tracker = OCSortTracker(**tracker_cfg)
    policy = SkipPolicy(fps=reader.fps, **(policy_cfg or {"name": "none"}))
    sink = VideoSink(vis_path, reader.fps, vis_width) if vis_path else None
    rows, timing, live = [], [], ({} if detector is not None else None)
    tracks = np.empty((0, 5))

    for fid, frame, decode_ms in reader:
        skip, info = policy.decide(frame, fid, tracker)
        rec = {"frame": fid, "decode_ms": decode_ms, "skipped": bool(skip), "sim": info["sim"],
               "decision_ms": info["decision_ms"]}
        if skip:
            t = time.perf_counter()
            if policy.name != "reuse":       # reuse keeps the previous boxes and does not touch the tracker
                tracks = tracker.skip()
            rec["est_ms"] = 1e3 * (time.perf_counter() - t)
        else:
            t0 = time.perf_counter()
            if detector is not None:
                d = detector([frame])[0]
                live[fid] = d
                det_t = detector.last_timing
            else:
                d = cached_dets.get(fid, np.empty((0, 6), np.float32))
                det_t = None
            t1 = time.perf_counter()
            tracks = tracker.update(d, frame.shape[:2])
            t2 = time.perf_counter()
            rec.update(track_ms=1e3 * (t2 - t1), n_dets=int(len(d)))
            if det_t is not None:
                rec.update(det_total_ms=1e3 * (t1 - t0), det_pre_ms=det_t["pre_ms"],
                           det_infer_ms=det_t["infer_ms"], det_post_ms=det_t["post_ms"])
        rec["decision_ms"] += policy.commit(frame, skip)
        rec["n_tracks"] = int(len(tracks))
        timing.append(rec)

        for x1, y1, x2, y2, tid in tracks:
            rows.append(f"{fid},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
        if sink:
            tag = "SKIP" if skip else "DET"
            sim = f" ncc={info['sim']:.2f}" if info["sim"] is not None else ""
            sink.write(draw_tracks(frame, tracks, f"frame {fid} {tag}{sim} tracks {len(tracks)}"))
    if sink:
        sink.close()
    return rows, timing, live


def _mean(timing, key, cond=lambda r: True):
    v = [r[key] for r in timing if key in r and r[key] is not None and cond(r)]
    return round(float(np.mean(v)), 3) if v else None


def summarize_timing(timing, live_detector):
    n = len(timing)
    n_skip = sum(r["skipped"] for r in timing)
    det = lambda r: not r["skipped"]  # noqa: E731
    s = {"frames": n, "detector": "live" if live_detector else "cache",
         "frames_detected": n - n_skip, "frames_skipped": n_skip, "skip_ratio": round(n_skip / max(n, 1), 4),
         "decode_ms": _mean(timing, "decode_ms"),
         "decision_ms": _mean(timing, "decision_ms"),                 # t_decision (every frame)
         "est_ms_on_skipped": _mean(timing, "est_ms", lambda r: r["skipped"]),  # t_estimation
         "track_ms_on_detected": _mean(timing, "track_ms", det),
         "mean_dets_per_detected_frame": _mean(timing, "n_dets", det),
         "mean_tracks_per_frame": _mean(timing, "n_tracks")}
    sims = [r["sim"] for r in timing if r["sim"] is not None]
    if sims:
        s["ncc_mean"], s["ncc_p10"], s["ncc_p90"] = (round(float(x), 3) for x in
                                                    (np.mean(sims), np.percentile(sims, 10), np.percentile(sims, 90)))
    if live_detector:
        s["det_total_ms_on_detected"] = _mean(timing, "det_total_ms", det)
        s["det_infer_ms_on_detected"] = _mean(timing, "det_infer_ms", det)
        # mean processing time per frame over ALL frames (what decides real-time capability)
        per_frame = [r["decision_ms"] + (r.get("est_ms", 0.0) if r["skipped"] else r["det_total_ms"] + r["track_ms"])
                     for r in timing]
        s["proc_ms_per_frame"] = round(float(np.mean(per_frame)), 2)
        s["fps_proc"] = round(1e3 / s["proc_ms_per_frame"], 1)
        s["fps_with_decode"] = round(1e3 / (s["proc_ms_per_frame"] + s["decode_ms"]), 1)
    return s
