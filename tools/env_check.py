"""Environment acceptance test (run it first on a new machine, e.g. the Jetson).

Checks
  1. environment: versions, GPU, resolved paths exist, pinned third-party commits
  2. detector: yolov5x COCO loads (80 classes), VNTraffic frame 1 gives vehicle detections
  3. speed: ms per forward at batch 1 and 4 (fixed 384x640 input), FP16 and/or FP32
  4. OC-SORT: tracks the first N frames of VNTraffic with real detections, IDs persist
  5. TrackEval self-check (GT vs GT = 100)

Writes <output_root>/env_check/report.json and det_frame1.jpg. Exit code 0 only if all checks pass.

    python tools/env_check.py                     # auto precision (FP16 on T4/Orin, FP32 on GTX 16xx)
    python tools/env_check.py --precision both    # benchmark both
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from emo_ocsort import paths  # noqa: E402
from emo_ocsort.detector import YoloV5Detector, auto_fp16  # noqa: E402
from emo_ocsort.ocsort_tracker import OCSort  # noqa: E402

RESULTS = {}
FAILS = []


def check(name, ok, detail=""):
    RESULTS.setdefault("checks", {})[name] = {"ok": bool(ok), "detail": detail}
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def git_head(path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def section_env():
    print("\n== 1. Environment")
    import scipy
    vers = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "opencv": cv2.__version__,
            "scipy": scipy.__version__}
    for mod in ("ultralytics", "filterpy", "lap"):
        try:
            vers[mod] = getattr(__import__(mod), "__version__", "ok")
        except ImportError:
            vers[mod] = None
    if torch.cuda.is_available():
        d = torch.device("cuda:0")
        vers.update(gpu=torch.cuda.get_device_name(d), gpu_capability=".".join(map(str, torch.cuda.get_device_capability(d))),
                    gpu_mem_gb=round(torch.cuda.get_device_properties(d).total_memory / 2**30, 1),
                    gpu_count=torch.cuda.device_count(), auto_fp16=auto_fp16(d))
    RESULTS["versions"] = vers
    RESULTS["paths"] = paths.summary()
    for k, v in vers.items():
        print(f"  {k:15s} {v}")
    for k, v in RESULTS["paths"].items():
        print(f"  {k:15s} {v}")
    check("cuda available", torch.cuda.is_available())
    for mod in ("ultralytics", "filterpy", "lap"):
        check(f"package {mod}", vers[mod] is not None)
    for key in ("data_root", "weights", "yolov5_dir", "trackeval_dir"):
        check(f"path {key} exists", Path(RESULTS["paths"][key]).exists(), RESULTS["paths"][key])
    check("sequences.json exists", paths.sequences_json().exists())
    yh = git_head(paths.yolov5_dir())
    check("yolov5 pinned commit", yh == paths.YOLOV5_COMMIT, f"HEAD={yh}")
    th = git_head(paths.trackeval_dir())
    check("TrackEval pinned commit", th is not None and th.startswith(paths.TRACKEVAL_COMMIT), f"HEAD={th}")


def read_frames(seq, n):
    cap = cv2.VideoCapture(str(paths.video_path(seq)))
    frames = []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames


def section_detector(det, frames):
    print("\n== 2. Detector")
    RESULTS["detector"] = {"weights": det.weights, "fp16": det.fp16, "input_hw": det.input_hw,
                           "classes": det.classes, "conf_thres": det.conf_thres}
    check("80 COCO classes", len(det.names) == 80, f"{len(det.names)} classes")
    check("vehicle class names", [det.names[i] for i in (2, 3, 5, 7)] == ["car", "motorcycle", "bus", "truck"])
    d = det([frames[0]])[0]
    per_cls = {det.names[int(c)]: int((d[:, 5] == c).sum()) for c in np.unique(d[:, 5])} if len(d) else {}
    strong = int((d[:, 4] >= 0.4).sum()) if len(d) else 0
    RESULTS["detector"].update(frame1_dets=len(d), frame1_dets_conf_ge_0_4=strong, frame1_per_class=per_cls)
    check("VNTraffic frame 1 detections", strong >= 20, f"{len(d)} dets (conf>=0.1), {strong} with conf>=0.4, {per_cls}")
    img = frames[0].copy()
    for x1, y1, x2, y2, s, c in d:
        if s >= 0.4:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{det.names[int(c)]} {s:.2f}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out = paths.output_root() / "env_check" / "det_frame1.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"  saved {out}")


def bench(det, frames, batch, iters):
    batch_frames = (frames * batch)[:batch]
    det.warmup(batch)
    t_inf, t_all = [], []
    for _ in range(iters):
        t = time.perf_counter()
        det(batch_frames)
        t_all.append(1e3 * (time.perf_counter() - t))
        t_inf.append(det.last_timing["infer_ms"])
    return {"batch": batch, "infer_ms": round(float(np.median(t_inf)), 1),
            "end2end_ms": round(float(np.median(t_all)), 1),
            "img_per_s_end2end": round(1e3 * batch / float(np.median(t_all)), 1)}


def section_speed(precisions, frames, iters):
    print("\n== 3. Speed (median over iterations; end2end = letterbox + forward + NMS + rescale)")
    RESULTS["speed"] = []
    for fp16 in precisions:
        det = YoloV5Detector(fp16=fp16)
        for b in (1, 4):
            r = bench(det, frames, b, iters)
            r["precision"] = "fp16" if fp16 else "fp32"
            RESULTS["speed"].append(r)
            print(f"  {r['precision']} batch {b}: forward {r['infer_ms']} ms, end2end {r['end2end_ms']} ms "
                  f"-> {r['img_per_s_end2end']} img/s")
        del det
        torch.cuda.empty_cache()
    RESULTS["speed_gpu_max_mem_mb"] = int(torch.cuda.max_memory_allocated() // 2**20) if torch.cuda.is_available() else None


def section_ocsort(det, frames):
    print(f"\n== 4. OC-SORT on the first {len(frames)} VNTraffic frames")
    trk = OCSort(det_thresh=0.4, max_age=30, min_hits=3, iou_threshold=0.3, delta_t=3, inertia=0.2)
    rows, n_out = [], []
    for i, f in enumerate(frames, 1):
        d = det([f])[0]
        dets = d[:, :5].astype(np.float64) if len(d) else np.empty((0, 5))
        h, w = f.shape[:2]
        out = trk.update(dets, (h, w), (h, w))  # same sizes -> OC-SORT applies no rescaling
        n_out.append(len(out))
        for x1, y1, x2, y2, tid in out:
            rows.append(f"{i},{int(tid)},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1")
    ids = {int(r.split(",")[1]) for r in rows}
    last_ids = {int(r.split(",")[1]) for r in rows if int(r.split(",")[0]) == len(frames)}
    first_ids = {int(r.split(",")[1]) for r in rows if int(r.split(",")[0]) == 10}
    kept = len(first_ids & last_ids) / max(len(first_ids), 1)
    out_file = paths.output_root() / "env_check" / "ocsort_first_frames.txt"
    out_file.write_text("\n".join(rows) + "\n", newline="\n")
    RESULTS["ocsort"] = {"frames": len(frames), "unique_ids": len(ids), "mean_tracks_per_frame": float(np.mean(n_out)),
                         "ids_at_frame10_still_alive_at_end": round(kept, 2), "file": str(out_file)}
    check("OC-SORT produces tracks", np.mean(n_out[5:]) >= 10, f"mean {np.mean(n_out):.1f} tracks/frame, {len(ids)} unique IDs")
    check("IDs persist", kept >= 0.5, f"{kept:.0%} of IDs at frame 10 still present at frame {len(frames)}")


def section_trackeval():
    print("\n== 5. TrackEval self-check")
    import io
    from contextlib import redirect_stdout
    import run_trackeval
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = run_trackeval.selfcheck(paths.output_root() / "env_check" / "trackeval_selfcheck")
    check("TrackEval GT vs GT = 100", ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["auto", "fp16", "fp32", "both"], default="auto")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--track_frames", type=int, default=60)
    a = ap.parse_args()

    section_env()
    if FAILS:
        print(f"\nStopping: environment checks failed: {FAILS}")
        raise SystemExit(1)

    frames = read_frames("VNTraffic", max(a.track_frames, 4))
    det = YoloV5Detector()  # auto precision
    section_detector(det, frames)
    section_ocsort(det, frames[:a.track_frames])
    del det
    torch.cuda.empty_cache()

    dev = torch.device("cuda:0")
    precisions = {"auto": [auto_fp16(dev)], "fp16": [True], "fp32": [False], "both": [True, False]}[a.precision]
    section_speed(precisions, frames[:4], a.iters)
    section_trackeval()

    RESULTS["passed"] = not FAILS
    RESULTS["failed_checks"] = FAILS
    rep = paths.output_root() / "env_check" / "report.json"
    rep.write_text(json.dumps(RESULTS, indent=2, default=str), newline="\n")
    print(f"\nreport: {rep}")
    print("ENV CHECK PASSED" if not FAILS else f"ENV CHECK FAILED: {FAILS}")
    raise SystemExit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
