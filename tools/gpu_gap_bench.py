"""Does the GPU forward get slower when the GPU idles between calls? (Phase 3 finding on T4:
14.4 ms back-to-back vs ~26 ms when detection alternates with CPU work.)

Measures the median forward time (batch 1 and 4) with a gap between calls in which the GPU is idle:
  - CPU busy-wait (like EMO's NCC / OC-SORT work)
  - sleep
and samples the graphics clock (nvidia-smi) during each setting.

    python tools/gpu_gap_bench.py
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from emo_ocsort import paths  # noqa: E402
from emo_ocsort.detector import YoloV5Detector  # noqa: E402


class ClockSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.on = [], True

    def run(self):
        while self.on:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.gr,pstate,utilization.gpu",
                                    "--format=csv,noheader,nounits", "-i", "0"], capture_output=True, text=True, timeout=5)
                c, p, u = (x.strip() for x in r.stdout.strip().split(","))
                self.samples.append((int(c), p, int(u)))
            except Exception:
                pass
            time.sleep(0.05)


def bench(det, frames, gap_ms, mode, n=40):
    ts = []
    sampler = ClockSampler()
    sampler.start()
    for _ in range(n):
        if gap_ms:
            if mode == "busy":
                t = time.perf_counter()
                while (time.perf_counter() - t) * 1e3 < gap_ms:
                    pass
            else:
                time.sleep(gap_ms / 1e3)
        det(frames)
        ts.append(det.last_timing["infer_ms"])
    sampler.on = False
    sampler.join()
    clocks = [s[0] for s in sampler.samples]
    return {"forward_ms_median": round(float(np.median(ts)), 2), "forward_ms_p90": round(float(np.percentile(ts, 90)), 2),
            "clock_mhz_median": int(np.median(clocks)) if clocks else None,
            "clock_mhz_min": int(np.min(clocks)) if clocks else None}


def main():
    det = YoloV5Detector()
    cap = cv2.VideoCapture(str(paths.video_path("VNTraffic")))
    frame = cap.read()[1]
    out = {"device_fp16": det.fp16, "results": []}
    for batch in (1, 4):
        frames = [frame] * batch
        det.warmup(batch, 10)
        for gap, mode in [(0, "none"), (5, "busy"), (10, "busy"), (20, "busy"), (40, "busy"), (20, "sleep"), (0, "none")]:
            r = {"batch": batch, "gap_ms": gap, "gap_mode": mode, **bench(det, frames, gap, mode)}
            out["results"].append(r)
            print(r, flush=True)
    p = paths.output_root() / "phase4" / "gpu_gap_bench.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), newline="\n")
    print(p)


if __name__ == "__main__":
    main()
