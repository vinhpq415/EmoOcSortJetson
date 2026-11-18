"""Per-frame detection cache.

A frame's detections do not depend on the tracker or on the skipping policy, so accuracy sweeps
(Phase 2 det_thresh, Phase 3 policies / max_age variants) can replay cached detections instead of
re-running YOLOv5x. Speed is measured only in runs that execute the detector.
"""
import json
from pathlib import Path

import numpy as np

from . import paths


def cache_tag(det_cfg):
    h, w = det_cfg["input_hw"]
    cls = "-".join(map(str, det_cfg["classes"])) if det_cfg["classes"] is not None else "all"
    prec = "fp16" if det_cfg["fp16"] else "fp32"
    agn = "agn" if det_cfg["agnostic_nms"] else "cls"
    return f"yolov5x_{h}x{w}_{prec}_conf{det_cfg['conf_thres']}_iou{det_cfg['iou_thres']}_{agn}_c{cls}"


def cache_dir(tag):
    return paths.output_root() / "dets_cache" / tag


def save(tag, seq, dets_by_frame, det_cfg):
    d = cache_dir(tag)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{seq}.npz", **{str(f): a.astype(np.float32) for f, a in dets_by_frame.items()})
    (d / "detector.json").write_text(json.dumps(det_cfg, indent=2), newline="\n")
    return d / f"{seq}.npz"


def load(tag, seq):
    p = cache_dir(tag) / f"{seq}.npz"
    if not p.exists():
        raise FileNotFoundError(f"no detection cache {p}; run once without --use_cache to create it")
    z = np.load(p)
    return {int(k): z[k] for k in z.files}
