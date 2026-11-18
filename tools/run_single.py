"""Single-stream experiment runner: detection (live or cached) + OC-SORT, then TrackEval.

Phase 2 (baseline B1, no skipping):
    python tools/run_single.py --exp B1_default --save_video           # live detector, fills the cache
    python tools/run_single.py --exp B1_dt0.3 --det_thresh 0.3 --use_cache

Phase 3 (EMO skipping; --skip_mode V0/V1/V2 = how OC-SORT treats skipped frames):
    python tools/run_single.py --phase phase3 --exp B3_V1 --policy periodic_est --k 1 --skip_mode V1 --use_cache
    python tools/run_single.py --phase phase3 --exp E1_V1 --policy ncc --k 4 --sim_thresh 0.75 --afd 1 --use_cache

Output: <output_root>/<phase>/<exp>/
    config.json, metrics.json, timing.json, raw/<seq>.txt, trackers/VEHICLE-test/<exp>/data/<seq>.txt,
    videos/<seq>.mp4 (with --save_video), trackeval.log
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import torch  # noqa: E402

from emo_ocsort import dets_cache, paths  # noqa: E402
from emo_ocsort.detector import COCO_VEHICLES, auto_fp16  # noqa: E402
from emo_ocsort.emo_policy import POLICIES  # noqa: E402
from emo_ocsort.pipeline import run_sequence, summarize_timing  # noqa: E402
from emo_ocsort.tracker import DEFAULT_OCSORT  # noqa: E402
from emo_ocsort.video import VideoReader  # noqa: E402

SEQS = ["VNTraffic", "AICC22-Custom"]
METRIC_COLS = ["HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDSW", "FP", "FN"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name (folder + TrackEval tracker name)")
    ap.add_argument("--phase", default="phase2")
    ap.add_argument("--seqs", nargs="+", default=SEQS)
    ap.add_argument("--max_frames", type=int, default=None, help="debug: stop after N frames (no eval)")
    # detector
    ap.add_argument("--use_cache", action="store_true", help="replay cached detections instead of running YOLOv5x")
    ap.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    ap.add_argument("--input_hw", type=int, nargs=2, default=[384, 640])
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--classes", type=int, nargs="+", default=list(COCO_VEHICLES))
    # EMO skip policy
    ap.add_argument("--policy", choices=POLICIES, default="none")
    ap.add_argument("--k", type=int, default=4, help="max consecutive skipped frames (repo: detect_frame_interval)")
    ap.add_argument("--sim_thresh", type=float, default=0.75, help="NCC threshold (paper: 0.75)")
    ap.add_argument("--afd", type=int, choices=[0, 1], default=1, help="adaptive forced detection (k/4 if FPS < 15)")
    # tracker
    for k, v in DEFAULT_OCSORT.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    # outputs
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--vis_width", type=int, default=960)
    return ap.parse_args()


def main():
    a = parse()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fp16 = auto_fp16(device) if a.precision == "auto" else a.precision == "fp16"
    det_cfg = {"input_hw": a.input_hw, "fp16": fp16, "conf_thres": a.conf, "iou_thres": a.iou,
               "classes": a.classes, "agnostic_nms": True}
    tag = dets_cache.cache_tag(det_cfg)
    trk_cfg = {k: getattr(a, k) for k in DEFAULT_OCSORT}
    policy_cfg = {"name": a.policy, "k": a.k, "sim_thresh": a.sim_thresh, "afd": bool(a.afd)}

    out = paths.output_root() / a.phase / a.exp
    out.mkdir(parents=True, exist_ok=True)
    config = {"exp": a.exp, "seqs": a.seqs, "detector": det_cfg, "dets_cache_tag": tag, "use_cache": a.use_cache,
              "tracker": trk_cfg, "policy": policy_cfg, "max_frames": a.max_frames, "paths": paths.summary(),
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(config, indent=2), newline="\n")
    print(f"[{a.exp}] detector={'cache:' + tag if a.use_cache else 'live (' + tag + ')'}")
    print(f"[{a.exp}] tracker={trk_cfg}")
    print(f"[{a.exp}] policy={policy_cfg}")

    detector = None
    if not a.use_cache:
        from emo_ocsort.detector import YoloV5Detector
        detector = YoloV5Detector(device=device, fp16=fp16, input_hw=a.input_hw, conf_thres=a.conf,
                                  iou_thres=a.iou, classes=a.classes, agnostic_nms=True)
        detector.warmup(1)

    from filter_preds import filter_file
    timing_all = {}
    for seq in a.seqs:
        reader = VideoReader(paths.video_path(seq), a.max_frames)
        vis = None
        if a.save_video:
            (out / "videos").mkdir(exist_ok=True)
            vis = out / "videos" / f"{seq}.mp4"
        cached = dets_cache.load(tag, seq) if a.use_cache else None
        rows, timing, live = run_sequence(reader, trk_cfg, detector=detector, cached_dets=cached,
                                          policy_cfg=policy_cfg, vis_path=vis, vis_width=a.vis_width)
        if live is not None and a.max_frames is None and a.policy == "none":  # cache needs every frame
            p = dets_cache.save(tag, seq, live, det_cfg)
            print(f"[{a.exp}] {seq}: detection cache -> {p}")
        raw = out / "raw" / f"{seq}.txt"
        raw.parent.mkdir(exist_ok=True)
        raw.write_text("\n".join(rows) + ("\n" if rows else ""), newline="\n")
        kept, removed = filter_file(raw, out / "trackers" / "VEHICLE-test" / a.exp / "data" / f"{seq}.txt", seq)
        timing_all[seq] = summarize_timing(timing, detector is not None)
        timing_all[seq]["rows_kept"], timing_all[seq]["rows_removed_outside_gt_range"] = kept, removed
        print(f"[{a.exp}] {seq}: {json.dumps(timing_all[seq])}")
    (out / "timing.json").write_text(json.dumps(timing_all, indent=2), newline="\n")

    if a.max_frames is not None:
        print(f"[{a.exp}] --max_frames set: skipping evaluation (partial sequences)")
        return

    import run_trackeval
    with open(out / "trackeval.log", "w") as log, contextlib.redirect_stdout(log):
        res, _ = run_trackeval.evaluate(out / "trackers", [a.exp], seqs=a.seqs, output_folder=out / "trackeval")
    metrics = run_trackeval.summarize(res, a.exp)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), newline="\n")

    print(f"\n[{a.exp}] " + " | ".join(f"{c:>6s}" for c in ["seq".ljust(14)] + METRIC_COLS))
    for seq, m in metrics.items():
        vals = [f"{m[c]:6.2f}" if isinstance(m[c], float) else f"{m[c]:6d}" for c in METRIC_COLS]
        print(f"[{a.exp}] {seq[:14].ljust(14)} | " + " | ".join(vals))
    print(f"[{a.exp}] results in {out}")


if __name__ == "__main__":
    main()
