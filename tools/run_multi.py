"""Phase 4: N copies of one video tracked on one GPU, EMO skipping per stream, batched detection.

    python tools/run_multi.py --exp RT_B1 --mode realtime
    python tools/run_multi.py --exp RT_B3k6 --mode realtime --policy periodic_est --k 6
    python tools/run_multi.py --exp RT_NCCk6 --mode realtime --policy ncc --k 6 --sim_thresh 0.75 --skip_oru False
    python tools/run_multi.py --exp OFF_B1 --mode offline

Output <output_root>/phase4/<exp>/: config.json, summary.json (speed, drops, batches, latency),
metrics.json (TrackEval per stream + mean/std), raw/s<i>.txt, trackers/VEHICLE-test/<exp>_s<i>/data/<seq>.txt
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from emo_ocsort import paths  # noqa: E402
from emo_ocsort.detector import COCO_VEHICLES, YoloV5Detector, auto_fp16  # noqa: E402
from emo_ocsort.emo_policy import POLICIES  # noqa: E402
from emo_ocsort.multistream import MultiStreamRunner, predecode  # noqa: E402
from emo_ocsort.tracker import DEFAULT_OCSORT  # noqa: E402

METRICS = ["HOTA", "DetA", "AssA", "LocA", "MOTA", "IDF1", "IDSW", "FP", "FN"]


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--phase", default="phase4")
    ap.add_argument("--seq", default="VNTraffic")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75], help="start delay (s) per stream")
    ap.add_argument("--mode", choices=["offline", "realtime"], default="realtime")
    ap.add_argument("--max_frames", type=int, default=None, help="debug: first N frames only (no eval)")
    ap.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    ap.add_argument("--policy", choices=POLICIES, default="none")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--sim_thresh", type=float, default=0.75)
    ap.add_argument("--afd", type=int, choices=[0, 1], default=1)
    ap.add_argument("--drops_count", type=int, choices=[0, 1], default=0,
                    help="1 = dropped frames count toward the k-consecutive-skips limit (first Phase 4 run)")
    ap.add_argument("--async_gpu", type=int, choices=[0, 1], default=1,
                    help="O4: detection in a worker thread, overlapped with CPU work (0 = synchronous ticks)")
    ap.add_argument("--cuda_graph", type=int, choices=[0, 1], default=1,
                    help="O7: replay the YOLOv5 forward as a CUDA graph (one launch, barely needs the GIL)")
    ap.add_argument("--switch_interval", type=float, default=0.0005,
                    help="O7: sys.setswitchinterval in seconds (Python default 0.005); 0 = leave the default")
    ap.add_argument("--closed_form", type=int, choices=[0, 1], default=1,
                    help="O3: one n-step Kalman prediction over n dropped frames (0 = n single steps)")
    for k, v in DEFAULT_OCSORT.items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    return ap.parse_args()


def main():
    a = parse()
    if len(a.offsets) < a.streams:
        raise SystemExit("--offsets needs one value per stream")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fp16 = auto_fp16(device) if a.precision == "auto" else a.precision == "fp16"
    trk_cfg = {k: getattr(a, k) for k in DEFAULT_OCSORT}
    pol_cfg = {"name": a.policy, "k": a.k, "sim_thresh": a.sim_thresh, "afd": bool(a.afd),
               "drops_count": bool(a.drops_count)}
    out = paths.output_root() / a.phase / a.exp
    out.mkdir(parents=True, exist_ok=True)

    if a.switch_interval > 0:
        sys.setswitchinterval(a.switch_interval)
    frames, decode_ms = predecode(paths.video_path(a.seq), a.max_frames)
    fps = json.loads(paths.sequences_json().read_text())["sequences"][a.seq]["fps"]
    det = YoloV5Detector(device=device, fp16=fp16, cuda_graph=bool(a.cuda_graph))
    for b in range(1, a.streams + 1):  # warm up every batch size that can occur
        det.warmup(b, 3)
    config = {"exp": a.exp, "seq": a.seq, "streams": a.streams, "offsets_s": a.offsets[:a.streams], "mode": a.mode,
              "fps": fps, "frames_per_stream": len(frames), "policy": pol_cfg, "tracker": trk_cfg, "fp16": fp16,
              "async_gpu": bool(a.async_gpu), "closed_form_advance": bool(a.closed_form),
              "cuda_graph": bool(a.cuda_graph), "switch_interval_s": sys.getswitchinterval(),
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "decode_ms_per_frame_measured_separately": round(decode_ms, 2), "paths": paths.summary(),
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(config, indent=2), newline="\n")
    print(f"[{a.exp}] {a.streams} x {a.seq} ({len(frames)} frames, {fps:g} FPS), mode={a.mode}, policy={pol_cfg}, "
          f"skip_mode={a.skip_mode}, skip_oru={a.skip_oru}, fp16={fp16}, async_gpu={bool(a.async_gpu)}, "
          f"closed_form={bool(a.closed_form)}, cuda_graph={bool(a.cuda_graph)}, "
          f"switch_interval={sys.getswitchinterval()}")

    runner = MultiStreamRunner(frames, fps, a.streams, a.offsets, det, trk_cfg, pol_cfg, a.mode,
                               async_gpu=bool(a.async_gpu), closed_form_advance=bool(a.closed_form)).run()
    summ = runner.summary()
    summ["decode_ms_per_frame"] = round(decode_ms, 2)
    (out / "summary.json").write_text(json.dumps(summ, indent=2), newline="\n")
    print(f"[{a.exp}] wall {summ['wall_s']} s | processed {summ['frames_processed']}/{summ['frames_total']} "
          f"(dropped {summ['drop_ratio']:.1%}) | skip {summ['skip_ratio_of_processed']:.1%} of processed | "
          f"throughput {summ['throughput_fps']} FPS (demand {summ['demand_fps']}) | batches {summ['batch_hist']}")
    print(f"[{a.exp}] ms per processed frame: {summ['ms_per_processed_frame']}")
    print(f"[{a.exp}] det ms per batch: {summ['det_ms_per_batch']} (forward {summ['det_infer_ms_per_batch']})")
    for s in summ["per_stream"]:
        print(f"[{a.exp}]   stream {s}")
    if a.max_frames is not None:
        print(f"[{a.exp}] --max_frames set: no evaluation")
        return

    from filter_preds import filter_file
    trackers = []
    for s in runner.streams:
        raw = out / "raw" / f"s{s.sid}.txt"
        raw.parent.mkdir(exist_ok=True)
        raw.write_text("\n".join(s.rows) + ("\n" if s.rows else ""), newline="\n")
        name = f"{a.exp}_s{s.sid}"
        filter_file(raw, out / "trackers" / "VEHICLE-test" / name / "data" / f"{a.seq}.txt", a.seq)
        trackers.append(name)
    import run_trackeval
    with open(out / "trackeval.log", "w") as log, contextlib.redirect_stdout(log):
        res, _ = run_trackeval.evaluate(out / "trackers", trackers, seqs=[a.seq], output_folder=out / "trackeval")
    per = {t: run_trackeval.summarize(res, t)[a.seq] for t in trackers}
    agg = {m: {"mean": round(float(np.mean([per[t][m] for t in trackers])), 3),
               "std": round(float(np.std([per[t][m] for t in trackers])), 3)} for m in METRICS}
    (out / "metrics.json").write_text(json.dumps({"per_stream": per, "mean_std": agg}, indent=2), newline="\n")
    print(f"[{a.exp}] " + " | ".join(f"{m} {agg[m]['mean']:.2f}±{agg[m]['std']:.2f}" for m in METRICS))


if __name__ == "__main__":
    main()
