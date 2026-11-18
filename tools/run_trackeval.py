"""Thin wrapper around TrackEval (MotChallenge2DBox) for the VEHICLE-test benchmark.

Expects tracker results at
    <trackers_folder>/VEHICLE-test/<tracker>/data/<seq>.txt     (MOT format, already filtered)

--selfcheck builds a tracker "GT_SELF" from the GT itself (through filter_preds) and evaluates it;
every metric must be perfect (HOTA = MOTA = IDF1 = 100, IDSW = 0). This validates the GT layout.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# TrackEval still references aliases removed in NumPy >= 1.24.
for _name, _typ in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _name):
        setattr(np, _name, _typ)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emo_ocsort import paths  # noqa: E402

sys.path.insert(0, str(paths.trackeval_dir()))
sys.path.insert(0, str(ROOT / "tools"))
import trackeval  # noqa: E402
from filter_preds import filter_file  # noqa: E402

GT_FOLDER = paths.gt_folder()


def evaluate(trackers_folder, trackers, seqs=None, output_folder=None):
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
                     "OUTPUT_DETAILED": True, "PLOT_CURVES": False})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(GT_FOLDER),
        "TRACKERS_FOLDER": str(trackers_folder),
        "OUTPUT_FOLDER": str(output_folder) if output_folder else None,
        "BENCHMARK": "VEHICLE",
        "SPLIT_TO_EVAL": "test",
        "TRACKERS_TO_EVAL": list(trackers),
        "CLASSES_TO_EVAL": ["pedestrian"],  # TrackEval's single MOT class id 1 (our GT class col = 1)
        "DO_PREPROC": False,                # MOT17 distractor removal does not apply to vehicles
        "PRINT_CONFIG": False,
    })
    if seqs:
        ds_cfg["SEQ_INFO"] = {s: None for s in seqs}
    evaluator = trackeval.Evaluator(eval_cfg)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, msg = evaluator.evaluate([trackeval.datasets.MotChallenge2DBox(ds_cfg)], metrics)
    return res, msg


def summarize(res, tracker):
    out = {}
    for seq, r in res["MotChallenge2DBox"][tracker].items():
        r = r["pedestrian"]
        out[seq] = {
            "HOTA": 100 * float(np.mean(r["HOTA"]["HOTA"])),
            "DetA": 100 * float(np.mean(r["HOTA"]["DetA"])),
            "AssA": 100 * float(np.mean(r["HOTA"]["AssA"])),
            "LocA": 100 * float(np.mean(r["HOTA"]["LocA"])),
            "MOTA": 100 * float(r["CLEAR"]["MOTA"]),
            "MOTP": 100 * float(r["CLEAR"]["MOTP"]),
            "IDF1": 100 * float(r["Identity"]["IDF1"]),
            "IDSW": int(r["CLEAR"]["IDSW"]),
            "FP": int(r["CLEAR"]["CLR_FP"]),
            "FN": int(r["CLEAR"]["CLR_FN"]),
        }
    return out


def selfcheck(work):
    tracker = "GT_SELF"
    work.mkdir(parents=True, exist_ok=True)
    seqs = [l.strip() for l in (GT_FOLDER / "seqmaps/VEHICLE-test.txt").read_text().splitlines()[1:] if l.strip()]
    for seq in seqs:
        gt = (GT_FOLDER / "VEHICLE-test" / seq / "gt/gt.txt").read_text().splitlines()
        raw = work / f"{seq}_raw.txt"
        # tracker format: frame,id,x,y,w,h,conf,-1,-1,-1
        raw.write_text("".join(",".join(l.split(",")[:6]) + ",1,-1,-1,-1\n" for l in gt if l.strip()), newline="\n")
        kept, removed = filter_file(raw, work / "trackers/VEHICLE-test" / tracker / "data" / f"{seq}.txt", seq)
        print(f"[selfcheck] {seq}: {kept} rows kept, {removed} removed by filter_preds")
    res, _ = evaluate(work / "trackers", [tracker], output_folder=work / "results")
    s = summarize(res, tracker)
    ok = True
    for seq, m in s.items():
        perfect = all(abs(m[k] - 100) < 1e-6 for k in ("HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1")) \
            and m["IDSW"] == 0 and m["FP"] == 0 and m["FN"] == 0
        ok &= perfect
        print(f"[selfcheck] {seq}: " + " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in m.items()) + (" -> OK" if perfect else " -> NOT PERFECT"))
    print("SELFCHECK OK" if ok else "SELFCHECK FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--work", default=str(paths.output_root() / "phase0_selfcheck"))
    ap.add_argument("--trackers_folder")
    ap.add_argument("--trackers", nargs="+")
    a = ap.parse_args()
    if a.selfcheck:
        raise SystemExit(0 if selfcheck(Path(a.work)) else 1)
    res, _ = evaluate(Path(a.trackers_folder), a.trackers)
    for t in a.trackers:
        for seq, m in summarize(res, t).items():
            print(t, seq, m)


if __name__ == "__main__":
    main()
