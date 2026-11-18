"""Table of all Phase 4 runs: speed, drops, skipping, latency and accuracy (mean ± std over streams).

    python tools/summarize_multi.py        # writes <output_root>/phase4/summary.md and summary.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emo_ocsort import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="phase4")
    a = ap.parse_args()
    d = paths.output_root() / a.phase
    rows = []
    for e in sorted(p for p in d.iterdir() if (p / "metrics.json").exists()):
        cfg = json.loads((e / "config.json").read_text())
        s = json.loads((e / "summary.json").read_text())
        m = json.loads((e / "metrics.json").read_text())["mean_std"]
        pol = cfg["policy"]
        ms = s["ms_per_processed_frame"]
        lat = [x["latency_ms_mean"] for x in s["per_stream"] if x["latency_ms_mean"] is not None]
        p95 = [x["latency_ms_p95"] for x in s["per_stream"] if x["latency_ms_p95"] is not None]
        rows.append({
            "exp": e.name, "mode": s["mode"], "async": int(s.get("async_gpu", False)),
            "closed_form": int(s.get("closed_form_advance", False)),
            "drops_count": int(pol.get("drops_count", True)),  # run 1 had no flag and counted drops
            "graph": int(cfg.get("cuda_graph", False)),
            "swi_ms": round(1e3 * cfg.get("switch_interval_s", 0.005), 2),
            "streams": s["streams"], "policy": pol["name"],
            "k": pol["k"] if pol["name"] != "none" else "", "sim": pol["sim_thresh"] if pol["name"] == "ncc" else "",
            "oru": "exact" if cfg["tracker"].get("skip_oru", True) else "fast",
            "drop%": round(100 * s["drop_ratio"], 1), "skip%_of_processed": round(100 * s["skip_ratio_of_processed"], 1),
            "detect%_of_all": round(100 * s["detect_ratio_of_all"], 1),
            "throughput_fps": s["throughput_fps"], "demand_fps": s["demand_fps"],
            "ms_decision": ms["decision"], "ms_advance": ms["advance"], "ms_est": ms["est"], "ms_det": ms["det"],
            "ms_track": ms["track"], "ms_tick": ms["tick"],
            "det_ms_b1": s["det_ms_per_batch"].get("1"), "det_ms_b4": s["det_ms_per_batch"].get("4"),
            "latency_ms": round(sum(lat) / len(lat), 1) if lat else "", "latency_p95_ms": round(max(p95), 1) if p95 else "",
            "HOTA": f"{m['HOTA']['mean']:.2f}±{m['HOTA']['std']:.2f}", "DetA": round(m["DetA"]["mean"], 2),
            "AssA": round(m["AssA"]["mean"], 2), "MOTA": round(m["MOTA"]["mean"], 2), "IDF1": round(m["IDF1"]["mean"], 2),
            "IDSW": round(m["IDSW"]["mean"], 1), "FN": round(m["FN"]["mean"], 0),
        })
    if not rows:
        raise SystemExit(f"no finished runs under {d}")
    cols = list(rows[0])
    with open(d / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (d / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    sys.stdout.buffer.write(("\n".join(md) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
