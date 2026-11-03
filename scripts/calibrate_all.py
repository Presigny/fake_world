"""
calibrate_all.py
================
Run the SP mixture calibration (src/calibration.py) on every dataset of
synthesis_utils.DATASETS whose files are present, and write the scalar
results to out/00_dataset_calibration.csv (one row per dataset, same
columns as out/00_dataset_calibration_reference.csv).

Datasets whose border or point file is missing are skipped with a message.
The large datasets (ENEDIS, roads, Germany) take from tens of minutes to a
few hours each, mostly in the m-scan.

Usage:
    python scripts/calibrate_all.py
    python scripts/calibrate_all.py --only "Belgium Cities" "RTE France"
    python scripts/calibrate_all.py --n-scan 20 --n-jobs 4
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import synthesis_utils as su        # noqa: E402
import calibration as cal           # noqa: E402

OUT = ROOT / "out" / "00_dataset_calibration.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None,
                    help="names of the datasets to calibrate (keys of DATASETS)")
    ap.add_argument("--n-scan", type=int, default=40, help="splits of the m-scan")
    ap.add_argument("--n-real", type=int, default=5,
                    help="final realisations of the mixture")
    ap.add_argument("--n-jobs", type=int, default=None, help="parallel jobs of the m-scan")
    ap.add_argument("--seed", type=int, default=42, help="seed of the SP draw")
    args = ap.parse_args()

    names = args.only or list(su.DATASETS)
    rows = []
    if OUT.exists():                 # keep the rows of datasets not recomputed
        old = pd.read_csv(OUT)
        rows = [r for r in old.to_dict("records") if r["name"] not in names]

    for name in names:
        ds = su.DATASETS[name]
        missing = [p for p in (ds["path_border"], ds["path_csv"]) if not Path(p).exists()]
        if missing:
            print(f"[{name}] skipped, missing file(s): "
                  + ", ".join(str(Path(p).relative_to(ROOT)) for p in missing))
            continue
        t0 = time.time()
        res = cal.calibrate(name, n_scan=args.n_scan, n_real=args.n_real,
                            n_jobs=args.n_jobs, seed=args.seed)
        row = {k: res[k] for k in cal.SUMMARY_KEYS}
        row["area_km2"] = float(res["gdf_edge"].area.sum() / 1e6)
        rows.append(row)
        print(f"[{name}] done in {time.time() - t0:.0f} s\n")
        pd.DataFrame(rows).to_csv(OUT, index=False)     # save after each dataset

    if rows:
        df = pd.DataFrame(rows)
        cols = ["name", "N", "area_km2"] + [k for k in cal.SUMMARY_KEYS
                                            if k not in ("name", "N")]
        df = df[cols]
        df.to_csv(OUT, index=False)
        print(f"written: {OUT.relative_to(ROOT)}")
        print(df[["name", "N", "gamma", "eta", "m_star", "lamb", "L", "rho"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
