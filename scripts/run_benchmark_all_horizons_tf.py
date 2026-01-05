
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def parse_horizons(spec: str) -> List[int]:
    """
    Accepts:
      - "1-24"
      - "1,2,3,24"
      - "24"
    """
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        return list(range(lo, hi + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Run TF benchmark for multiple horizons (one run per horizon).")
    p.add_argument("--horizons", default="1-24", help='Horizons to run, e.g. "1-24" or "1,6,12,24"')
    p.add_argument("--dataset", default="data/processed/features_h24_perm90_yh24.parquet")
    p.add_argument("--experiment", default="pm25_h24")
    p.add_argument("--run-name-base", default="benchmark_proposed_models_tf")
    p.add_argument("--reports-dir", default="reports/tables")

    # shared training knobs
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seq-batch-size", type=int, default=256)
    p.add_argument("--lookback", type=int, default=168)
    p.add_argument("--seq-max-rows", type=int, default=250000)

    # TF runtime knobs
    p.add_argument("--tf-gpu-growth", action="store_true")
    p.add_argument("--tf-mixed-precision", action="store_true")
    p.add_argument("--tf-xla", action="store_true")
    p.add_argument("--tf-intra-threads", type=int, default=0)
    p.add_argument("--tf-inter-threads", type=int, default=0)

    # SARIMA knobs
    p.add_argument("--sarima-order", default="1,0,1")
    p.add_argument("--sarima-seasonal", default="1,0,1,24")
    p.add_argument("--sarima-n-jobs", type=int, default=-1)

    p.add_argument("--ridge-alpha", type=float, default=1.0)

    # Random Forest
    p.add_argument("--rf-n-estimators", type=int, default=300)
    p.add_argument("--rf-max-depth", type=int, default=0)
    p.add_argument("--rf-min-samples-leaf", type=int, default=1)
    p.add_argument("--rf-max-features", default="sqrt")
    p.add_argument("--rf-n-jobs", type=int, default=-1)

    p.add_argument("--continue-on-error", action="store_true")

    args = p.parse_args()

    horizons = parse_horizons(args.horizons)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    script = Path("scripts/bench_proposed_models_yh24_tf.py")
    if not script.exists():
        raise FileNotFoundError(f"Expected script not found: {script}. Run from repo root.")

    all_rows = []

    for h in horizons:
        run_name = f"{args.run_name_base}_h{h:02d}"
        cmd = [
            sys.executable, str(script),
            "--dataset", args.dataset,
            "--experiment", args.experiment,
            "--run-name", run_name,
            "--reports-dir", args.reports_dir,
            "--target-h", str(h),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seq-batch-size", str(args.seq_batch_size),
            "--lookback", str(args.lookback),
            "--seq-max-rows", str(args.seq_max_rows),
            "--sarima-order", args.sarima_order,
            "--sarima-seasonal", args.sarima_seasonal,
            "--sarima-n-jobs", str(args.sarima_n_jobs),
            "--ridge-alpha", str(args.ridge_alpha),
            "--rf-n-estimators", str(args.rf_n_estimators),
            "--rf-max-depth", str(args.rf_max_depth),
            "--rf-min-samples-leaf", str(args.rf_min_samples_leaf),
            "--rf-max-features", str(args.rf_max_features),
            "--rf-n-jobs", str(args.rf_n_jobs),
            "--tf-intra-threads", str(args.tf_intra_threads),
            "--tf-inter-threads", str(args.tf_inter_threads),
        ]
        if args.tf_gpu_growth:
            cmd.append("--tf-gpu-growth")
        if args.tf_mixed_precision:
            cmd.append("--tf-mixed-precision")
        if args.tf_xla:
            cmd.append("--tf-xla")

        print(f"\n=== Horizon h={h:02d} | run_name={run_name} ===")
        rc = subprocess.call(cmd)
        if rc != 0:
            msg = f"Horizon h={h:02d} failed with return code {rc}"
            print(msg, file=sys.stderr)
            if not args.continue_on_error:
                raise RuntimeError(msg)
            continue

        # Collect per-horizon summary CSV
        y_col = f"y_h{h:02d}"
        out_csv = reports_dir / f"benchmark_proposed_models_{y_col}_tf.csv"
        if out_csv.exists():
            dfh = pd.read_csv(out_csv)
            dfh.insert(0, "horizon", h)
            all_rows.append(dfh)
        else:
            print(f"Warning: expected output not found: {out_csv}", file=sys.stderr)

    if all_rows:
        out_all = pd.concat(all_rows, ignore_index=True)
        out_all_path = reports_dir / "benchmark_proposed_models_all_horizons_tf.csv"
        out_all.to_csv(out_all_path, index=False)
        print(f"\nSaved aggregated table: {out_all_path}")
    else:
        print("\nNo results aggregated. Check earlier errors.", file=sys.stderr)


if __name__ == "__main__":
    main()
