from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_one(path: Path, baseline_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normaliza columnas esperadas
    if "baseline" not in df.columns:
        df["baseline"] = baseline_name
    return df


def main(reports_dir: Path) -> None:
    # Ajusta nombres según tus archivos finales
    files = [
        ("persistence", reports_dir / "persistence_h24_metrics_by_horizon.csv"),
        (
            "seasonal_naive_24h",
            reports_dir / "seasonal_naive_h24_metrics_by_horizon.csv",
        ),
        (
            "climatology_hourly_station",
            reports_dir / "climatology_hourly_station_h24_metrics_by_horizon.csv",
        ),
    ]

    dfs = []
    missing = []
    for name, fp in files:
        if fp.exists():
            dfs.append(load_one(fp, name))
        else:
            missing.append(str(fp))

    if missing:
        raise FileNotFoundError(
            "No se encontraron estos archivos:\n" + "\n".join(missing)
        )

    all_m = pd.concat(dfs, ignore_index=True)

    # --- tablas comparativas ---
    # 1) Resumen: promedio y h24
    summary = (
        all_m.groupby(["baseline", "split"])
        .agg(
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            n_eval_mean=("n_eval", "mean"),
        )
        .reset_index()
    )

    last = all_m[all_m["horizon_h"] == 24].rename(
        columns={"rmse": "rmse_h24", "r2": "r2_h24", "n_eval": "n_eval_h24"}
    )[["baseline", "split", "rmse_h24", "r2_h24", "n_eval_h24"]]

    summary = summary.merge(last, on=["baseline", "split"], how="left")

    out_sum = reports_dir / "baseline_comparison_summary.csv"
    summary.to_csv(out_sum, index=False)

    # 2) Tabla pivot por horizonte (RMSE)
    pivot_rmse = all_m.pivot_table(
        index=["split", "horizon_h"],
        columns="baseline",
        values="rmse",
        aggfunc="mean",
    ).reset_index()
    out_pivot = reports_dir / "baseline_comparison_rmse_by_horizon.csv"
    pivot_rmse.to_csv(out_pivot, index=False)

    # --- gráficas ---
    # RMSE vs horizonte por split
    for split in ["test", "val"]:
        part = all_m[all_m["split"] == split].sort_values(["baseline", "horizon_h"])
        plt.figure(figsize=(8, 4.5))
        for b in part["baseline"].unique():
            pb = part[part["baseline"] == b]
            plt.plot(pb["horizon_h"], pb["rmse"], label=b)
        plt.xlabel("Horizonte (h)")
        plt.ylabel("RMSE")
        plt.title(f"Comparación baselines: RMSE por horizonte ({split})")
        plt.legend()
        plt.tight_layout()
        out_fig = reports_dir / f"baseline_comparison_rmse_{split}.png"
        plt.savefig(out_fig, dpi=180)

    print("Guardado:")
    print("-", out_sum)
    print("-", out_pivot)
    print("-", reports_dir / "baseline_comparison_rmse_test.png")
    print("-", reports_dir / "baseline_comparison_rmse_val.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--reports-dir", default="reports/tables")
    args = p.parse_args()
    main(Path(args.reports_dir))
