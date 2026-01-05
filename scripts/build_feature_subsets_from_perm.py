from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS_DEFAULT = ["station_id", "timestamp", "split"]


def ensure_perm_cum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas:
    - importance_pos (clip lower 0)
    - importance_share (normalizada sobre positivos)
    - importance_cum (cumsum)
    """
    df = df.copy()
    if "importance_pos" not in df.columns:
        df["importance_pos"] = df["importance_mean"].clip(lower=0.0)
    if "importance_share" not in df.columns:
        tot = df["importance_pos"].sum()
        df["importance_share"] = df["importance_pos"] / tot if tot > 0 else 0.0
    if "importance_cum" not in df.columns:
        df["importance_cum"] = df["importance_share"].cumsum()
    return df


def select_features_by_cum(df_perm: pd.DataFrame, threshold: float) -> list[str]:
    df_perm = df_perm.sort_values("importance_mean", ascending=False).reset_index(
        drop=True
    )
    df_perm = ensure_perm_cum(df_perm)

    cum = df_perm["importance_cum"].to_numpy()
    idx = int(np.argmax(cum >= threshold))  # primer índice que cumple
    if cum[idx] < threshold:
        raise ValueError(
            f"No se alcanzó el umbral {threshold:.2f}. Revisa tu archivo de importancias."
        )
    k = idx + 1
    return df_perm.loc[: k - 1, "feature"].tolist()


def build_dataset(
    features_path: Path,
    perm_path: Path,
    out_path: Path,
    out_list_path: Path,
    threshold: float,
    keep_meta: list[str],
    keep_all_targets: bool,
    horizon: int,
) -> None:
    df = pd.read_parquet(features_path)

    perm = pd.read_csv(perm_path)
    selected = select_features_by_cum(perm, threshold)

    # Targets
    y_cols = [f"y_h{h:02d}" for h in range(1, horizon + 1)]
    targets_present = [c for c in y_cols if c in df.columns]

    cols = []
    # Meta
    cols += [c for c in keep_meta if c in df.columns]
    # Targets
    if keep_all_targets:
        cols += targets_present
    # Selected features
    cols += selected

    # Dedup preservando orden
    seen = set()
    cols_final = []
    missing_selected = []
    for c in cols:
        if c not in df.columns:
            # Si es una feature seleccionada y falta, lo reportamos
            if c in selected:
                missing_selected.append(c)
            continue
        if c not in seen:
            cols_final.append(c)
            seen.add(c)

    if missing_selected:
        raise KeyError(
            "Estas variables fueron seleccionadas por Permutation Importance pero no existen en el parquet:\n"
            + "\n".join(missing_selected)
        )

    df_out = df[cols_final].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)

    # Guarda lista de features seleccionadas (solo X, no meta/targets)
    out_list_path.parent.mkdir(parents=True, exist_ok=True)
    out_list_path.write_text("\n".join(selected))

    print("=" * 80)
    print(f"Threshold: {threshold:.2f}")
    print(f"Selected features (k): {len(selected)}")
    print(f"Saved dataset: {out_path} | shape={df_out.shape}")
    print(f"Saved feature list: {out_list_path}")
    print("Selected features:")
    for f in selected:
        print(" -", f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features-parquet", default="data/processed/features_h24_v1.parquet"
    )
    p.add_argument("--perm-csv", default="reports/tables/rf_perm_importance_y_h24.csv")
    p.add_argument("--reports-dir", default="reports/tables")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--keep-all-targets", action="store_true", default=True)
    p.add_argument("--meta-cols", default="station_id,timestamp,split")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    features_path = Path(args.features_parquet)
    perm_path = Path(args.perm_csv)
    reports_dir = Path(args.reports_dir)
    out_dir = Path(args.out_dir)

    keep_meta = [c.strip() for c in args.meta_cols.split(",") if c.strip()]

    # 90%
    build_dataset(
        features_path=features_path,
        perm_path=perm_path,
        out_path=out_dir / "features_h24_perm90_yh24.parquet",
        out_list_path=reports_dir / "rf_perm_selected_features_y_h24_90.txt",
        threshold=0.90,
        keep_meta=keep_meta,
        keep_all_targets=args.keep_all_targets,
        horizon=args.horizon,
    )

    # 95% (sensibilidad)
    build_dataset(
        features_path=features_path,
        perm_path=perm_path,
        out_path=out_dir / "features_h24_perm95_yh24.parquet",
        out_list_path=reports_dir / "rf_perm_selected_features_y_h24_95.txt",
        threshold=0.95,
        keep_meta=keep_meta,
        keep_all_targets=args.keep_all_targets,
        horizon=args.horizon,
    )
