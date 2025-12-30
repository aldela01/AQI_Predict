# scripts/run_preprocess_day2.py
from pathlib import Path
import argparse

# Add parent directory to sys.path for module imports
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.make_features_h24 import (
    FeaturesH24Config,
    build_features,
    _configure_logging,
)

_configure_logging()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--master-long", default="data/interim/master_long_v1.parquet")
    p.add_argument("--neighbors", default="data/interim/neighbors_by_station_k5.json")
    p.add_argument("--out", default="data/processed/features_h24_v1.parquet")
    args = p.parse_args()

    cfg = FeaturesH24Config(
        master_long_parquet=Path(args.master_long),
        neighbors_json=Path(args.neighbors),
        out_processed_parquet=Path(args.out),
    )
    build_features(cfg)


if __name__ == "__main__":
    main()
