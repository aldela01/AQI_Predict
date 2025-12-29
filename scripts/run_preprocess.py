# scripts/run_preprocess.py

from pathlib import Path
import argparse

# Add parent directory to sys.path for module imports
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.make_master import MasterBuildConfig, build_master


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-master", type=str, default="data/raw/siata_merged_data.csv")
    p.add_argument("--coords", type=str, default="data/external/stations_coords.csv")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--coords-station-col", type=str, default="station_id")
    p.add_argument("--coords-lat-col", type=str, default="lat")
    p.add_argument("--coords-lon-col", type=str, default="lon")
    args = p.parse_args()

    cfg = MasterBuildConfig(
        raw_master_csv=Path(args.raw_master),
        coords_csv=Path(args.coords),
        k_neighbors=args.k,
        coords_station_col=args.coords_station_col,
        coords_lat_col=args.coords_lat_col,
        coords_lon_col=args.coords_lon_col,
    )
    build_master(cfg)


if __name__ == "__main__":
    main()
