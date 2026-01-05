from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPORTS_DIR = Path("reports/figures")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Ajusta si tus rutas difieren
PERSIST = Path("reports/tables/persistence_h24_metrics_by_horizon.csv")
SEASONAL = Path("reports/tables/seasonal_naive_h24_metrics_by_horizon.csv")
CLIM = Path("reports/tables/climatology_hourly_station_h24_metrics_by_horizon.csv")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def load_all():
    pers = pd.read_csv(PERSIST)
    seas = pd.read_csv(SEASONAL)
    clim = pd.read_csv(CLIM)

    pers["baseline"] = "Persistencia"
    seas["baseline"] = "Naive estacional (24h)"
    clim["baseline"] = "Climatología (estación-hora)"
    return pd.concat([pers, seas, clim], ignore_index=True)

def make_plot(df, split: str):
    part = df[df["split"] == split].sort_values(["baseline", "horizon_h"])

    plt.figure(figsize=(8.5, 4.8))
    for b in ["Persistencia", "Naive estacional (24h)", "Climatología (estación-hora)"]:
        pb = part[part["baseline"] == b]
        plt.plot(pb["horizon_h"], pb["rmse"], marker="o", markersize=3, linewidth=1.5, label=b)

    plt.xlabel("Horizonte de pronóstico (horas)")
    plt.ylabel("RMSE de PM2.5 (µg/m³)")
    title_split = "prueba (test)" if split == "test" else "validación (val)"
    plt.title(f"Baselines: RMSE por horizonte – {title_split}")
    plt.grid(True, which="major", linewidth=0.6, alpha=0.35)
    plt.xlim(1, 24)
    plt.xticks(range(1, 25, 1))
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    out_png = REPORTS_DIR / f"baselines_rmse_por_horizonte_{split}.png"
    out_pdf = REPORTS_DIR / f"baselines_rmse_por_horizonte_{split}.pdf"
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

def main():
    df = load_all()
    make_plot(df, "test")
    make_plot(df, "val")
    print(f"Figuras guardadas en: {REPORTS_DIR.resolve()}")

if __name__ == "__main__":
    main()
