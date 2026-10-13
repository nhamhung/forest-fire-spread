"""Phase 5: real fire activity vs El Niño, from public NASA and NOAA data.

Data sources (both free, no account needed):

* NASA FIRMS active-fire detections from the MODIS instruments on the Terra
  and Aqua satellites, as yearly per-country files
  (https://firms.modaps.eosdis.nasa.gov/data/country/). Each row is one
  1 km pixel seen burning at one overpass. Counts of detections are a standard
  proxy for fire activity: more detections = more and bigger fires. They are
  not burned area.
* NOAA PSL Niño 3.4 index: monthly sea-surface temperature anomaly (°C) in the
  central equatorial Pacific (https://psl.noaa.gov/data/correlation/nina34.anom.data).
  Sustained values above +0.5 °C mean El Niño, below -0.5 °C La Niña.

El Niño shifts rainfall away from the Maritime Continent and Australia, so the
dry season there is drier and longer. The model predicts that drier fuel can
push a landscape past its critical point, so fire activity should rise faster
than linearly with drought.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/data/country/modis/{year}/modis_{year}_{country}.csv"
NINO34_URL = "https://psl.noaa.gov/data/correlation/nina34.anom.data"

# fire-season months used for each region (the dry season, when nearly all burning happens)
SEASONS = {
    "Indonesia": {"months": [8, 9, 10, 11], "label": "Aug-Nov"},
    "Australia": {"months": [9, 10, 11, 12], "label": "Sep-Dec"},
}
EL_NINO, LA_NINA = 0.5, -0.5   # °C thresholds on the seasonal Niño 3.4 anomaly


def parse_nino34(text: str) -> pd.DataFrame:
    """Parse NOAA PSL's text format (one row per year, 12 monthly values, -99.99 = missing)."""
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue                                   # header / footer lines
        year = int(parts[0])
        for month, value in enumerate(parts[1:], start=1):
            v = float(value)
            if v > -99:
                rows.append({"year": year, "month": month, "nino34": v})
    return pd.DataFrame(rows)


def monthly_fire_counts(csv_path: Path, min_confidence: int = 30) -> pd.DataFrame:
    """Count vegetation-fire detections per month in one FIRMS yearly file.

    Keeps detections with confidence >= `min_confidence` (MODIS: 0-100; below 30
    is "low confidence") and type 0 ("presumed vegetation fire"). Types 1-3 are
    volcanoes, static industrial heat sources and offshore gas flares.
    """
    df = pd.read_csv(csv_path, usecols=["acq_date", "confidence", "type"])
    df = df[(df["confidence"] >= min_confidence) & (df["type"] == 0)]
    dates = pd.to_datetime(df["acq_date"])
    counts = dates.groupby([dates.dt.year, dates.dt.month]).size()
    counts.index.names = ["year", "month"]
    return counts.rename("detections").reset_index()


def seasonal_table(fires: pd.DataFrame, nino: pd.DataFrame, country: str) -> pd.DataFrame:
    """One row per year: fire-season detections and the mean Niño 3.4 anomaly over the same months."""
    months = SEASONS[country]["months"]
    f = fires[(fires["country"] == country) & fires["month"].isin(months)].groupby("year")["detections"].sum()
    n = nino[nino["month"].isin(months)].groupby("year")["nino34"].mean()
    table = pd.concat([f, n], axis=1, join="inner").reset_index()
    table["phase"] = np.select([table["nino34"] >= EL_NINO, table["nino34"] <= LA_NINA],
                               ["El Niño", "La Niña"], "neutral")
    table["country"] = country
    return table


def phase_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Median and mean fire-season detections per ENSO phase, and the ratio to neutral years."""
    g = table.groupby("phase")["detections"]
    out = pd.DataFrame({"years": g.count(), "median": g.median(), "mean": g.mean()})
    out["mean_vs_neutral"] = out["mean"] / out.loc["neutral", "mean"] if "neutral" in out.index else np.nan
    return out.reindex([p for p in ["La Niña", "neutral", "El Niño"] if p in out.index])


def permutation_test(table: pd.DataFrame, n_perm: int = 20000, seed: int = 0) -> float:
    """One-sided p-value that El Niño years burn more than the rest, by shuffling the phase labels.

    Makes no assumption about the distribution of fire counts, which is very skewed
    (a few extreme years dominate).
    """
    rng = np.random.default_rng(seed)
    is_nino = (table["phase"] == "El Niño").to_numpy()
    logs = np.log(table["detections"].to_numpy())
    observed = logs[is_nino].mean() - logs[~is_nino].mean()
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(is_nino)
        count += logs[perm].mean() - logs[~perm].mean() >= observed
    return (count + 1) / (n_perm + 1)
