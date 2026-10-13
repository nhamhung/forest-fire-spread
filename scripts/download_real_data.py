"""Download NASA FIRMS (MODIS) fire detections and the NOAA Niño 3.4 index, then aggregate to monthly counts.

Raw yearly files (~600 MB) go to data/raw/ (not committed); the small monthly tables
written to data/processed/ are what the analysis, notebook and app use.
"""

import urllib.request

import pandas as pd

from forest_fire.real_data import FIRMS_URL, NINO34_URL, PROCESSED, RAW, monthly_fire_counts, parse_nino34

COUNTRIES = ["Indonesia", "Australia"]
YEARS = range(2001, 2026)          # full MODIS years available in the FIRMS country archive


def fetch(url: str, path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    urllib.request.urlretrieve(url, path)


def main():
    nino_path = RAW / "nina34.anom.data"
    fetch(NINO34_URL, nino_path)
    nino = parse_nino34(nino_path.read_text())

    frames = []
    for country in COUNTRIES:
        for year in YEARS:
            path = RAW / "firms" / f"modis_{year}_{country}.csv"
            fetch(FIRMS_URL.format(year=year, country=country), path)
            frames.append(monthly_fire_counts(path).assign(country=country))
    fires = pd.concat(frames, ignore_index=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    fires.to_csv(PROCESSED / "firms_monthly.csv", index=False)
    nino.to_csv(PROCESSED / "nino34_monthly.csv", index=False)
    print(f"{len(fires)} country-months, {fires['detections'].sum():,} detections; "
          f"Niño 3.4 {nino['year'].min()}-{nino['year'].max()} (latest {nino.iloc[-1].to_dict()})")


if __name__ == "__main__":
    main()
