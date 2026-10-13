"""Phase 5 data handling, checked offline on small hand-made inputs."""

import pandas as pd
import pytest

from forest_fire import real_data as rd

NINO_TEXT = """  2014        2015
 2014    -0.50   -0.40    0.10    0.20    0.30    0.40    0.50    0.60    0.70    0.80    0.90    1.00
 2015     1.00    1.10    1.20    1.30    1.40    1.50    1.60    1.70    1.80    1.90    2.00  -99.99
  -99.99
  Nino 3.4 SST anomaly
"""


def test_parse_nino34_skips_headers_and_missing():
    df = rd.parse_nino34(NINO_TEXT)
    assert len(df) == 23 and df["nino34"].max() == 2.0
    assert df.iloc[0].to_dict() == {"year": 2014, "month": 1, "nino34": -0.5}


def test_fire_counts_keep_confident_vegetation_fires(tmp_path):
    csv = tmp_path / "f.csv"
    pd.DataFrame({"acq_date": ["2015-09-01", "2015-09-02", "2015-09-03", "2015-10-01", "2015-10-02"],
                  "confidence": [80, 10, 90, 50, 60], "type": [0, 0, 2, 0, 0]}).to_csv(csv, index=False)
    out = rd.monthly_fire_counts(csv)
    assert out.set_index("month")["detections"].to_dict() == {9: 1, 10: 2}   # low confidence and type 2 dropped


def test_seasonal_table_and_phases():
    fires = pd.DataFrame({"country": "Indonesia", "year": [2014] * 4 + [2015] * 4, "month": [8, 9, 10, 11] * 2,
                          "detections": [1, 2, 3, 4, 10, 20, 30, 40]})
    t = rd.seasonal_table(fires, rd.parse_nino34(NINO_TEXT), "Indonesia").set_index("year")
    assert t.loc[2014, "detections"] == 10 and t.loc[2015, "detections"] == 100
    assert t.loc[2014, "nino34"] == pytest.approx(0.75) and t.loc[2014, "phase"] == "El Niño"


def test_permutation_test_detects_a_clear_difference():
    t = pd.DataFrame({"phase": ["El Niño"] * 6 + ["neutral"] * 6, "detections": [100] * 6 + [10] * 6})
    assert rd.permutation_test(t, n_perm=2000) < 0.01
    same = pd.DataFrame({"phase": ["El Niño", "neutral"] * 6, "detections": [10] * 12})
    assert rd.permutation_test(same, n_perm=2000) > 0.5


@pytest.mark.skipif(not (rd.PROCESSED / "firms_monthly.csv").exists(), reason="run scripts/download_real_data.py")
def test_processed_data_shows_the_el_nino_signal_in_indonesia():
    fires, nino = pd.read_csv(rd.PROCESSED / "firms_monthly.csv"), pd.read_csv(rd.PROCESSED / "nino34_monthly.csv")
    cmp = rd.phase_comparison(rd.seasonal_table(fires, nino, "Indonesia"))
    assert cmp.loc["El Niño", "mean_vs_neutral"] > 2
