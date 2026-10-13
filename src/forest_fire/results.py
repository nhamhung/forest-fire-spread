"""Save every result together with the settings that produced it."""

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def save_result(name: str, settings: dict, table: pd.DataFrame | None = None, summary: dict | None = None,
                figure=None, results_dir: Path = RESULTS_DIR) -> Path:
    """Write results/<name>/ with settings.json and, when given, data.csv and figure.png.

    settings.json holds the inputs (including the random seed), summary numbers
    (e.g. the fitted critical density) and library versions, so every result can
    be traced back to and regenerated from its inputs.
    """
    out = Path(results_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "settings": settings,
        "summary": summary or {},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / "settings.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    if table is not None:
        table.to_csv(out / "data.csv", index=False)
    if figure is not None:
        figure.savefig(out / "figure.png", dpi=150, bbox_inches="tight")
    return out
