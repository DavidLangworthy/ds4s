"""Shared utilities for the Data Stories for Sustainability notebooks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import requests

RAW_BASE = "https://raw.githubusercontent.com/DavidLangworthy/ds4s/master/"


def ensure_local_path(relative_path: str, *, base_url: str = RAW_BASE) -> Path:
    """Return a local path to a repository asset, downloading it if necessary."""
    path = Path(relative_path)
    if path.exists():
        return path

    url = base_url + relative_path.replace("\\", "/")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def load_csv(relative_path: str, *, read_csv_kwargs: Mapping | None = None, base_url: str = RAW_BASE) -> pd.DataFrame:
    """Load a CSV from the repo, downloading it from GitHub when absent locally."""
    read_csv_kwargs = dict(read_csv_kwargs or {})
    csv_path = ensure_local_path(relative_path, base_url=base_url)
    return pd.read_csv(csv_path, **read_csv_kwargs)


def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}")
    else:
        print(f"✅ Columns present: {list(required)}")


def expect_rows_between(df: pd.DataFrame, *, minimum: int, maximum: int) -> None:
    rows = len(df)
    if rows < minimum or rows > maximum:
        print(f"⚠️ Unexpected row count: {rows} (expected between {minimum} and {maximum}).")
    else:
        print(f"✅ Row count looks good: {rows}")


def quick_check(df: pd.DataFrame, *, name: str = "DataFrame", head: int = 3) -> pd.DataFrame:
    """Print a standard quick inspection and return the head for convenience."""
    print(f"{name}: shape={df.shape}")
    print(f"Columns: {list(df.columns)}")
    preview = df.head(head)
    print(preview)
    return preview


def baseline_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )


def validate_story_elements(elements: Mapping[str, str]) -> None:
    blanks = [key for key, value in elements.items() if not str(value).strip()]
    if blanks:
        print(f"⚠️ Missing story elements: {blanks}")
    else:
        print("✅ Story checklist complete.")


@dataclass
class FigureExport:
    figure: plt.Figure
    path: Path


def save_last_fig(fig: plt.Figure | None, relative_path: str) -> FigureExport | None:
    """Save a Matplotlib figure if it exists and return metadata for logging."""
    if fig is None:
        print("⚠️ No figure supplied to save.")
        return None

    output_path = Path(relative_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    return FigureExport(fig, output_path)


def save_plotly_fig(fig: Any, relative_path: str) -> Path | None:
    """Persist a Plotly figure to HTML for archiving."""
    if fig is None:
        print("⚠️ No Plotly figure supplied to save.")
        return None

    output_path = Path(relative_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"Saved Plotly figure to {output_path}")
    return output_path


__all__ = [
    "FigureExport",
    "baseline_style",
    "ensure_local_path",
    "expect_rows_between",
    "load_csv",
    "quick_check",
    "save_last_fig",
    "save_plotly_fig",
    "validate_columns",
    "validate_story_elements",
]
