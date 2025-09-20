"""Shared helper utilities for the DS4S notebooks."""
from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import requests

RAW_BASE_URL = "https://raw.githubusercontent.com/DavidLangworthy/ds4s/main/"


def project_root() -> Path:
    """Return the repository root path."""
    return Path(__file__).resolve().parent


def data_directory() -> Path:
    """Return the local data directory."""
    return project_root() / "data"


def _resolve_data_target(path_like: str | Path) -> tuple[Path | None, str]:
    """Resolve a dataset path locally and remotely."""
    path = Path(path_like)
    if not path.is_absolute():
        candidate = data_directory() / path
    else:
        candidate = path
    if candidate.exists():
        return candidate, candidate.name
    # Fall back to GitHub raw URL
    parts = [p for p in path.parts if p not in ("..", ".")]
    relative = "/".join(["data", *parts])
    url = f"{RAW_BASE_URL}{relative}"
    return None, url


def load_data(path_like: str | Path, *, reader: str = "csv", **kwargs) -> pd.DataFrame:
    """Load a dataset from disk, or fetch it from GitHub if missing locally."""
    local_path, remote_target = _resolve_data_target(path_like)

    if reader != "csv":
        raise ValueError("Currently only CSV datasets are supported.")

    read_csv_kwargs = {"encoding": "utf-8", **kwargs}

    if local_path is not None:
        return pd.read_csv(local_path, **read_csv_kwargs)

    response = requests.get(remote_target, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text), **read_csv_kwargs)


def validate_columns(df: pd.DataFrame, required: Iterable[str], *, name: str = "DataFrame") -> None:
    """Warn if expected columns are missing."""
    required_set = set(required)
    missing = required_set.difference(df.columns)
    if missing:
        warnings.warn(
            f"{name} is missing expected columns: {sorted(missing)}",
            stacklevel=2,
        )


def expect_rows_between(
    df: pd.DataFrame,
    low: int,
    high: int,
    *,
    name: str = "DataFrame",
    units: str = "rows",
) -> None:
    """Warn if the row count falls outside an expected range."""
    count = len(df)
    if not (low <= count <= high):
        warnings.warn(
            f"{name} has {count} {units}; expected between {low} and {high}.",
            stacklevel=2,
        )


def diagnose_dataframe(
    df: pd.DataFrame,
    *,
    name: str = "DataFrame",
    sample: int = 5,
) -> None:
    """Print quick diagnostics for a dataframe."""
    print(f"🔍 {name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values per column:")
    print(df.isna().sum())
    print(df.head(sample))


def baseline_style() -> None:
    """Apply consistent styling for plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.titleweight": "semibold",
            "axes.grid": True,
            "legend.frameon": False,
        }
    )


def validate_story_elements(metadata: Mapping[str, str]) -> None:
    """Warn if any storytelling metadata fields are blank."""
    empty = [key for key, value in metadata.items() if not str(value).strip()]
    if empty:
        warnings.warn(
            f"Please fill in these storytelling fields: {', '.join(empty)}",
            stacklevel=2,
        )


def save_last_fig(
    filename: str,
    *,
    directory: Path | None = None,
    fig: object | None = None,
) -> Path:
    """Save a Matplotlib or Plotly figure to the shared plots directory."""
    if directory is None:
        directory = project_root() / "plots"
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / filename

    if fig is None:
        fig = plt.gcf()

    if hasattr(fig, "savefig"):
        fig.savefig(output_path, bbox_inches="tight")
    elif hasattr(fig, "write_image"):
        try:
            fig.write_image(str(output_path))
        except Exception as exc:  # noqa: BLE001 - provide a graceful fallback for plotly
            warnings.warn(
                f"Fell back to HTML export for {filename} because static image export failed: {exc}",
                stacklevel=2,
            )
            html_path = output_path.with_suffix(".html")
            fig.write_html(str(html_path))
            return html_path
    else:
        raise TypeError("Unsupported figure type for saving.")

    return output_path


__all__ = [
    "baseline_style",
    "data_directory",
    "diagnose_dataframe",
    "expect_rows_between",
    "load_data",
    "project_root",
    "save_last_fig",
    "validate_columns",
    "validate_story_elements",
]
