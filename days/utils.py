"""Reusable helpers for the Data Stories for Sustainability notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

try:  # Optional dependency; falls back to Matplotlib defaults if unavailable.
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - seaborn is optional in runtime envs
    sns = None  # type: ignore[assignment]


def _project_root() -> Path:
    """Return the repository root (the folder that contains the ``data`` dir)."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "data").exists():
            return candidate
    return cwd


def load_data(relative_path: str | Path, **read_csv_kwargs) -> pd.DataFrame:
    """Load a CSV file relative to the project root with consistent defaults."""
    root = _project_root()
    csv_path = (root / Path(relative_path)).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not locate data file: {csv_path}")
    defaults = {"na_values": ["", "NA", "NaN", "***"], "keep_default_na": True}
    defaults.update(read_csv_kwargs)
    return pd.read_csv(csv_path, **defaults)


def quick_diagnostics(
    df: pd.DataFrame,
    *,
    expected_columns: Sequence[str] | None = None,
    rows_between: tuple[int, int] | None = None,
    head_rows: int = 3,
) -> None:
    """Print a lightweight diagnostic block for formative self-checks."""
    print("👀 Data snapshot")
    print(f"Shape: {df.shape}")
    if expected_columns is not None:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            print(f"⚠️ Missing columns: {missing}")
        else:
            print(f"✅ Columns present: {expected_columns}")
    if rows_between is not None:
        low, high = rows_between
        n_rows = len(df)
        if low <= n_rows <= high:
            print(f"✅ Row count {n_rows} within expected range [{low}, {high}].")
        else:
            print(f"⚠️ Row count {n_rows} outside expected range [{low}, {high}].")
    print("Null values per column:")
    print(df.isna().sum())
    if head_rows > 0:
        print("Sample rows:")
        print(df.head(head_rows))


def baseline_style() -> None:
    """Apply a consistent visual baseline across notebooks."""
    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f9fb",
            "grid.color": "#d7dde5",
        }
    )


def check_story_metadata(**metadata: str) -> None:
    """Warn if any storytelling metadata strings are left blank."""
    empty = [key for key, value in metadata.items() if not value.strip()]
    if empty:
        print("⚠️ Please complete these storytelling fields:", ", ".join(empty))
    else:
        print("✅ Story metadata ready: " + ", ".join(metadata.keys()))


def add_story_footer(ax: plt.Axes, source: str, units: str) -> None:
    """Append a consistent footer with source and units information."""
    footer = f"{source} · {units}".strip()
    ax.text(
        0.5,
        -0.18,
        footer,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=9,
        color="#4a4a4a",
    )


def plots_directory() -> Path:
    """Return the shared plots directory, creating it if needed."""
    root = _project_root()
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir



def save_last_fig(filename: str) -> Path | None:
    """Persist the most recent Matplotlib figure to the shared plots directory."""
    fig = plt.gcf()
    if not fig.axes:
        print("⚠️ No Matplotlib figure detected; skipping save.")
        return None
    output_path = plots_directory() / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"💾 Saved figure to {output_path}")
    return output_path


def expect_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Secondary guardrail: raise a gentle warning if columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"⚠️ Unexpected column mismatch: {missing}")
    else:
        print("✅ Column names match the expectation.")


__all__ = [
    "baseline_style",
    "check_story_metadata",
    "expect_columns",
    "load_data",
    "quick_diagnostics",
    "plots_directory",
    "save_last_fig",
    "add_story_footer",
]
