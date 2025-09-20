"""Shared helpers for the Data Stories for Sustainability course notebooks."""
from __future__ import annotations

import textwrap
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent / "data"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
BASE_DATA_URL = "https://raw.githubusercontent.com/DavidLangworthy/ds4s/master/data"


def _quote_filename(filename: str) -> str:
    return requests.utils.requote_uri(filename)


def load_data(
    filename: str,
    *,
    data_dir: Path | None = None,
    auto_download: bool = True,
    silent: bool = False,
    **read_kwargs,
) -> pd.DataFrame:
    """Load a CSV file, downloading it from GitHub if it is missing locally."""
    data_dir = Path(data_dir or DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / filename
    if auto_download and not file_path.exists():
        url = f"{BASE_DATA_URL}/{_quote_filename(filename)}"
        response = requests.get(url, timeout=30)
        if response.ok:
            file_path.write_bytes(response.content)
            if not silent:
                print(f"⬇️  Downloaded {filename} from {url}")
        else:
            warnings.warn(
                f"Could not download {filename} from {url} (status {response.status_code})."
            )

    read_kwargs = {"encoding": "utf-8-sig", **read_kwargs}
    df = pd.read_csv(file_path, **read_kwargs)
    if not silent:
        print(f"📦 Loaded {filename} with shape {df.shape}")
    return df


def baseline_style(context: str = "notebook") -> None:
    """Apply consistent styling for Matplotlib and Seaborn plots."""
    sns.set_theme(context=context, style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 110,
            "axes.titleweight": "bold",
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def diagnostics(
    df: pd.DataFrame,
    name: str,
    *,
    expected_columns: Sequence[str] | None = None,
    expected_row_range: tuple[int, int] | None = None,
    head: int = 3,
) -> None:
    """Print quick diagnostics for a dataframe."""
    print(f"📝 {name}")
    print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    if expected_columns is not None:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            warnings.warn(f"Missing columns: {missing}")
        else:
            print(f"   ✅ Expected columns present: {', '.join(expected_columns)}")
    if expected_row_range is not None:
        low, high = expected_row_range
        if not (low <= len(df) <= high):
            warnings.warn(
                f"Row count {len(df)} outside expected range ({low}–{high})."
            )
        else:
            print(
                f"   ✅ Row count within expected range ({low}–{high} rows)."
            )
    print("   Null counts:")
    print(df.isna().sum().head(10))
    try:
        from IPython.display import display

        display(df.head(head))
    except Exception:
        print(df.head(head))



def validate_columns(df: pd.DataFrame, required: Sequence[str]) -> list[str]:
    missing = [col for col in required if col not in df.columns]
    if missing:
        warnings.warn(f"Missing columns: {missing}")
    else:
        print(f"✅ Columns look good: {', '.join(required)}")
    return missing


def expect_rows_between(df: pd.DataFrame, low: int, high: int) -> bool:
    within = low <= len(df) <= high
    if within:
        print(f"✅ Row count between {low} and {high} (actual: {len(df)})")
    else:
        warnings.warn(f"Row count {len(df)} outside {low}–{high}")
    return within


STORY_FIELDS = ("title", "subtitle", "annotation", "source", "units")


def validate_story_elements(metadata: Mapping[str, str]) -> None:
    missing = [key for key in STORY_FIELDS if not metadata.get(key, "").strip()]
    if missing:
        warnings.warn(
            "Story metadata missing values for: " + ", ".join(missing)
        )
    else:
        print("✅ Story metadata complete.")


def apply_story_template(
    ax: plt.Axes,
    *,
    title: str,
    subtitle: str,
    source: str,
    units: str,
) -> None:
    ax.set_title(title, loc="left", pad=14)
    ax.text(
        0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=12,
        ha="left",
        va="bottom",
        color="#4f4f4f",
    )
    ax.set_ylabel(units)
    ax.text(
        0,
        -0.18,
        f"Source: {source}",
        transform=ax.transAxes,
        fontsize=10,
        color="#666666",
    )


def save_last_fig(filename: str, *, directory: Path | None = None, fig=None) -> Path:
    directory = Path(directory or PLOTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    output_path = directory / filename
    if hasattr(fig, "savefig"):
        fig.savefig(output_path, bbox_inches="tight")
        print(f"💾 Saved figure to {output_path}")
    elif hasattr(fig, "write_html"):
        fig.write_html(str(output_path.with_suffix(".html")))
        print(f"💾 Saved interactive figure to {output_path.with_suffix('.html')}")
    else:
        warnings.warn("Figure object does not support saving.")
    return output_path


def summarize_claim(claim: str, evidence: str, takeaway: str) -> str:
    return textwrap.dedent(
        f"""
        **Claim:** {claim}
        **Evidence:** {evidence}
        **Takeaway:** {takeaway}
        """
    ).strip()
