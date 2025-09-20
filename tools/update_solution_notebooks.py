import inspect

import nbformat as nbf
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[1]

UTILITIES_CODE = "\n".join(
    [
        "from __future__ import annotations",
        "",
        "from pathlib import Path",
        "from typing import Mapping, Sequence",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "",
        "try:",
        "    import plotly.express as px  # noqa: F401 - imported for student use",
        "except ModuleNotFoundError:  # pragma: no cover - Plotly installed in Colab",
        "    px = None",
        "",
        "pd.options.display.float_format = \"{:.2f}\".format",
        "sns.set_theme(style=\"whitegrid\", context=\"talk\")",
        "plt.rcParams.update(",
        "    {",
        "        \"axes.titlesize\": 18,",
        "        \"axes.titleweight\": \"bold\",",
        "        \"axes.labelsize\": 13,",
        "        \"axes.grid\": True,",
        "        \"grid.alpha\": 0.25,",
        "        \"figure.dpi\": 120,",
        "        \"axes.spines.top\": False,",
        "        \"axes.spines.right\": False,",
        "    }",
        ")",
        "",
        "STORY_KEYS = (",
        "    \"title\",",
        "    \"subtitle\",",
        "    \"claim\",",
        "    \"evidence\",",
        "    \"visual\",",
        "    \"takeaway\",",
        "    \"source\",",
        "    \"units\",",
        "    \"annotation\",",
        "    \"alt_text\",",
        ")",
        "",
        "",
        "def load_csv(path: Path, *, description: str = \"\", **read_kwargs) -> pd.DataFrame:",
        "    df = pd.read_csv(path, **read_kwargs)",
        "    label = description or path.name",
        "    print(",
        "        f\"✅ Loaded {label} with shape {df.shape[0]} rows × {df.shape[1]} columns.\"",
        "    )",
        "    return df",
        "",
        "",
        "def validate_columns(",
        "    df: pd.DataFrame, required: Sequence[str], *, df_name: str = \"DataFrame\"",
        ") -> None:",
        "    missing = [col for col in required if col not in df.columns]",
        "    if missing:",
        "        raise ValueError(f\"{df_name} is missing columns: {missing}\")",
        "    print(f\"✅ {df_name} includes required columns: {', '.join(required)}\")",
        "",
        "",
        "def expect_rows_between(",
        "    df: pd.DataFrame,",
        "    lower: int,",
        "    upper: int,",
        "    *,",
        "    df_name: str = \"DataFrame\",",
        ") -> None:",
        "    rows = len(df)",
        "    if not (lower <= rows <= upper):",
        "        raise ValueError(",
        "            f\"{df_name} has {rows} rows; expected between {lower} and {upper}.\"",
        "        )",
        "    print(f\"✅ {df_name} row count {rows} within [{lower}, {upper}].\")",
        "",
        "",
        "def quick_null_check(df: pd.DataFrame, *, df_name: str = \"DataFrame\") -> pd.Series:",
        "    nulls = df.isna().sum()",
        "    print(f\"{df_name} missing values per column:\\n{nulls}\")",
        "    return nulls",
        "",
        "",
        "def quick_preview(",
        "    df: pd.DataFrame, *, n: int = 5, df_name: str = \"DataFrame\"",
        ") -> pd.DataFrame:",
        "    print(f\"🔍 Previewing {df_name} (first {n} rows):\")",
        "    return df.head(n)",
        "",
        "",
        "def numeric_sanity_check(",
        "    series: pd.Series,",
        "    *,",
        "    minimum: float | None = None,",
        "    maximum: float | None = None,",
        "    name: str = \"Series\",",
        ") -> None:",
        "    if minimum is not None and series.min() < minimum:",
        "        raise ValueError(",
        "            f\"{name} has values below the expected minimum of {minimum}.\"",
        "        )",
        "    if maximum is not None and series.max() > maximum:",
        "        raise ValueError(",
        "            f\"{name} has values above the expected maximum of {maximum}.\"",
        "        )",
        "    print(",
        "        f\"✅ {name} within expected range\"",
        "        f\"{f' ≥ {minimum}' if minimum is not None else ''}\"",
        "        f\"{f' and ≤ {maximum}' if maximum is not None else ''}.\"",
        "    )",
        "",
        "",
        "def story_fields_are_complete(story: Mapping[str, str]) -> None:",
        "    missing = [key for key in STORY_KEYS if not str(story.get(key, \"\")).strip()]",
        "    if missing:",
        "        raise ValueError(",
        "            \"Please complete the storytelling scaffold before plotting: \"",
        "            + \", \".join(missing)",
        "        )",
        "    print(",
        "        \"✅ Story scaffold complete (title, subtitle, claim, evidence, visual,\"",
        "        \" takeaway, source, units, annotation, alt text).\"",
        "    )",
        "",
        "",
        "def print_story_scaffold(story: Mapping[str, str]) -> None:",
        "    story_fields_are_complete(story)",
        "    print(\"\\n📖 Story Scaffold\")",
        "    print(f\"Claim: {story['claim']}\")",
        "    print(f\"Evidence: {story['evidence']}\")",
        "    print(f\"Visual focus: {story['visual']}\")",
        "    print(f\"Takeaway: {story['takeaway']}\")",
        "    print(f\"Source: {story['source']} ({story['units']})\")",
        "",
        "",
        "def apply_matplotlib_story(ax: plt.Axes, story: Mapping[str, str]) -> None:",
        "    story_fields_are_complete(story)",
        "    ax.set_title(f\"{story['title']}\\n{story['subtitle']}\", loc=\"left\", pad=18)",
        "    ax.figure.text(",
        "        0.01,",
        "        -0.08,",
        "        (",
        "            f\"Claim: {story['claim']} | Evidence: {story['evidence']}\"",
        "            f\" | Takeaway: {story['takeaway']}\"",
        "            f\"\\nSource: {story['source']} • Units: {story['units']}\"",
        "        ),",
        "        ha=\"left\",",
        "        fontsize=10,",
        "    )",
        "",
        "",
        "def annotate_callout(",
        "    ax: plt.Axes,",
        "    *,",
        "    xy: tuple[float, float],",
        "    xytext: tuple[float, float],",
        "    text: str,",
        ") -> None:",
        "    ax.annotate(",
        "        text,",
        "        xy=xy,",
        "        xytext=xytext,",
        "        arrowprops=dict(arrowstyle=\"->\", color=\"black\", lw=1),",
        "        bbox=dict(boxstyle=\"round,pad=0.3\", fc=\"white\", ec=\"black\", alpha=0.8),",
        "    )",
        "",
        "",
        "def record_alt_text(text: str) -> None:",
        "    print(f\"📝 Alt text ready: {text}\")",
        "",
        "",
        "def accessibility_checklist(",
        "    *, palette: str, has_alt_text: bool, contrast_passed: bool = True",
        ") -> None:",
        "    print(\"♿ Accessibility checklist:\")",
        "    print(f\" • Palette: {palette}\")",
        "    print(",
        "        f\" • Alt text provided: {'yes' if has_alt_text else 'add alt text before sharing'}\"",
        "    )",
        "    print(f\" • Contrast OK: {'yes' if contrast_passed else 'adjust colors'}\")",
        "",
        "",
        "def save_figure(fig: plt.Figure, filename: str) -> Path:",
        "    plots_dir = Path.cwd() / \"plots\"",
        "    plots_dir.mkdir(parents=True, exist_ok=True)",
        "    output_path = plots_dir / filename",
        "    fig.savefig(output_path, dpi=300, bbox_inches=\"tight\")",
        "    print(f\"💾 Saved figure to {output_path}\")",
        "    return output_path",
        "",
        "",
        "def save_plotly_figure(fig, filename: str) -> Path:",
        "    plots_dir = Path.cwd() / \"plots\"",
        "    plots_dir.mkdir(parents=True, exist_ok=True)",
        "    html_path = plots_dir / filename.replace(\".png\", \".html\")",
        "    fig.write_html(html_path)",
        "    print(f\"💾 Saved interactive figure to {html_path}\")",
        "    try:",
        "        static_path = plots_dir / filename",
        "        fig.write_image(str(static_path))",
        "        print(f\"💾 Saved static image to {static_path}\")",
        "    except Exception as exc:  # pragma: no cover - depends on kaleido",
        "        print(f\"⚠️ Static export skipped: {exc}\")",
        "    return html_path",
    ]
)


def colab_badge_cell(url: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        dedent(
            f"""
            ## 🔗 Open This Notebook in Google Colab

            [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})
            """
        ).strip()
    )


def learning_path_cell(objectives: list[str]) -> nbf.NotebookNode:
    bullet_list = "\n".join(f"- [ ] {item}" for item in objectives)
    return nbf.v4.new_markdown_cell(
        dedent(
            f"""
            ### ⏱️ Learning Path for Today

            Each loop takes about 10–15 minutes:
            {bullet_list}

            > 👩‍🏫 **Teacher tip:** Use these checkpoints for quick formative assessment. Have students raise a colored card after each check cell to signal confidence or questions.
            """
        ).strip()
    )


def teacher_sidebar_cell(timing: str, misconceptions: str, extensions: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        dedent(
            f"""
            > ### 👩‍🏫 Teacher Sidebar
            > **Suggested timing:** {timing}
            >
            > **Likely misconceptions:** {misconceptions}
            >
            > **Fast finisher extension:** {extensions}
            """
        ).strip()
    )


def data_card_cell(title: str, source: str, coverage: str, units: str, notes: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(
        dedent(
            f"""
            ### 🗂️ Data Card
            | Field | Details |
            | --- | --- |
            | **Dataset** | {title} |
            | **Source & link** | {source} |
            | **Temporal / spatial coverage** | {coverage} |
            | **Key units** | {units} |
            | **Method & caveats** | {notes} |
            """
        ).strip()
    )


def setup_paths_cell() -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(
        dedent(
            """
            from pathlib import Path

            DATA_DIR = Path.cwd() / "data"
            PLOTS_DIR = Path.cwd() / "plots"
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)

            print(f"Data directory: {DATA_DIR}")
            print(f"Plots directory: {PLOTS_DIR}")
            """
        ).strip()
    )

def make_day01(metadata: dict) -> None:
    cells: list[nbf.NotebookNode] = []
    cells.append(
        colab_badge_cell(
            "https://colab.research.google.com/github/DavidLangworthy/ds4s/blob/master/Day%201_%20Introduction%20%E2%80%93%20Climate%20Change%20%26%20Basic%20Plotting.ipynb"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # 🌎 Day 1 – Visualizing Global Warming
                ### Introduction to Jupyter, Python, and Climate Data

                Welcome! Today you will:
                1. Load NASA’s global temperature anomaly data.
                2. Practice quick diagnostics so mistakes are caught early.
                3. Build a climate line chart with a clear claim-evidence-takeaway story.
                """
            ).strip()
        )
    )
    cells.append(
        data_card_cell(
            title="NASA GISTEMP Global Surface Temperature Anomalies",
            source="NASA Goddard Institute for Space Studies — [GISTEMP v4](https://data.giss.nasa.gov/gistemp/)",
            coverage="Global, annual mean anomaly, 1880–present",
            units="Temperature anomaly in °C relative to 1951–1980 baseline",
            notes="Annual means compiled from meteorological stations and sea-surface temperatures. 2024 values are preliminary and may be revised.",
        )
    )
    cells.append(
        learning_path_cell(
            [
                "Set up folders and shared utilities (you run this once).",
                "Load and validate the temperature dataset.",
                "Run quick diagnostics (shape, columns, nulls, ranges).",
                "Smooth the signal with a 5-year rolling average.",
                "Build the annotated climate line chart and log accessibility checks.",
            ]
        )
    )
    cells.append(
        teacher_sidebar_cell(
            timing="~45 minutes including discussion pauses.",
            misconceptions="Confusing absolute temperature with anomaly; forgetting that missing values exist before 1880.",
            extensions="Compare land-only vs. ocean-only anomalies using additional columns in the CSV.",
        )
    )
    cells.append(nbf.v4.new_code_cell(UTILITIES_CODE.strip()))
    cells.append(setup_paths_cell())
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 1 · Load & Label the Data
                Learn how to read the CSV and confirm the schema.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                temperature_path = DATA_DIR / "GLB.Ts+dSST.csv"

                df_raw = load_csv(
                    temperature_path,
                    description="NASA GISTEMP anomalies",
                    skiprows=1,
                    usecols=[0, 13],
                    names=["Year", "TempAnomaly"],
                    header=0,
                )

                validate_columns(df_raw, ["Year", "TempAnomaly"], df_name="temperature anomalies")
                expect_rows_between(df_raw, 140, 200, df_name="temperature anomalies")
                df_raw["TempAnomaly"] = pd.to_numeric(df_raw["TempAnomaly"], errors="coerce")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                quick_preview(df_raw, n=5, df_name="temperature anomalies")
                quick_null_check(df_raw, df_name="temperature anomalies")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 2 · Clean & Focus the Time Series
                Tighten the data to the years we can trust and prepare helper columns.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                df_temp = (
                    df_raw.dropna(subset=["TempAnomaly"]).query("Year >= 1880").reset_index(drop=True)
                )
                df_temp["Rolling5"] = df_temp["TempAnomaly"].rolling(window=5, center=True).mean()
                expect_rows_between(df_temp, 140, 170, df_name="clean anomalies")
                numeric_sanity_check(df_temp["TempAnomaly"], minimum=-1.0, maximum=1.5, name="TempAnomaly (°C)")
                numeric_sanity_check(df_temp["Rolling5"].dropna(), minimum=-1.0, maximum=1.5, name="Rolling5 (°C)")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                quick_preview(df_temp, n=5, df_name="clean anomalies")
                quick_preview(df_temp.tail(), n=5, df_name="recent anomalies")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 3 · Plot with Story-first Scaffolding
                Fill out the storytelling checklist **before** drawing the figure.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                latest_year = int(df_temp["Year"].iloc[-1])
                latest_value = df_temp["Rolling5"].dropna().iloc[-1]

                story = {
                    "title": "Earth is Warming Faster Than the 20th Century Baseline",
                    "subtitle": f"Global surface temperature anomalies, 1880–{latest_year}",
                    "claim": "Recent decades are the warmest in the 140+ year record.",
                    "evidence": "5-year rolling average now sits over 1°C above the 1951–1980 baseline.",
                    "visual": "Line chart comparing annual anomalies with a smoothed trend.",
                    "takeaway": "Warming is persistent and accelerating, underscoring urgency for mitigation.",
                    "source": "NASA GISTEMP v4",
                    "units": "Temperature anomaly (°C relative to 1951–1980)",
                    "annotation": f"{latest_year}: {latest_value:.2f}°C above baseline",
                    "alt_text": (
                        "Line chart showing annual global temperature anomalies from 1880 to "
                        f"{latest_year} with a 5-year rolling average climbing from near 0°C to over "
                        f"1°C above the mid-20th century baseline."
                    ),
                }

                print_story_scaffold(story)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                fig, ax = plt.subplots(figsize=(11, 6))
                ax.plot(
                    df_temp["Year"],
                    df_temp["TempAnomaly"],
                    label="Annual anomaly",
                    color="#d1495b",
                    alpha=0.4,
                )
                ax.plot(
                    df_temp["Year"],
                    df_temp["Rolling5"],
                    label="5-year rolling average",
                    color="#00798c",
                    linewidth=3,
                )
                ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
                ax.set_xlabel("Year")
                ax.set_ylabel("Temperature anomaly (°C)")
                ax.legend(loc="upper left")

                apply_matplotlib_story(ax, story)
                annotate_callout(
                    ax,
                    xy=(latest_year, latest_value),
                    xytext=(1975, 1.2),
                    text=story["annotation"],
                )

                record_alt_text(story["alt_text"])
                accessibility_checklist(palette="Colorbrewer warm/cool contrast", has_alt_text=True)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                save_figure(fig, "day01_solution_plot.png")
                """
            ).strip()
        )
    )

    nb = nbf.v4.new_notebook(metadata=metadata)
    nb["cells"] = cells
    output_path = REPO_ROOT / "days" / "day01" / "solution" / "day01_solution.ipynb"
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

def make_day02(metadata: dict) -> None:
    cells: list[nbf.NotebookNode] = []
    cells.append(
        colab_badge_cell(
            "https://colab.research.google.com/github/DavidLangworthy/ds4s/blob/master/Day%202%20%E2%80%93%20Fossil%20Fuels%20vs.%20Renewables.ipynb"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # ⚡ Day 2 – Exploring Energy Trends
                ### Fossil Fuels vs. Renewables in a Warming World

                Today you will investigate how quickly the world is scaling up renewable energy. The micro-loops walk you through loading multiple datasets, merging them safely, and communicating uncertainty.
                """
            ).strip()
        )
    )
    cells.append(
        data_card_cell(
            title="Our World in Data – Renewable Energy Share Datasets",
            source="OWID Energy Data Explorer — [Renewables share](https://ourworldindata.org/grapher/renewable-share-energy)",
            coverage="Global (World aggregate), annual values 1965–2023",
            units="Share of primary energy from renewables (% of total)",
            notes="Individual CSVs per technology. Shares are calculated from primary energy equivalents; small discrepancies occur across series due to rounding.",
        )
    )
    cells.append(
        learning_path_cell(
            [
                "Inventory and load each renewable energy CSV.",
                "Filter to the world aggregate and align the schemas.",
                "Merge sources into a tidy long-form table with diagnostics.",
                "Plot stacked area + line overlays with storytelling scaffold.",
            ]
        )
    )
    cells.append(
        teacher_sidebar_cell(
            timing="~50 minutes including discussion.",
            misconceptions="Mixing percentage units with absolute energy; forgetting to align on the same year column.",
            extensions="Have students add biofuels or geothermal series and compare regional splits.",
        )
    )
    cells.append(nbf.v4.new_code_cell(UTILITIES_CODE.strip()))
    cells.append(setup_paths_cell())
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 1 · Load Renewable Series
                Read each CSV and document its structure so merges are predictable.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                renewable_files = {
                    "total": "01 renewable-share-energy.csv",
                    "hydro": "06 hydro-share-energy.csv",
                    "wind": "10 wind-share-energy.csv",
                    "solar": "14 solar-share-energy.csv",
                }

                dfs_world = {}
                for label, filename in renewable_files.items():
                    df = load_csv(
                        DATA_DIR / filename,
                        description=f"OWID {label} renewables share",
                    )
                    validate_columns(
                        df,
                        ["Entity", "Code", "Year"],
                        df_name=f"{label} share",
                    )
                    dfs_world[label] = df.query("Entity == 'World'").copy()
                    expect_rows_between(
                        dfs_world[label], 50, 70, df_name=f"world {label}"
                    )
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                {key: quick_preview(value, n=3, df_name=f"world {key}") for key, value in dfs_world.items()}
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 2 · Harmonise & Merge
                Keep only the needed columns, rename consistently, and combine for plotting.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                df_total = dfs_world["total"][["Year", "Renewables (% equivalent primary energy)"]].rename(
                    columns={"Renewables (% equivalent primary energy)": "Total Renewable"}
                )
                df_components = []
                rename_map = {
                    "hydro": "Hydro",
                    "wind": "Wind",
                    "solar": "Solar",
                }
                for key, label in rename_map.items():
                    df_part = dfs_world[key][["Year", f"{label} (% equivalent primary energy)"]].rename(
                        columns={f"{label} (% equivalent primary energy)": label}
                    )
                    df_components.append(df_part)

                df_merged = df_total.copy()
                for component in df_components:
                    df_merged = df_merged.merge(component, on="Year", how="inner")

                expect_rows_between(df_merged, 50, 70, df_name="merged renewables")
                quick_null_check(df_merged, df_name="merged renewables")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                quick_preview(df_merged, n=5, df_name="merged renewables")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 3 · Reshape for Storytelling
                Move to a tidy, long format and compute contribution shares.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                value_columns = ["Hydro", "Wind", "Solar"]
                df_long = df_merged.melt(
                    id_vars="Year", value_vars=value_columns, var_name="Technology", value_name="Share"
                )
                numeric_sanity_check(df_long["Share"], minimum=0, maximum=50, name="Renewables share (%)")
                quick_preview(df_long, n=6, df_name="tidy renewables")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 4 · Visualise & Annotate
                Use the storytelling scaffold before drawing the figure.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                latest_year = int(df_merged["Year"].iloc[-1])
                latest_total = df_merged["Total Renewable"].iloc[-1]

                story = {
                    "title": "Renewables Are Growing — But Fossil Fuels Still Dominate",
                    "subtitle": f"World share of primary energy from renewables, 1965–{latest_year}",
                    "claim": "Renewable energy has more than doubled since 2000, led by wind and solar.",
                    "evidence": (
                        f"Total renewable share reached {latest_total:.1f}% globally with solar/wind rising fastest."
                    ),
                    "visual": "Stacked area for hydro/wind/solar with total share overlay.",
                    "takeaway": "Growth is accelerating yet absolute share remains below 15%, highlighting decarbonisation gap.",
                    "source": "Our World in Data (BP Statistical Review)",
                    "units": "% of primary energy",
                    "annotation": f"{latest_year}: {latest_total:.1f}% of world energy",
                    "alt_text": (
                        "Stacked area chart showing hydro stable near 6%, with wind and solar climbing sharply after 2005, bringing"
                        f" total renewables to roughly {latest_total:.0f}% of global primary energy by {latest_year}."
                    ),
                }

                print_story_scaffold(story)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                fig, ax = plt.subplots(figsize=(11, 6))

                technologies = ["Hydro", "Wind", "Solar"]
                colors = {"Hydro": "#4c78a8", "Wind": "#9ecae9", "Solar": "#f28e2b"}

                ax.stackplot(
                    df_merged["Year"],
                    [df_merged[tech] for tech in technologies],
                    labels=[f"{tech} share" for tech in technologies],
                    colors=[colors[tech] for tech in technologies],
                    alpha=0.8,
                )
                ax.plot(
                    df_merged["Year"],
                    df_merged["Total Renewable"],
                    color="#2f4b7c",
                    linewidth=3,
                    label="Total renewable share",
                )
                ax.set_ylabel("Share of primary energy (%)")
                ax.set_xlabel("Year")
                ax.set_ylim(bottom=0)
                ax.legend(loc="upper left", ncol=2)

                apply_matplotlib_story(ax, story)
                annotate_callout(
                    ax,
                    xy=(latest_year, latest_total),
                    xytext=(1995, latest_total + 5),
                    text=story["annotation"],
                )

                record_alt_text(story["alt_text"])
                accessibility_checklist(
                    palette="Colorblind-safe blues and orange overlay",
                    has_alt_text=True,
                )
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                save_figure(fig, "day02_solution_plot.png")
                """
            ).strip()
        )
    )

    nb = nbf.v4.new_notebook(metadata=metadata)
    nb["cells"] = cells
    output_path = REPO_ROOT / "days" / "day02" / "solution" / "day02_solution.ipynb"
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

def make_day03(metadata: dict) -> None:
    cells: list[nbf.NotebookNode] = []
    cells.append(
        colab_badge_cell(
            "https://colab.research.google.com/github/DavidLangworthy/ds4s/blob/master/Day%203_%20Pollution%20and%20Public%20Health.ipynb"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # 🌫️ Day 3 – Visualizing Pollution and Public Health
                ### How Air Quality and Economic Development Intersect

                Today you’ll blend two datasets, add diagnostics, and build an explorable scatter plot that connects air pollution with income.
                """
            ).strip()
        )
    )
    cells.append(
        data_card_cell(
            title="World Bank – PM2.5 Exposure & GDP per Capita",
            source="World Bank DataBank — [EN.ATM.PM25.MC.M3](https://data.worldbank.org/indicator/EN.ATM.PM25.MC.M3)",
            coverage="Country-level, annual, 1990–2019",
            units="PM2.5 exposure (µg/m³), GDP per capita (current USD)",
            notes="GDP data use current USD; PM2.5 exposure estimates modelled from satellite retrievals. Missing data in small nations are common.",
        )
    )
    cells.append(
        learning_path_cell(
            [
                "Load both datasets and extract the target year.",
                "Run diagnostics (shape, nulls, ranges) before merging.",
                "Merge, clean, and engineer bins for interpretation.",
                "Build an interactive scatter with storytelling and accessibility.",
            ]
        )
    )
    cells.append(
        teacher_sidebar_cell(
            timing="~45 minutes including gallery walk.",
            misconceptions="Interpreting correlation as causation; forgetting to log-scale income axis.",
            extensions="Segment by World Bank income groups or regions and compare slopes.",
        )
    )
    cells.append(nbf.v4.new_code_cell(UTILITIES_CODE.strip()))
    cells.append(setup_paths_cell())
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 1 · Load & Slice 2019 Data
                Work with a single recent year for a clear snapshot.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                pm_path = DATA_DIR / "pm25_exposure.csv"
                gdp_path = DATA_DIR / "gdp_per_country.csv"

                df_pm = load_csv(pm_path, description="World Bank PM2.5 exposure")
                df_gdp = load_csv(gdp_path, description="World Bank GDP per capita")

                validate_columns(
                    df_pm,
                    ["Country Name", "Country Code", "2019"],
                    df_name="PM dataset",
                )
                validate_columns(
                    df_gdp,
                    ["Country Name", "Country Code", "2019"],
                    df_name="GDP dataset",
                )

                df_pm_2019 = df_pm[["Country Name", "Country Code", "2019"]].rename(
                    columns={"2019": "PM25"}
                )
                df_gdp_2019 = df_gdp[["Country Name", "Country Code", "2019"]].rename(
                    columns={"2019": "GDP_per_capita"}
                )
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                quick_preview(df_pm_2019, n=5, df_name="PM 2019")
                quick_preview(df_gdp_2019, n=5, df_name="GDP 2019")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 2 · Merge & Diagnose
                Catch mismatched records and nulls before plotting.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                df_merged = pd.merge(
                    df_pm_2019,
                    df_gdp_2019,
                    on=["Country Name", "Country Code"],
                    how="inner",
                )
                df_merged = df_merged.dropna()
                df_merged["GDP_per_capita"] = pd.to_numeric(df_merged["GDP_per_capita"], errors="coerce")
                df_merged = df_merged.dropna()

                expect_rows_between(df_merged, 140, 190, df_name="PM25+GDP merged")
                numeric_sanity_check(df_merged["PM25"], minimum=1, maximum=120, name="PM2.5 (µg/m³)")
                numeric_sanity_check(
                    df_merged["GDP_per_capita"], minimum=300, maximum=120000, name="GDP per capita (USD)"
                )
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                quick_preview(df_merged, n=5, df_name="merged snapshot")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 3 · Enrich for Storytelling
                Classify countries into broad income groups for annotation.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                bins = [0, 4000, 13000, 25000, 1000000]
                labels = ["Low", "Lower-middle", "Upper-middle", "High"]
                df_merged["IncomeGroup"] = pd.cut(
                    df_merged["GDP_per_capita"], bins=bins, labels=labels, include_lowest=True
                )
                income_counts = df_merged["IncomeGroup"].value_counts().sort_index()
                print("Income group counts:\n", income_counts)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 4 · Build the Plotly Story
                Apply the storytelling scaffold, annotation, and accessibility checks.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                worst_pm = df_merged.nlargest(1, "PM25").iloc[0]
                cleanest = df_merged.nsmallest(1, "PM25").iloc[0]

                story = {
                    "title": "Air Pollution Drops as Economies Grow — With Stark Exceptions",
                    "subtitle": "PM2.5 exposure vs. GDP per capita (2019)",
                    "claim": "Higher-income countries generally breathe cleaner air, yet some lower-middle income nations face severe pollution.",
                    "evidence": (
                        f"{worst_pm['Country Name']} reports PM2.5 above {worst_pm['PM25']:.0f} µg/m³ while {cleanest['Country Name']} sits near {cleanest['PM25']:.0f} µg/m³."
                    ),
                    "visual": "Log-scale scatter with hover details and income-group color encoding.",
                    "takeaway": "Economic growth helps but is not sufficient; targeted clean air policies are essential.",
                    "source": "World Bank (2024 update)",
                    "units": "PM2.5 (µg/m³), GDP per capita (USD)",
                    "annotation": f"{worst_pm['Country Name']} faces the highest PM2.5 exposure",
                    "alt_text": (
                        "Scatter plot on log x-axis showing most wealthy countries clustered at low PM2.5 levels,"
                        " while lower income nations span much higher pollution levels with wide variation."
                    ),
                }

                print_story_scaffold(story)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                fig = px.scatter(
                    df_merged,
                    x="GDP_per_capita",
                    y="PM25",
                    color="IncomeGroup",
                    hover_name="Country Name",
                    size="PM25",
                    size_max=18,
                    log_x=True,
                    template="plotly_white",
                    labels={
                        "GDP_per_capita": "GDP per capita (USD, log scale)",
                        "PM25": "PM2.5 exposure (µg/m³)",
                    },
                )

                fig.update_layout(
                    legend_title="World Bank income group",
                    margin=dict(l=40, r=40, t=80, b=120),
                )

                fig.add_annotation(
                    x=worst_pm["GDP_per_capita"],
                    y=worst_pm["PM25"],
                    text=story["annotation"],
                    showarrow=True,
                    arrowcolor="#d1495b",
                )

                story_fields_are_complete(story)
                fig.update_layout(
                    title=dict(
                        text=f"<b>{story['title']}</b><br><sup>{story['subtitle']}</sup>",
                        x=0,
                        xanchor="left",
                    )
                )
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.25,
                    align="left",
                    showarrow=False,
                    text=(
                        f"Claim: {story['claim']}<br>Evidence: {story['evidence']}<br>Takeaway: {story['takeaway']}<br>Source: {story['source']} • Units: {story['units']}"
                    ),
                )

                record_alt_text(story["alt_text"])
                accessibility_checklist(
                    palette="Colorblind-safe Plotly qualitative", has_alt_text=True
                )

                fig.show()
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                save_plotly_figure(fig, "day03_solution_plot.png")
                """
            ).strip()
        )
    )

    nb = nbf.v4.new_notebook(metadata=metadata)
    nb["cells"] = cells
    output_path = REPO_ROOT / "days" / "day03" / "solution" / "day03_solution.ipynb"
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

def make_day04(metadata: dict) -> None:
    cells: list[nbf.NotebookNode] = []
    cells.append(
        colab_badge_cell(
            "https://colab.research.google.com/github/DavidLangworthy/ds4s/blob/master/Day%204_%20Mapping%20Biodiversity%20%26%20Deforestation.ipynb"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # 🌍 Day 4 – Mapping Biodiversity & Deforestation
                ### Advanced Visualization: Forest Loss Over Time

                Today’s focus is geographic storytelling. You’ll animate forest cover change, highlight uncertainty, and practice ethical map design.
                """
            ).strip()
        )
    )
    cells.append(
        data_card_cell(
            title="World Bank – Forest Area (% of Land Area)",
            source="World Bank SDG Atlas — [AG.LND.FRST.ZS](https://data.worldbank.org/indicator/AG.LND.FRST.ZS)",
            coverage="Country-level, annual 1990–2020",
            units="Forest area as % of total land area",
            notes="Interpolated values for some countries. Differences in national reporting can introduce year-to-year noise.",
        )
    )
    cells.append(
        learning_path_cell(
            [
                "Load and inspect the long-form forest dataset.",
                "Check for missing values and prepare friendly metadata.",
                "Design the choropleth scale and narrative scaffold.",
                "Render the animated map and document limitations.",
            ]
        )
    )
    cells.append(
        teacher_sidebar_cell(
            timing="~55 minutes including reflection.",
            misconceptions="Interpreting lighter colors as ‘better’ without context; forgetting map projections.",
            extensions="Ask students to compute forest loss by region and annotate biodiversity hotspots.",
        )
    )
    cells.append(nbf.v4.new_code_cell(UTILITIES_CODE.strip()))
    cells.append(setup_paths_cell())
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 1 · Load & Validate Forest Data
                Confirm schema and range before designing the map.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                forest_path = DATA_DIR / "forest_area_long.csv"
                df_forest = load_csv(
                    forest_path, description="World Bank forest area long-form"
                )
                validate_columns(
                    df_forest,
                    ["Country Name", "Country Code", "Year", "ForestPercent"],
                    df_name="forest data",
                )
                expect_rows_between(df_forest, 5000, 8000, df_name="forest data")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                quick_preview(df_forest, n=5, df_name="forest data")
                quick_null_check(df_forest, df_name="forest data")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 2 · Prepare Metadata & Diagnostics
                Clean types, compute regional summaries, and surface limitations.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                df_forest["Year"] = df_forest["Year"].astype(int)
                numeric_sanity_check(
                    df_forest["ForestPercent"], minimum=0, maximum=100, name="ForestPercent"
                )

                global_summary = df_forest.groupby("Year")["ForestPercent"].mean()
                quick_preview(
                    global_summary.reset_index().tail(),
                    n=5,
                    df_name="global forest % by year",
                )

                caveats = (
                    "Some countries interpolate census years; small island nations may report 0% forest despite mangroves."
                )
                print("Caveats to mention: ", caveats)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 3 · Story Scaffold for the Map
                Capture the claim, evidence, and annotation before rendering.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                latest_year = df_forest["Year"].max()
                earliest_year = df_forest["Year"].min()
                brazil_recent = df_forest[
                    (df_forest["Country Name"] == "Brazil") & (df_forest["Year"] == latest_year)
                ]["ForestPercent"].iloc[0]
                brazil_past = df_forest[
                    (df_forest["Country Name"] == "Brazil") & (df_forest["Year"] == earliest_year)
                ]["ForestPercent"].iloc[0]

                story = {
                    "title": "Forest Cover Has Fallen in Key Biodiversity Hotspots",
                    "subtitle": f"Share of land area covered by forest, {earliest_year}–{latest_year}",
                    "claim": "Global forest cover is shrinking, with pronounced declines in tropical nations.",
                    "evidence": (
                        f"Brazil dropped from {brazil_past:.1f}% forest cover in {earliest_year} to {brazil_recent:.1f}% in {latest_year}."
                    ),
                    "visual": "Animated choropleth highlighting forest percent by country.",
                    "takeaway": "Protecting remaining forests is critical for biodiversity and carbon sinks.",
                    "source": "World Bank SDG Atlas (2024 download)",
                    "units": "% of national land area covered by forest",
                    "annotation": "Brazil’s forest share has declined in three decades",
                    "alt_text": (
                        "Animated world map where deep green indicates high forest share; many tropical countries lighten"
                        f" between {earliest_year} and {latest_year}, showing shrinking forest cover."
                    ),
                }

                print_story_scaffold(story)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 4 · Render the Animated Choropleth
                Apply ethical mapping practices: consistent scales, clear annotation, and limitations.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                fig = px.choropleth(
                    df_forest,
                    locations="Country Code",
                    color="ForestPercent",
                    hover_name="Country Name",
                    animation_frame="Year",
                    color_continuous_scale="Greens",
                    range_color=[0, 100],
                    labels={"ForestPercent": "Forest area (% of land)"},
                )

                fig.update_layout(
                    title=dict(
                        text=f"<b>{story['title']}</b><br><sup>{story['subtitle']}</sup>",
                        x=0,
                        xanchor="left",
                    ),
                    coloraxis_colorbar=dict(title="% forest"),
                    margin=dict(l=10, r=10, t=80, b=120),
                )

                story_fields_are_complete(story)
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.2,
                    align="left",
                    showarrow=False,
                    text=(
                        f"Claim: {story['claim']}<br>Evidence: {story['evidence']}<br>Takeaway: {story['takeaway']}<br>Source: {story['source']} • Units: {story['units']}"
                    ),
                )

                fig.add_annotation(
                    x=0.8,
                    y=0.2,
                    xref="paper",
                    yref="paper",
                    text=story["annotation"],
                    showarrow=True,
                    arrowcolor="#386641",
                    arrowhead=2,
                )

                record_alt_text(story["alt_text"])
                accessibility_checklist(
                    palette="Sequential greens with fixed endpoints",
                    has_alt_text=True,
                )

                fig.show()
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ### 🔍 Map Limitations to Discuss
                - Interpolated values can create smooth trends where reality is jagged.
                - Choropleths emphasise area, so large countries appear visually dominant.
                - Animations can hide precise numbers; pair with a static chart for reporting.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                save_plotly_figure(fig, "day04_solution_plot.png")
                """
            ).strip()
        )
    )

    nb = nbf.v4.new_notebook(metadata=metadata)
    nb["cells"] = cells
    output_path = REPO_ROOT / "days" / "day04" / "solution" / "day04_solution.ipynb"
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

def make_day05(metadata: dict) -> None:
    cells: list[nbf.NotebookNode] = []
    cells.append(
        colab_badge_cell(
            "https://colab.research.google.com/github/DavidLangworthy/ds4s/blob/master/Day%205_%20Capstone%20%E2%80%93%20CO%E2%82%82%20Emissions%20%26%20Global%20Temperature.ipynb"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # 🔥 Day 5 – Capstone Project: CO₂ and Climate
                ### Telling the Big Story with Data

                The capstone weaves together data cleaning, diagnostics, multi-source merging, and a polished narrative. You’ll build a two-panel story that shows how human-caused CO₂ emissions track with global temperature change.
                """
            ).strip()
        )
    )
    cells.append(
        data_card_cell(
            title="Global Fossil CO₂ Emissions & NASA GISTEMP",
            source="Our World in Data (Global Carbon Project) & NASA GISTEMP v4",
            coverage="Global totals, annual 1880–2023",
            units="CO₂: gigatonnes per year • Temperature: anomaly in °C (1951–1980 baseline)",
            notes="CO₂ data aggregates fossil fuels and cement. Temperature anomalies follow NASA’s 1951–1980 baseline; latest year provisional.",
        )
    )
    cells.append(
        learning_path_cell(
            [
                "Load and validate the CO₂ and temperature datasets.",
                "Align years and run diagnostics on ranges and nulls.",
                "Engineer summary metrics for evidence statements.",
                "Compose a two-panel narrative figure with accessibility checks.",
            ]
        )
    )
    cells.append(
        teacher_sidebar_cell(
            timing="~60 minutes including peer feedback.",
            misconceptions="Assuming correlation equals causation; misreading anomalies as absolute temperatures.",
            extensions="Invite students to add emissions per capita or mitigation milestones as context.",
        )
    )
    cells.append(nbf.v4.new_code_cell(UTILITIES_CODE.strip()))
    cells.append(setup_paths_cell())
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 1 · Load & Inspect Both Datasets
                Keep schemas transparent before merging anything.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                co2_path = DATA_DIR / "global_co2.csv"
                temp_path = DATA_DIR / "GLB.Ts+dSST.csv"

                df_co2 = load_csv(co2_path, description="Global fossil CO₂ emissions")
                df_temp_raw = load_csv(
                    temp_path,
                    description="NASA GISTEMP anomalies",
                    skiprows=1,
                    usecols=["Year", "J-D"],
                )

                validate_columns(df_co2, ["Year", "CO2"], df_name="CO₂ data")
                validate_columns(df_temp_raw, ["Year", "J-D"], df_name="temperature data")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                quick_preview(df_co2, n=5, df_name="CO₂ data")
                quick_preview(df_temp_raw, n=5, df_name="temperature data")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 2 · Clean, Align, and Diagnose
                Convert to numeric, align year index, and make sure both series overlap.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                df_co2["Year"] = pd.to_numeric(df_co2["Year"], errors="coerce")
                df_co2["CO2"] = pd.to_numeric(df_co2["CO2"], errors="coerce")

                df_temp = df_temp_raw.rename(columns={"J-D": "TempAnomaly"})
                df_temp["TempAnomaly"] = pd.to_numeric(df_temp["TempAnomaly"], errors="coerce")

                df_co2 = df_co2.dropna().set_index("Year")
                df_temp = df_temp.dropna().set_index("Year")

                df_merged = df_co2.join(df_temp, how="inner")
                df_merged = df_merged[df_merged.index >= 1880]

                expect_rows_between(df_merged, 120, 200, df_name="merged climate data")
                numeric_sanity_check(df_merged["CO2"], minimum=0, maximum=40, name="CO₂ (Gt)")
                numeric_sanity_check(df_merged["TempAnomaly"], minimum=-1.0, maximum=1.5, name="Temperature anomaly (°C)")
                quick_null_check(df_merged, df_name="merged climate data")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                quick_preview(df_merged.head(), n=5, df_name="merged climate data")
                quick_preview(df_merged.tail(), n=5, df_name="recent climate data")
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 3 · Engineer Evidence Metrics
                Quantify change for your claim-evidence statement.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                baseline_year = 1960
                start_year = df_merged.index.min()
                latest_year = int(df_merged.index.max())

                def safe_value(year: int, series: pd.Series) -> float:
                    if year in series.index:
                        return float(series.loc[year])
                    return float(series.loc[series.index[series.index.searchsorted(year)]])

                co2_baseline = safe_value(baseline_year, df_merged["CO2"])
                co2_latest = float(df_merged["CO2"].iloc[-1])
                temp_baseline = safe_value(baseline_year, df_merged["TempAnomaly"])
                temp_latest = float(df_merged["TempAnomaly"].iloc[-1])
                corr = df_merged["CO2"].corr(df_merged["TempAnomaly"])

                print(
                    f"Since {baseline_year}, CO₂ rose from {co2_baseline:.1f} Gt to {co2_latest:.1f} Gt." \
                    f" Temperature anomaly increased from {temp_baseline:.2f}°C to {temp_latest:.2f}°C."
                )
                print(f"Pearson correlation between series: {corr:.2f}")

                story = {
                    "title": "CO₂ Emissions and Global Temperatures Rise in Lockstep",
                    "subtitle": f"Global totals, {start_year}–{latest_year}",
                    "claim": "Burning fossil fuels drives a steep climb in atmospheric CO₂ and global temperature anomalies.",
                    "evidence": (
                        f"CO₂ emissions quadrupled after {baseline_year}, and temperature anomalies climbed ~{temp_latest - temp_baseline:.1f}°C." \
                        f" Correlation = {corr:.2f}."
                    ),
                    "visual": "Two-panel line chart sharing a timeline (top: CO₂, bottom: temperature anomaly).",
                    "takeaway": "Cutting emissions is essential to stabilise temperatures within agreed climate targets.",
                    "source": "Global Carbon Project & NASA GISTEMP (2024 release)",
                    "units": "CO₂ in gigatonnes; temperature anomaly in °C relative to 1951–1980",
                    "annotation": f"{latest_year}: {co2_latest:.1f} Gt CO₂ and {temp_latest:.2f}°C anomaly",
                    "alt_text": (
                        "Two aligned line charts from the late 1800s to present showing CO₂ emissions climbing from under 5 Gt"
                        f" to over {co2_latest:.0f} Gt while temperature anomalies rise from near 0°C to about {temp_latest:.1f}°C."
                    ),
                }

                print_story_scaffold(story)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ## Loop 4 · Compose the Capstone Figure
                Use a shared timeline, consistent storytelling scaffold, and explicit accessibility log.
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                fig, axes = plt.subplots(
                    2,
                    1,
                    figsize=(12, 9),
                    sharex=True,
                    gridspec_kw={"height_ratios": [2, 1.2], "hspace": 0.05},
                )

                ax_co2, ax_temp = axes
                ax_co2.plot(df_merged.index, df_merged["CO2"], color="#6a4c93", linewidth=3)
                ax_co2.set_ylabel("CO₂ emissions (Gt)")
                ax_co2.axvspan(1950, latest_year, color="#f6bd60", alpha=0.15, label="Great Acceleration")
                ax_co2.legend(loc="upper left")

                ax_temp.plot(df_merged.index, df_merged["TempAnomaly"], color="#ef476f", linewidth=2.5)
                ax_temp.axhline(0, color="black", linestyle="--", linewidth=1)
                ax_temp.set_ylabel("Temp anomaly (°C)")
                ax_temp.set_xlabel("Year")

                apply_matplotlib_story(ax_co2, story)
                annotate_callout(
                    ax_co2,
                    xy=(latest_year, co2_latest),
                    xytext=(1985, co2_latest - 10),
                    text=story["annotation"],
                )
                annotate_callout(
                    ax_temp,
                    xy=(latest_year, temp_latest),
                    xytext=(1940, temp_latest + 0.2),
                    text=f"{latest_year}: {temp_latest:.2f}°C anomaly",
                )

                record_alt_text(story["alt_text"])
                accessibility_checklist(
                    palette="Purple/rose contrast with shared timeline",
                    has_alt_text=True,
                )

                fig.align_ylabels(axes)
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ### 🧾 Reflection Prompt
                - Where might the causal chain break? What other evidence would you gather?
                - How could you adapt this chart for a policymaker vs. a public audience?
                """
            ).strip()
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                save_figure(fig, "day05_solution_plot.png")
                """
            ).strip()
        )
    )

    nb = nbf.v4.new_notebook(metadata=metadata)
    nb["cells"] = cells
    output_path = REPO_ROOT / "days" / "day05" / "solution" / "day05_solution.ipynb"
    with output_path.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)

def load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    return nb.metadata


def main() -> None:
    day_builders = [
        ("day01", make_day01),
        ("day02", make_day02),
        ("day03", make_day03),
        ("day04", make_day04),
        ("day05", make_day05),
    ]

    for folder, builder in day_builders:
        notebook_path = REPO_ROOT / "days" / folder / "solution" / f"{folder}_solution.ipynb"
        metadata = load_metadata(notebook_path)
        builder(metadata)
        print(f"Updated {notebook_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
