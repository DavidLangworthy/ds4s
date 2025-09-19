#!/usr/bin/env python3
"""Generate student starter notebooks from instructor solutions."""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import nbformat

ELIDE_MARKER = "# --- Starter Notebook elide"


def _sanitize_code_cell(cell: nbformat.NotebookNode) -> None:
    """Remove instructor-only snippets identified by ELIDE_MARKER comments."""
    source = cell.get("source", "")
    if ELIDE_MARKER not in source:
        # Also make sure student notebooks have no lingering outputs.
        cell["execution_count"] = None
        cell["outputs"] = []
        return

    lines = source.splitlines()
    new_lines: list[str] = []
    block_start = 0

    def insert_placeholder(message: str) -> None:
        cleaned = message.strip().strip("-")
        if not cleaned:
            return
        lower = cleaned.lower()
        if lower.startswith("remove"):
            return
        if lower.startswith("show"):
            cleaned = cleaned[4:].strip()
        cleaned = cleaned[:1].upper() + cleaned[1:]
        new_lines.append(f"# TODO: {cleaned}")

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            new_lines.append(line)
            block_start = len(new_lines)
            continue
        if line.lstrip().startswith(ELIDE_MARKER):
            message = line.split(":", 1)[1] if ":" in line else ""
            message = message.rsplit("---", 1)[0]
            new_lines[:] = new_lines[:block_start]
            insert_placeholder(message)
            block_start = len(new_lines)
            continue
        new_lines.append(line)

    cell["source"] = "\n".join(new_lines).rstrip() + "\n"
    cell["execution_count"] = None
    cell["outputs"] = []


def build_starter_notebook(solution_path: Path) -> nbformat.NotebookNode:
    solution_nb = nbformat.read(solution_path, as_version=4)
    starter_nb = copy.deepcopy(solution_nb)

    for cell in starter_nb.cells:
        if cell.get("cell_type") == "code":
            _sanitize_code_cell(cell)
        else:
            # Clear execution artifacts from any non-code cells just in case.
            cell.pop("outputs", None)

    return starter_nb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that the committed starter notebooks are up to date",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    solution_paths = sorted(repo_root.glob("days/day*/solution/*_solution.ipynb"))
    if not solution_paths:
        print("No solution notebooks found.", file=sys.stderr)
        return 1

    failed = False

    for solution_path in solution_paths:
        starter_path = solution_path.with_name(solution_path.name.replace("_solution", "_starter"))
        starter_path = starter_path.parent.parent / "notebook" / starter_path.name

        starter_nb = build_starter_notebook(solution_path)
        rendered = nbformat.writes(starter_nb)

        if args.check:
            if not starter_path.exists():
                print(f"Starter missing: {starter_path}", file=sys.stderr)
                failed = True
                continue
            existing = starter_path.read_text(encoding="utf-8")
            if existing != rendered:
                print(f"Starter out of date: {starter_path}", file=sys.stderr)
                failed = True
        else:
            starter_path.parent.mkdir(parents=True, exist_ok=True)
            starter_path.write_text(rendered, encoding="utf-8")
            print(f"Wrote {starter_path.relative_to(repo_root)}")

    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
