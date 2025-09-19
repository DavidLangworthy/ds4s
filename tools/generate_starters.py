#!/usr/bin/env python3
"""Generate learner starter notebooks from instructor solutions."""
from __future__ import annotations

import argparse
import copy
import difflib
import sys
from pathlib import Path

import nbformat

MARKER = "# --- Starter Notebook elide"


def strip_elided_source(source: str) -> str:
    """Remove instructor-only sections marked with the elide sentinel."""
    lines = source.split("\n")
    to_remove: set[int] = set()

    for idx, line in enumerate(lines):
        if not line.strip().startswith(MARKER):
            continue
        to_remove.add(idx)
        j = idx - 1
        while j >= 0 and lines[j].strip() == "":
            to_remove.add(j)
            j -= 1
        while j >= 0 and lines[j].strip() and not lines[j].strip().startswith(MARKER):
            to_remove.add(j)
            j -= 1

    filtered = [line for i, line in enumerate(lines) if i not in to_remove]
    while filtered and filtered[-1].strip() == "":
        filtered.pop()
    return "\n".join(filtered)


def build_starter_notebook(solution_path: Path) -> nbformat.NotebookNode:
    solution_nb = nbformat.read(solution_path, as_version=nbformat.NO_CONVERT)
    starter_nb = copy.deepcopy(solution_nb)

    for cell in starter_nb.cells:
        if cell.cell_type != "code":
            continue
        cell.source = strip_elided_source(cell.source)
        cell.outputs = []
        cell.execution_count = None

    return starter_nb


def paired_paths(solution_path: Path) -> Path:
    if "solution" not in solution_path.parts:
        raise ValueError(f"Expected 'solution' in path: {solution_path}")
    day_dir = solution_path.parent.parent
    starter_dir = day_dir / "notebook"
    starter_dir.mkdir(parents=True, exist_ok=True)
    return starter_dir / solution_path.name.replace("_solution", "_notebook")


def generate_all(root: Path) -> None:
    for solution_path in sorted(root.glob("days/day*/solution/*_solution.ipynb")):
        starter_nb = build_starter_notebook(solution_path)
        starter_path = paired_paths(solution_path)
        nbformat.write(starter_nb, starter_path)


def check_only(root: Path) -> int:
    exit_code = 0
    for solution_path in sorted(root.glob("days/day*/solution/*_solution.ipynb")):
        starter_nb = build_starter_notebook(solution_path)
        starter_path = paired_paths(solution_path)
        expected_text = nbformat.writes(starter_nb)
        if not starter_path.exists():
            print(f"Starter notebook missing: {starter_path}", file=sys.stderr)
            exit_code = 1
            continue
        actual_nb = nbformat.read(starter_path, as_version=nbformat.NO_CONVERT)
        actual_text = nbformat.writes(actual_nb)
        if actual_text != expected_text:
            print(f"Starter notebook out of date: {starter_path}", file=sys.stderr)
            diff = difflib.unified_diff(
                actual_text.splitlines(),
                expected_text.splitlines(),
                fromfile=str(starter_path),
                tofile=str(starter_path) + " (expected)",
                lineterm="",
            )
            for line in diff:
                print(line, file=sys.stderr)
            exit_code = 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if starters are stale")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent

    if args.check:
        return check_only(repo_root)

    generate_all(repo_root)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
