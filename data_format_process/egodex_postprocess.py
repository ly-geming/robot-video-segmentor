#!/usr/bin/env python3
"""Merge EgoDex-style `data/` videos with `runs/.../samples/` annotations into `processed_Data/`."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _video2tasks_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _copy_tree_merge(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name == "__pycache__":
            continue
        target = dst / p.name
        if p.is_dir():
            _copy_tree_merge(p, target)
        else:
            shutil.copy2(p, target, follow_symlinks=True)


def main() -> int:
    root = _video2tasks_root()
    parser = argparse.ArgumentParser(
        description=(
            "For each data/<subset>/<sample_id>/ (videos), copy contents together with "
            "runs/<subset>/<run_id>/samples/<sample_id>/ (annotations) into "
            "<output>/<subset>/<sample_id>/."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=root / "data",
        help="Prepared video root (subset/sample_id/Frame_0.mp4 etc.)",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=root / "runs",
        help="Runs root containing <subset>/<run_id>/samples/",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="egodex_10x100_qwen35",
        help="Run id folder name under each subset (e.g. egodex_10x100_qwen35)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "processed_Data",
        help="Output root: processed_Data/<subset>/<sample_id>/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only, do not write files",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.resolve()
    runs_root: Path = args.runs_root.resolve()
    out_root: Path = args.output.resolve()
    run_id: str = args.run_id

    if not data_root.is_dir():
        print(f"[Err] data-root not found: {data_root}", file=sys.stderr)
        return 1

    n_out = 0
    n_skip_no_runs = 0

    for subset_dir in sorted(data_root.iterdir(), key=lambda p: p.name):
        if not subset_dir.is_dir() or subset_dir.name.startswith("."):
            continue
        subset = subset_dir.name
        run_samples = runs_root / subset / run_id / "samples"

        for sample_dir in sorted(subset_dir.iterdir(), key=lambda p: p.name):
            if not sample_dir.is_dir():
                continue
            sid = sample_dir.name
            if not sid.isdigit():
                continue

            ann_dir = run_samples / sid
            dst = out_root / subset / sid

            if args.dry_run:
                print(f"would merge: {sample_dir} + {ann_dir} -> {dst}")
                n_out += 1
                continue

            dst.mkdir(parents=True, exist_ok=True)
            _copy_tree_merge(sample_dir, dst)
            if ann_dir.is_dir():
                _copy_tree_merge(ann_dir, dst)
            else:
                n_skip_no_runs += 1
                print(f"[Warn] no annotations dir, video only: {subset}/{sid} -> missing {ann_dir}")

            n_out += 1

    print(
        f"[Done] merged {n_out} sample folders into {out_root} "
        f"(run_id={run_id}; missing annotation dirs: {n_skip_no_runs})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
