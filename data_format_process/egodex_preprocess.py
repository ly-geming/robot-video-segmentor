#!/usr/bin/env python3
"""Convert EgoDex flat task folders to video2tasks dataset layout.

Target layout:
  <output_root>/<subset_name>/<sample_id>/Frame_0.mp4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


def find_task_dirs(egodex_root: Path) -> List[Path]:
    """Return directories that directly contain mp4 files, sorted by relative path."""
    task_dirs: List[Path] = []
    for p in egodex_root.rglob("*"):
        if not p.is_dir():
            continue
        if any(child.is_file() and child.suffix.lower() == ".mp4" for child in p.iterdir()):
            task_dirs.append(p)
    task_dirs.sort(key=lambda x: str(x.relative_to(egodex_root)))
    return task_dirs


def ensure_unique_subset_dir(data_root: Path, base_name: str) -> Path:
    """Create unique subset directory name; append _2, _3 ... when needed."""
    candidate = data_root / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    idx = 2
    while True:
        candidate = data_root / f"{base_name}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        idx += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EgoDex data for video2tasks")
    parser.add_argument("--egodex-root", required=True, help="Path to EgoDex root folder")
    parser.add_argument("--output-data-root", required=True, help="Path to video2tasks data root")
    parser.add_argument("--task", type=int, required=True, help="How many task folders to take")
    parser.add_argument(
        "--num",
        type=int,
        required=True,
        help="How many videos per task to take (capped by 100 and available count)",
    )
    args = parser.parse_args()

    egodex_root = Path(args.egodex_root).expanduser().resolve()
    output_data_root = Path(args.output_data_root).expanduser().resolve()

    if not egodex_root.is_dir():
        raise NotADirectoryError(f"EgoDex root not found: {egodex_root}")
    output_data_root.mkdir(parents=True, exist_ok=True)

    task_dirs = find_task_dirs(egodex_root)
    if not task_dirs:
        raise RuntimeError(f"No task directories with mp4 files found under: {egodex_root}")

    selected_task_dirs = task_dirs[: max(0, args.task)]
    if not selected_task_dirs:
        raise RuntimeError("No task selected. Check --task value.")

    print(f"[Info] Total discovered task dirs: {len(task_dirs)}")
    print(f"[Info] Selected task dirs: {len(selected_task_dirs)}")

    for task_idx, task_dir in enumerate(selected_task_dirs, start=1):
        # Keep only raw video files in this task folder.
        mp4s = sorted([p for p in task_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])
        take_n = min(max(0, args.num), 100, len(mp4s))
        if take_n == 0:
            print(f"[Skip] {task_dir} has no mp4 to copy")
            continue

        subset_base = task_dir.name
        subset_dir = ensure_unique_subset_dir(output_data_root, subset_base)
        selected_mp4s = mp4s[:take_n]
        print(
            f"[Task {task_idx}] {task_dir.relative_to(egodex_root)} -> subset={subset_dir.name}, "
            f"videos={take_n}/{len(mp4s)}"
        )

        for sample_id, src_mp4 in enumerate(selected_mp4s):
            sample_dir = subset_dir / str(sample_id)
            sample_dir.mkdir(parents=True, exist_ok=True)
            dst_mp4 = sample_dir / "Frame_0.mp4"
            shutil.copy2(src_mp4, dst_mp4)

    print(f"[Done] Organized data written under: {output_data_root}")


if __name__ == "__main__":
    main()
