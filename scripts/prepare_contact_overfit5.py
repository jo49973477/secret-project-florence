"""Create a new five-episode subset: SOURCE TARGET. Source stays unchanged."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    src = args.source.resolve()
    dst = args.target.resolve()
    if dst.exists():
        raise SystemExit("Target already exists; choose a new isolated subset path")
    if src == dst or src in dst.parents:
        raise SystemExit("Subset must be outside the source dataset")
    (dst / "meta").mkdir(parents=True)
    src_meta = src / "meta"
    dst_meta = dst / "meta"

    selected_ids = list(range(5))

    with (src_meta / "episodes.jsonl").open("r", encoding="utf-8") as f:
        episodes = [json.loads(line) for line in f if line.strip()]

    by_id = {int(ep["episode_index"]): ep for ep in episodes}
    missing = [idx for idx in selected_ids if idx not in by_id]
    if missing:
        raise SystemExit(f"[ERROR] Missing episode metadata for IDs: {missing}")

    selected = [by_id[idx] for idx in selected_ids]

    # Copy every metadata item except files that depend on the full dataset.
    skip = {"info.json", "episodes.jsonl", "stats.json", "relative_stats.json"}
    for item in src_meta.iterdir():
        if item.name in skip:
            continue
        target = dst_meta / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)

    with (dst_meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for ep in selected:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    with (src_meta / "info.json").open("r", encoding="utf-8") as f:
        info = json.load(f)

    total_frames = sum(int(ep["length"]) for ep in selected)
    chunks_size = int(info.get("chunks_size", 1000))

    info["total_episodes"] = 5
    info["total_frames"] = total_frames
    info["splits"] = {"train": "0:5"}
    info["total_chunks"] = math.ceil(5 / chunks_size)

    # Keep total_tasks/tasks.jsonl unchanged because parquet task_index values
    # should not be remapped. Only update total_videos when it can be inferred.
    video_features = [
        value
        for value in info.get("features", {}).values()
        if isinstance(value, dict) and value.get("dtype") == "video"
    ]
    if video_features:
        info["total_videos"] = 5 * len(video_features)

    with (dst_meta / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
        f.write("\n")

    # Preserve non-directory dataset-root sidecars, if any.
    for item in src.iterdir():
        if item.name == "meta" or item.is_dir():
            continue
        target = dst / item.name
        if not target.exists():
            target.symlink_to(item.resolve())

    print(f"Selected episodes: {selected_ids}")
    print(f"Selected frames:   {total_frames}")
    # Preserve all episode-specific artifacts, including tactile MP4 and NPZ.

    for source in src.rglob("episode_*"):
        if not source.is_file() or "meta" in source.relative_to(src).parts:
            continue
        match = re.match(r"episode_(\d{6})\.", source.name)
        if match and int(match[1]) in selected_ids:
            target = dst / source.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source.resolve())
    if len(list((dst / "data").rglob("*.parquet"))) != 5:
        raise SystemExit("Expected five parquet episode files; inspect the new subset")


if __name__ == "__main__":
    main()
