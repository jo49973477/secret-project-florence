#!/usr/bin/env python
"""Summarize per-rank checkpoint trace JSONL without adding nested I/O counters twice."""

import argparse
import json
from pathlib import Path
import warnings


def summarize(directory):
    ranks = {}
    for path in sorted(Path(directory).glob("rank-*-pid-*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                warnings.warn(f"Ignoring incomplete trace record {path}:{number}")
                continue
            key = f"rank-{event['rank']}/pid-{event['pid']}"
            rank = ranks.setdefault(key, {"peak_kib": 0, "phases": {}})
            rank["peak_kib"] = max(rank["peak_kib"], event.get("peak_kib", 0))
            if event["event"] != "exit":
                continue
            phase = rank["phases"].setdefault(
                event["phase"],
                {"calls": 0, "elapsed_seconds": 0, "read_bytes_delta": 0, "rchar_delta": 0},
            )
            phase["calls"] += 1
            for field in ("elapsed_seconds", "read_bytes_delta", "rchar_delta"):
                phase[field] += event.get(field, 0)
    return ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--baseline")
    args = parser.parse_args()
    report = {"candidate": summarize(args.directory)}
    if args.baseline:
        report["baseline"] = summarize(args.baseline)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
