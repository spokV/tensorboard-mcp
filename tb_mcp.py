from __future__ import annotations

import os
import glob
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("TensorBoard MCP Server")


def _load_event_accumulator(log_dir: str):
    """Lazy import and load EventAccumulator for a given log dir."""
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    # Lazy import to avoid heavy imports during discovery
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    acc = EventAccumulator(log_dir)
    acc.Reload()
    return acc


@mcp.tool()
async def list_runs(base_dir: str) -> Dict[str, Any]:
    """Discover TensorBoard runs under a base directory.

    Returns directories containing TF event files. Useful when you don't
    know the exact run path to query.
    """
    if not os.path.isdir(base_dir):
        return {"error": f"Base directory not found: {base_dir}"}

    # Find directories that contain event files
    event_paths: List[str] = []
    # Glob for common event file patterns
    for pattern in (
        "**/events.out.tfevents.*",
        "**/tfevents.*",
    ):
        event_paths.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))

    run_dirs = sorted(list({os.path.dirname(p) for p in event_paths}))
    return {"base_dir": base_dir, "runs": run_dirs}


@mcp.tool()
async def list_tags(log_dir: str) -> Dict[str, Any]:
    """List available tags in a run directory, grouped by type."""
    try:
        acc = _load_event_accumulator(log_dir)
        tags = acc.Tags()
        # Ensure JSON-serializable
        return {
            "log_dir": log_dir,
            "tags": {
                "scalars": list(tags.get("scalars", [])),
                "images": list(tags.get("images", [])),
                "histo": list(tags.get("histograms", [])),
                "audio": list(tags.get("audio", [])),
            },
        }
    except Exception as e:
        return {"error": str(e), "log_dir": log_dir}


@mcp.tool()
async def get_latest_scalar(log_dir: str, tag: str) -> Dict[str, Any]:
    """Get the latest scalar value and step for a given tag."""
    try:
        acc = _load_event_accumulator(log_dir)
        if tag not in acc.Tags().get("scalars", []):
            return {"error": f"Tag not found: {tag}", "log_dir": log_dir, "tag": tag}
        events = acc.Scalars(tag)
        if not events:
            return {"error": f"No data for tag: {tag}", "log_dir": log_dir, "tag": tag}
        latest = events[-1]
        return {
            "log_dir": log_dir,
            "tag": tag,
            "value": float(latest.value),
            "step": int(latest.step),
            "wall_time": float(latest.wall_time),
        }
    except Exception as e:
        return {"error": str(e), "log_dir": log_dir, "tag": tag}


@mcp.tool()
async def get_scalar_series(
    log_dir: str, tag: str, max_points: Optional[int] = 200
) -> Dict[str, Any]:
    """Get a downsampled scalar time series for a given tag.

    Parameters
    - log_dir: path to the TensorBoard run directory
    - tag: scalar tag name
    - max_points: optionally limit the number of returned points (downsample)
    """
    try:
        acc = _load_event_accumulator(log_dir)
        if tag not in acc.Tags().get("scalars", []):
            return {"error": f"Tag not found: {tag}", "log_dir": log_dir, "tag": tag}
        series = acc.Scalars(tag)
        points = [
            {"step": int(e.step), "wall_time": float(e.wall_time), "value": float(e.value)}
            for e in series
        ]
        if max_points and len(points) > max_points:
            # Uniform downsample
            stride = max(1, len(points) // max_points)
            points = points[::stride]
            # Ensure last point included
            if points[-1] != {
                "step": int(series[-1].step),
                "wall_time": float(series[-1].wall_time),
                "value": float(series[-1].value),
            }:
                points.append(
                    {
                        "step": int(series[-1].step),
                        "wall_time": float(series[-1].wall_time),
                        "value": float(series[-1].value),
                    }
                )

        return {"log_dir": log_dir, "tag": tag, "points": points}
    except Exception as e:
        return {"error": str(e), "log_dir": log_dir, "tag": tag}


@mcp.tool()
async def summarize_run(log_dir: str) -> Dict[str, Any]:
    """Provide a quick summary of a TB run: tag counts and latest steps."""
    try:
        acc = _load_event_accumulator(log_dir)
        tags = acc.Tags()
        scalars = tags.get("scalars", [])
        latest_by_tag: Dict[str, Dict[str, Any]] = {}
        for t in scalars:
            ev = acc.Scalars(t)
            if ev:
                latest_by_tag[t] = {"step": int(ev[-1].step), "value": float(ev[-1].value)}
        return {
            "log_dir": log_dir,
            "counts": {k: len(v) if hasattr(v, '__len__') else 0 for k, v in tags.items()},
            "latest": latest_by_tag,
        }
    except Exception as e:
        return {"error": str(e), "log_dir": log_dir}


if __name__ == "__main__":
    mcp.run()
