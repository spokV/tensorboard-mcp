from __future__ import annotations

import os
import glob
import time
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("TensorBoard MCP Server")


# Simple cache to avoid reloading event files on every request
_ACC_CACHE: Dict[str, Tuple["EventAccumulator", float]] = {}


def _latest_event_mtime(log_dir: str) -> float:
    """Get latest modification time of event files under a run directory."""
    latest = 0.0
    for pattern in ("events.out.tfevents.*", "tfevents.*"):
        for p in glob.glob(os.path.join(log_dir, pattern)):
            try:
                latest = max(latest, os.path.getmtime(p))
            except OSError:
                continue
    # Fallback to dir mtime if no files matched
    return latest or os.path.getmtime(log_dir)


def _load_event_accumulator(log_dir: str):
    """Get a cached EventAccumulator for a given log dir, reloading as needed."""
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    # Lazy import to avoid heavy imports during discovery
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    latest_mtime = _latest_event_mtime(log_dir)
    cached = _ACC_CACHE.get(log_dir)
    if cached is None:
        acc = EventAccumulator(log_dir)
        acc.Reload()
        _ACC_CACHE[log_dir] = (acc, latest_mtime)
        return acc
    acc, cached_mtime = cached
    if latest_mtime > cached_mtime:
        acc.Reload()
        _ACC_CACHE[log_dir] = (acc, latest_mtime)
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
async def get_multi_scalar_series(
    log_dir: str, tags: List[str], max_points: Optional[int] = 200
) -> Dict[str, Any]:
    """Batch fetch scalar series for multiple tags to reduce round-trips."""
    try:
        acc = _load_event_accumulator(log_dir)
        available = set(acc.Tags().get("scalars", []))
        result: Dict[str, Any] = {"log_dir": log_dir, "series": {}, "missing": []}
        for tag in tags:
            if tag not in available:
                result["missing"].append(tag)
                continue
            series = acc.Scalars(tag)
            points = [
                {"step": int(e.step), "wall_time": float(e.wall_time), "value": float(e.value)}
                for e in series
            ]
            if max_points and len(points) > max_points:
                stride = max(1, len(points) // max_points)
                points = points[::stride]
                if points:
                    last = {
                        "step": int(series[-1].step),
                        "wall_time": float(series[-1].wall_time),
                        "value": float(series[-1].value),
                    }
                    if points[-1] != last:
                        points.append(last)
            result["series"][tag] = points
        return result
    except Exception as e:
        return {"error": str(e), "log_dir": log_dir}


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


@mcp.tool()
async def compare_runs(
    log_dirs: List[str], tag: str, align_by: str = "step", max_points: Optional[int] = 200
) -> Dict[str, Any]:
    """Compare a scalar tag across multiple runs.

    - align_by: "step" or "wall_time" (metadata only; series are provided per run)
    """
    out: Dict[str, Any] = {"tag": tag, "align_by": align_by, "runs": {}}
    for ld in log_dirs:
        try:
            acc = _load_event_accumulator(ld)
            if tag not in acc.Tags().get("scalars", []):
                out["runs"][ld] = {"error": "tag not found"}
                continue
            series = acc.Scalars(tag)
            pts = [
                {"step": int(e.step), "wall_time": float(e.wall_time), "value": float(e.value)}
                for e in series
            ]
            if max_points and len(pts) > max_points:
                stride = max(1, len(pts) // max_points)
                pts = pts[::stride]
                last = {
                    "step": int(series[-1].step),
                    "wall_time": float(series[-1].wall_time),
                    "value": float(series[-1].value),
                }
                if pts and pts[-1] != last:
                    pts.append(last)
            out["runs"][ld] = {
                "points": pts,
                "latest": {
                    "step": int(series[-1].step),
                    "wall_time": float(series[-1].wall_time),
                    "value": float(series[-1].value),
                }
            }
        except Exception as e:
            out["runs"][ld] = {"error": str(e)}
    return out


@mcp.tool()
async def get_eval_summary(log_dir: str) -> Dict[str, Any]:
    """Aggregate latest eval/* scalars into a single payload.

    Also includes best eval/loss (min) step if available.
    """
    try:
        acc = _load_event_accumulator(log_dir)
        tags = acc.Tags().get("scalars", [])
        eval_tags = [t for t in tags if t.startswith("eval/")]
        latest: Dict[str, Any] = {}
        best_loss = None
        best_step = None
        for t in eval_tags:
            ev = acc.Scalars(t)
            if not ev:
                continue
            last = ev[-1]
            latest[t] = {"step": int(last.step), "value": float(last.value)}
            if t == "eval/loss":
                # Find global min
                for x in ev:
                    if best_loss is None or x.value < best_loss:
                        best_loss = float(x.value)
                        best_step = int(x.step)
        # Try to map to a checkpoint path if present
        best_ckpt = None
        if best_step is not None:
            ckpt_path = os.path.join(os.path.dirname(log_dir), f"checkpoint-{best_step}")
            if os.path.isdir(ckpt_path):
                best_ckpt = ckpt_path
        return {
            "log_dir": log_dir,
            "latest": latest,
            "best_eval_loss": best_loss,
            "best_eval_step": best_step,
            "best_checkpoint": best_ckpt,
        }
    except Exception as e:
        return {"error": str(e), "log_dir": log_dir}


@mcp.tool()
async def best_checkpoint(
    log_dir: str,
    metric_tag: str = "eval/loss",
    mode: str = "min",
) -> Dict[str, Any]:
    """Return best checkpoint based on a scalar metric.

    - metric_tag: scalar tag to optimize (e.g., eval/loss or eval/action_success_rate_overall)
    - mode: 'min' or 'max'
    """
    try:
        acc = _load_event_accumulator(log_dir)
        tags = acc.Tags().get("scalars", [])
        if metric_tag not in tags:
            return {"error": f"Tag not found: {metric_tag}", "log_dir": log_dir}
        series = acc.Scalars(metric_tag)
        if not series:
            return {"error": f"No data for tag: {metric_tag}", "log_dir": log_dir}
        best_val = None
        best_step = None
        for e in series:
            v = float(e.value)
            if best_val is None or ((mode == "min" and v < best_val) or (mode == "max" and v > best_val)):
                best_val = v
                best_step = int(e.step)
        ckpt = None
        if best_step is not None:
            maybe = os.path.join(os.path.dirname(log_dir), f"checkpoint-{best_step}")
            if os.path.isdir(maybe):
                ckpt = maybe
        return {
            "log_dir": log_dir,
            "metric_tag": metric_tag,
            "mode": mode,
            "best_value": best_val,
            "best_step": best_step,
            "checkpoint": ckpt,
        }
    except Exception as e:
        return {"error": str(e), "log_dir": log_dir}


def main():
    """Main entry point for the TensorBoard MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
