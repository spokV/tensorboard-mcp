TensorBoard MCP Server
======================

This directory provides a lightweight Python Model Context Protocol (MCP) server exposing common TensorBoard inspection tools. It is inspired by GeorgePearse/tensorboard-mcp and built with the Python `mcp` library's `FastMCP` helpers.

What it provides
----------------
- `list_runs(base_dir)`: discover run directories containing TF event files
- `list_tags(log_dir)`: list available tags grouped by type (scalars, images, histograms, audio)
- `get_latest_scalar(log_dir, tag)`: latest scalar value/step with timestamp
- `get_scalar_series(log_dir, tag, max_points=200)`: downsampled scalar time series data
- `summarize_run(log_dir)`: tag counts and latest scalar values for all metrics

## Status
✅ **Tested and Working** - All functions operational as of 2024-09-14
- Successfully discovers 22+ training runs in project
- Retrieves metrics like train/loss, eval/action_mae, learning rates, etc.
- Handles downsampling for large time series
- Fixed bug in `summarize_run` for proper tag counting

Requirements
------------
- Python 3.10+
- `mcp` (Model Context Protocol Python library)
- `tensorboard` or `tensorflow` (for the event accumulator)


Run the server
--------------
From the repo root, run the script by path (not `-m`):
```
python .mcp/tb_mcp.py
```
This starts the MCP server process that an MCP-compatible client (e.g. Claude Desktop, Cursor, or other MCP-aware tools) can connect to.

Codex Configuration
--------------------------
Add to your `config.toml`:
```toml
[mcp_servers.tensorboard]
command = "python"
args = [".mcp/tb_mcp.py"]
```
