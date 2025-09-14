TensorBoard MCP Server
======================

This directory provides a lightweight Python Model Context Protocol (MCP) server exposing common TensorBoard inspection tools. It is inspired by GeorgePearse/tensorboard-mcp and built with the Python `mcp` library's `FastMCP` helpers.

What it provides
----------------
- `list_runs(base_dir)`: discover run directories containing TF event files
- `list_tags(log_dir)`: list available tags grouped by type (scalars, images, histograms, audio)
- `get_latest_scalar(log_dir, tag)`: latest scalar value/step with timestamp
- `get_scalar_series(log_dir, tag, max_points=200)`: downsampled scalar time series data
- `get_multi_scalar_series(log_dir, tags, max_points=200)`: batch fetch for many tags
- `compare_runs(log_dirs, tag, align_by='step'|'wall_time', max_points=200)`: compare a metric across runs
- `get_eval_summary(log_dir)`: aggregate latest `eval/*` scalars and best `eval/loss`
- `best_checkpoint(log_dir, metric_tag='eval/loss', mode='min'|'max')`: pick best step and map to checkpoint
- `summarize_run(log_dir)`: tag counts and latest scalar values for all metrics

## Status
✅ **Tested and Working** - All functions operational as of 2025-09-14
- Successfully discovers 22+ training runs in project
- Retrieves metrics like train/loss, eval/action_mae, learning rates, etc.
- Handles downsampling for large time series
- Fixed bug in `summarize_run` for proper tag counting
  
New in 1.0.1
- Added simple in‑process cache for EventAccumulator to reduce reload overhead.
- Added multi‑tag series, cross‑run comparison, eval summary, and best‑checkpoint tools.

## Installation

### Option 1: Install as Package (Recommended)
```bash
# Development install (editable)
pip install -e .mcp/tensorboard-mcp/

# Regular install
pip install .mcp/tensorboard-mcp/

# Upgrade an existing install to latest from this repo
pip install -U .mcp/tensorboard-mcp/

# If you need to force a rebuild
pip install --force-reinstall --no-deps .mcp/tensorboard-mcp/
```

### Option 2: Direct Usage
Run the module directly (no extra wrapper script):
```bash
python -m tb_mcp_server
```

### Requirements
- Python 3.10+
- `mcp` (Model Context Protocol Python library)
- `tensorboard` or `tensorflow` (for the event accumulator)

Install dependencies:
```bash
pip install mcp tensorboard
```


## Usage

### After Package Installation
Once installed as a package, you can run the server in multiple ways:

```bash
# Using console script (after pip install)
tensorboard-mcp-server

# Using module (after pip install)
python -m tb_mcp_server
```

### Updating the installed package
- Editable install (`-e`): no reinstall needed; just restart your MCP client so it reloads the server.
- Non‑editable install: reinstall to pick up changes, then restart your MCP client.
```bash
pip uninstall -y tensorboard-mcp-server
pip install -U .mcp/tensorboard-mcp/

# Verify version and entrypoints
pip show tensorboard-mcp-server | rg Version
which tensorboard-mcp-server
python -c "import tb_mcp_server as m; print(m.__file__)"
```

### Claude Code Configuration
Add to your `config.toml`:

**If installed as package:**
```toml
[mcp_servers.tensorboard]
command = "tensorboard-mcp-server"
```

**If using module directly:**
```toml
[mcp_servers.tensorboard]
command = "python"
args = ["-m", "tb_mcp_server"]
```

### Other MCP Client Configuration
For other MCP-compatible clients, use command array:
```json
["tensorboard-mcp-server"]
```
or
```json
["python", "-m", "tb_mcp_server"]
```

## Tool reference and examples

- `get_multi_scalar_series(log_dir, tags, max_points=200)`
  - Example: fetch train loss, grad norm, and LR together
  - Input:
    - `log_dir`: `tmp/vla/<run>/runs/<tb-run-dir>`
    - `tags`: `["train/loss", "train/grad_norm", "train/learning_rate"]`
  - Returns: `{ series: { tag: [{step, wall_time, value}, ...] }, missing: [] }`

- `compare_runs(log_dirs, tag, align_by='step', max_points=200)`
  - Example: compare `eval/loss` across two runs
  - Input: `log_dirs`: list of TB run dirs
  - Returns: per‑run `{ points: [...], latest: {step, value} }`

- `get_eval_summary(log_dir)`
  - Returns latest values for all `eval/*` scalars, plus `best_eval_loss`, `best_eval_step`, and inferred `best_checkpoint` if a matching `checkpoint-<step>` dir exists next to the run dir.

- `best_checkpoint(log_dir, metric_tag='eval/loss', mode='min')`
  - Returns best metric value, step, and checkpoint path if present.

Notes
- Series endpoints apply uniform downsampling when `max_points` is set.
- The server caches EventAccumulator instances and only reloads when event file mtimes change.
