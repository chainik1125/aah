# MCP Server Spec: RunPod Figure Pipeline

## Purpose

An MCP server that gives Claude Code tools to autonomously run the full
compute → plot → copy-to-TeX pipeline for paper figures on RunPod GPU
instances. Designed to run unattended via `claude -p`.

---

## Tools

### 1. `remote_exec`

Run a shell command on the remote instance via SSH.

| Parameter  | Type   | Default | Description                              |
|------------|--------|---------|------------------------------------------|
| `command`  | string | —       | Shell command to execute                  |
| `timeout`  | int    | `3600`  | Timeout in seconds                       |
| `cwd`      | string | `"~"`   | Working directory on the remote           |

**Returns:** `{ exit_code, stdout, stderr }`

### 2. `sync_to_remote`

Rsync the project to the instance (mirrors `submit_cluster_job.sh` excludes).

| Parameter   | Type   | Default          | Description               |
|-------------|--------|------------------|---------------------------|
| `local_dir` | string | project root     | Local directory to sync   |
| `remote_dir`| string | `"~/aah"`        | Remote destination        |

**Returns:** `{ files_transferred, bytes_transferred }`

### 3. `sync_from_remote`

Rsync results back from the instance to local machine.

| Parameter    | Type     | Default | Description                            |
|--------------|----------|---------|----------------------------------------|
| `remote_dir` | string   | —       | Remote source directory                |
| `local_dir`  | string   | —       | Local destination directory             |
| `patterns`   | string[] | `["*.pkl", "*.html", "*.csv", "*.svg", "*.pdf"]` | File patterns to include |

**Returns:** `{ files_transferred, bytes_transferred }`

### 4. `run_computation`

High-level tool: run a `cluster_runner.py` command on the remote.
Wraps `remote_exec` with project-specific setup (venv, install, cd).

| Parameter    | Type     | Default       | Description                         |
|--------------|----------|---------------|-------------------------------------|
| `command`    | string   | —             | cluster_runner command name          |
| `yaml_config`| string   | `null`        | YAML config file (relative path)    |
| `args`       | object   | `{}`          | CLI arguments as key-value pairs    |

**Returns:** `{ exit_code, stdout, stderr, output_files }`

### 5. `make_plots`

Run `make_paper_plots.py` locally to regenerate figures from pickle data.

| Parameter    | Type     | Default | Description                          |
|--------------|----------|---------|--------------------------------------|
| `figures`    | string[] | `["all"]` | Which figures to generate: `"hub_comparison"`, `"v_convergence"`, `"fixed_supercluster"`, or `"all"` |

**Returns:** `{ generated_files: string[] }`

---

## Configuration

Stored in `mcp/config.yaml` (git-ignored):

```yaml
remote:
  host: "69.30.85.136"
  port: 22112
  user: "root"
  key_path: "~/.ssh/id_ed25519"

paths:
  project_root: "/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/new_code/aah"
  remote_project_root: "~/aah"
  tex_images: "/Users/dmitrymanning-coe/Documents/Research/Barry Bradlyn/Moire/K_blocking/TeX/paper/AAH_AAHK_Paper/images"

rsync:
  excludes:
    - ".venv/"
    - ".git/"
    - "large_files/"
    - "__pycache__/"
    - "*.pyc"
    - "logs/"
```

**Instance specs:** 96-core CPU, 503GB RAM, Python 3.12.3 (RunPod team persistent instance).

---

## Typical Workflow

When Claude Code receives "run the figure pipeline", it would:

1. **`sync_to_remote`** → push code to the instance
2. **`remote_exec`** → install: `pip install -e .`
3. **`run_computation`** for each figure that needs new data:
   - `compressibility_with_int_cluster_sizes` (Fig 2 appendix)
   - (Fig 4/5 appendix — to be added)
4. **`sync_from_remote`** → pull `.pkl` files back to `large_files/`
5. **`make_plots`** → regenerate all PDFs locally

---

## Implementation Plan

```
mcp/
├── SPEC.md            ← this file
├── config.yaml        ← user config (git-ignored)
├── config.example.yaml
├── server.py          ← MCP server entry point
├── tools/
│   ├── __init__.py
│   ├── remote.py      ← remote_exec, sync_to_remote, sync_from_remote
│   ├── compute.py     ← run_computation (high-level wrapper)
│   └── plots.py       ← make_plots (local plotting)
└── requirements.txt   ← paramiko, pyyaml
```

---

## Registration

Add to `~/.claude/claude_desktop_config.json` (or project `.mcp.json`):

```json
{
  "mcpServers": {
    "runpod-figures": {
      "command": "python",
      "args": ["-m", "aah_code.cluster_model.mcp.server"],
      "cwd": "<project_root>/aah"
    }
  }
}
```

---

## Appendix Figure Requirements

The key goal is to write an appendix section corresponding to each of Fig 2,
Fig 4, Fig 5 in the main text. For each of these figures we will need
different kinds of plots.

### Appendix for Fig 2: Hubbard comparison (interaction separations)

**1a. Raw energy plot (LOCAL ONLY — no new data needed)**

A raw energy version of the relative energy plot. Call the same plotting func
(`compare_filling_with_int_cluster_sizes`) but set `plot_relative_error=False`.
Uses the existing data file:
`large_files/plots/filling_int_cluster_comparison_L48_chi32_20260206_113516.pkl`

**1b. Compressibility plot (NEEDS COMPUTE)**

Run func: `plots.compare_compressibility_with_int_cluster_sizes`
CLI: `cluster_runner.py` → `run_compressibility_with_int_cluster_sizes`

Sweep over mu_0, calculate compressibilities for the same interaction
separation schemes as Fig 2. Use U_values = [0, 1, 2, 5, 10, 30].

Layout must be rearranged from the existing appendix version:
- **Columns** = N_c (matching Fig 2 horizontal layout)
- **Rows** = different U values (going down the vertical)

This corresponds to `axes=('U', 'Nc')` in the existing function signature.

Use the same `int_sep_ratios_by_Nc` as the original Fig 2 data.

### Appendix for Fig 4: V-convergence

*(User spec pending — to be added)*

### Appendix for Fig 5: Fixed supercluster

*(User spec pending — to be added)*

---

## Implementation Notes (Context for Agent)

### Existing compressibility infrastructure

- `plots.compare_compressibility_with_int_cluster_sizes()` (line 3614 of plots.py)
  already supports the `axes=('U', 'Nc')` parameter to control row/column layout.
  It generates Plotly interactive figures + optional matplotlib PDF/SVG via `save_pdf=True`.
- The function sweeps `mu_0` over `[-mu_range_factor * U, mu_range_factor * U]`
  with a minimum half-range of `mu_min_range` (default 2.0).
- It accepts `results` parameter to reload from a pickle file (same as other funcs).
- There is a CLI wrapper in `cluster_runner.py` (`run_compressibility_with_int_cluster_sizes`).

### Raw energy plots

- The `plot_relative_error` parameter exists on `compare_filling_with_int_cluster_sizes`
  and is passed through from `cluster_runner.py`. Setting it to `False` plots raw
  energies instead of relative error. This can be done locally by reloading from
  the existing pickle via `results=<path>`.
- However, `make_paper_plots.py` currently only has `plot_hub_comparison()` which
  hardcodes relative error. A new function (or a `raw=True` option) will be
  needed in `make_paper_plots.py` to produce publication-quality raw energy figures.

### Paper-quality plotting gap

- `make_paper_plots.py` does NOT yet have functions for:
  - Raw energy version of hub comparison
  - Compressibility figures (n vs mu_0)
  - Any appendix figures
- These will need to be added, following the same style conventions
  (`setup_style()`, YlOrRd gradient, L-shaped spines, etc.)

---

## Resolved Questions

1. **Raw energy y-axis** — Plot the actual energies from each scheme (what
   `plot_relative_error=False` already produces in the data-generating funcs).
   Not absolute error — just the raw E values overlaid.

2. **Compressibility y-axis** — It's ν(μ₀) (filling vs chemical potential),
   not dν/dμ. So "compressibility plot" is a slight misnomer — it's the
   filling curve from which compressibility can be read off.

3. **Appendix for Fig 4 & Fig 5** — Deferred. Will be specified once the
   pipeline is verified working on the simpler Fig 2 case.

## Open Questions

1. **Appendix for Fig 4 & Fig 5** — Spec pending, will be added after Fig 2
   pipeline is verified working.
2. **Multiprocessing** — Add `multiprocessing` or `joblib` parallelism to
   `compare_compressibility_with_int_cluster_sizes` to exploit the 96 cores.
   Each (U, Nc, int_sep) point is independent.
