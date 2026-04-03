# CLAUDE.md — MCP Figure Pipeline

You are operating the AAH/AAHK paper figure pipeline. This MCP server gives
you tools to run computations on a remote 96-core CPU instance, sync results,
and generate publication-quality plots locally.

## Available MCP Tools

| Tool | Purpose |
|------|---------|
| `remote_exec` | Run any shell command on the remote instance |
| `sync_to_remote` | Rsync project code to the remote (excludes .venv, .git, large_files) |
| `sync_from_remote` | Pull result files (pkl, html, pdf, etc.) from remote to local |
| `run_computation` | Run a `cluster_runner.py` command on the remote (handles venv activation) |
| `setup_remote` | Create venv and `pip install -e .` on the remote |
| `make_plots` | Run `make_paper_plots.py` locally to generate publication figures |
| `replot_from_pickle` | Reload existing pickle data and replot with different options |

## Standard Workflow

1. **`sync_to_remote`** — push latest code
2. **`setup_remote`** — ensure deps are installed (only needed after dependency changes)
3. **`run_computation`** — run the needed calculation (see commands below)
4. **`sync_from_remote`** — pull `.pkl` results back to local `large_files/`
5. **`make_plots`** or **`replot_from_pickle`** — generate publication figures locally

## Computation Commands

These are the `command` values for `run_computation`. Each maps to a function
in `plots.py` via `cluster_runner.py`:

| Command | What it computes | Typical use |
|---------|-----------------|-------------|
| `filling_cluster_sizes` | Energy vs V for different Nc (maximal sep) | Fig 4 data |
| `filling_with_int_cluster_sizes` | Energy vs U for different int seps per Nc | Fig 2 data |
| `fixed_supercluster` | Energy vs V at fixed supercluster size | Fig 5 data |
| `compressibility_with_int_cluster_sizes` | ν(μ₀) for different int seps per Nc | Fig 2 appendix |
| `compressibility_cluster_sizes` | ν(μ₀) for different Nc | Fig 4 appendix |
| `compressibility_fixed_supercluster` | ν(μ₀) at fixed supercluster size | Fig 5 appendix |

## Key Parameters

All computation commands accept these via the `args` dict:

| Parameter | Type | Typical values | Notes |
|-----------|------|---------------|-------|
| `L` | int | 48, 120 | System size |
| `chi` | int | 32, 64 | DMRG bond dimension |
| `U_values` | list[float] | [0,1,2,5,10,30] | Hubbard U |
| `V_values` | list[float] | [0,1,2,3,4,5] | Nearest-neighbor V |
| `cluster_sizes` | list[int] | [2,3,4,6,8] | Cluster sizes Nc |
| `v_sep_ratio` | tuple(int,int) | (1,2) | β modulation ratio |
| `solver_method` | str | "sparse_ED" | "dense_ED" or "sparse_ED" |
| `include_finite_dmrg` | bool | True | Include DMRG reference |
| `plot_relative_error` | bool | True | Relative error vs raw energy |
| `axes` | tuple(str,str) | ("U","Nc") | Row/col layout for compressibility |
| `results` | str | path to .pkl | Reload from existing results |

## Project File Layout

```
aah/
├── aah_code/cluster_model/
│   ├── plots.py                  ← Core computation + rough plotting functions
│   ├── cluster_runner.py         ← CLI wrappers for plots.py functions
│   ├── large_files/
│   │   ├── plots/                ← Output pickle and HTML files
│   │   ├── partials/             ← Merged array job results
│   │   └── paper_plots/
│   │       ├── make_paper_plots.py  ← Publication-quality matplotlib figures
│   │       └── README.md            ← Detailed docs on each figure
│   └── mcp/                      ← This MCP server
└── aah_code/currently_used_data/
    ├── figures.csv               ← Source of truth: all figures, status, data/plot funcs
    ├── figure_registry.py        ← Loads CSV, generates used_data.md, programmatic access
    └── used_data.md              ← Rendered markdown table (auto-generated from CSV)
```

## Figure Tracking

### Principles

The paper has three main-text figures (Fig 2, Fig 4, Fig 5). Each main-text
figure shows **relative error of energy**. For each main-text figure, there
is a corresponding appendix section with supporting plots. The general
pattern for appendix plots is:

1. **Absolute energy** — the raw energies (not relative error) from the same
   data as the main-text figure. No new computation needed.
2. **Filling relative error** — relative error of the filling ν (cluster vs
   DMRG), analogous to the energy relative error in the main text.
3. **Filling values** — the filling curves ν(μ₀) themselves, showing how
   cluster ED compares to DMRG across chemical potential.
4. **Small-parameter zoom** — repeat the absolute energy and filling plots
   zoomed into a regime where the interaction is small (e.g. U ∈ [0,1] for
   Fig 2), with a finer parameter grid (e.g. 10 values). This requires
   **new computation**.

So each main-text figure typically generates ~6 total plots (1 main + 5
appendix), of which:
- The main-text plot is already done.
- 2 appendix plots reuse existing data (just need plot code).
- 1 appendix plot has data + plot code but may need review.
- 2 appendix plots need new computation with a finer parameter grid.

### Source of truth

All figure status is tracked in a CSV:

**`aah_code/currently_used_data/figures.csv`**

Each row has: key, parent_figure, paper_ref, description, status, data_files,
parameters, run_func, plot_func, notes. The status values are:

| Status | Meaning |
|--------|---------|
| `done` | Data exists, plot code exists, PDF generated and reviewed |
| `needs_review` | PDF exists but hasn't been visually checked |
| `needs_plot` | Data exists, but no publication plot function yet |
| `needs_data` | Computation hasn't been run yet |
| `blocked` | Blocked on something external |

To see a rendered summary, open `aah_code/currently_used_data/used_data.md`.
To regenerate it after editing the CSV:
```bash
python -m aah_code.currently_used_data.figure_registry
```

To query programmatically:
```python
from aah_code.currently_used_data.figure_registry import load_figures, Status
figures = load_figures()
todo = {k: v for k, v in figures.items() if v["status"] != Status.DONE}
```

---

## Appendix Plot Specs

### Fig 2 Appendix: Hubbard comparison (V=0, varying interaction separations)

The main-text Fig 2 shows relative error of energy vs U/t for V=0, comparing
different interaction separation schemes at each cluster size Nc. The appendix
expands on this with 5 additional plots:

**App Fig 2a — Absolute energy** (`app_fig2_abs_energy`)
- What: Raw energy E vs U/t (not relative error), same layout as Fig 2.
- Data: Same pickle as Fig 2 (`filling_int_cluster_comparison_L48_chi32_*.pkl`).
- Compute: None needed (LOCAL ONLY).
- Plot: Needs new function in `make_paper_plots.py`. Same structure as
  `plot_hub_comparison` but y-axis is E instead of relative error.
- Status: `needs_plot`

**App Fig 2b — Filling relative error** (`app_fig2_filling_relerr`)
- What: Relative error of filling |ν_cl − ν_DMRG|/ν_DMRG vs U/t.
- Data: Compressibility pickle (`compressibility_combined_L48_chi32_*.pkl`).
- Compute: None needed (data exists).
- Plot: Needs new function. Layout: rows=U, cols=Nc. Compare cluster filling
  to DMRG filling at each (U, Nc, int_sep) point.
- Status: `needs_plot`

**App Fig 2c — Filling values ν(μ₀)** (`app_fig2_filling_vals`)
- What: Filling curves ν vs μ₀ for each (U, Nc) panel. Lines = different
  interaction separations (YlOrRd gradient, maximal = black). DMRG = grey
  dashed.
- Data: Same compressibility pickle as App Fig 2b.
- Compute: None needed.
- Plot: `make_paper_plots.plot_hub_compressibility` — EXISTS, PDF generated.
- Status: `needs_review`

**App Fig 2d — Small U zoom: absolute energy** (`app_fig2_smallU_abs_energy`)
- What: Absolute energy E vs U/t, zoomed to U ∈ [0, 1] with 10 evenly
  spaced U values. Same layout as App Fig 2a but finer grid.
- Data: NEEDS NEW COMPUTATION.
- Compute: `run_computation` with command `filling_with_int_cluster_sizes`,
  args: `U_values=linspace(0,1,10)`, same `int_sep_ratios_by_Nc` as Fig 2,
  `L=48`, `chi=32`.
- Plot: Needs new function (or reuse App Fig 2a function with different data).
- Status: `needs_data`

**App Fig 2e — Small U zoom: filling values** (`app_fig2_smallU_filling`)
- What: Filling curves ν(μ₀) for U ∈ [0, 1] with 10 U values.
- Data: NEEDS NEW COMPUTATION.
- Compute: `run_computation` with command
  `compressibility_with_int_cluster_sizes`, args: `U_values=linspace(0,1,10)`,
  same `int_sep_ratios_by_Nc` as Fig 2, `L=48`, `chi=32`.
- Plot: Needs new function (or reuse App Fig 2c function with different data).
- Status: `needs_data`

### Fig 4 Appendix — spec pending.

### Fig 5 Appendix — spec pending.

---

## Skills

These are multi-step workflows that compose the low-level MCP tools above.
When asked to perform one of these, follow the steps in order and report
progress at each stage.

### Skill 1: Offload to RunPod

Use this when a figure row in `figures.csv` has `status=needs_data` and the
computation is too heavy to run locally (large L, many parameter points, or
DMRG reference needed).

**Steps:**

1. **Read the figure registry.** Load `figures.csv` and identify which rows
   need computation (`status=needs_data`). Note the `run_func`, `parameters`,
   and `key` for each.

2. **Sync code to remote.**
   ```
   sync_to_remote()
   ```

3. **Ensure remote environment is set up** (only needed once per session or
   after dependency changes).
   ```
   setup_remote()
   ```

4. **Run the computation remotely.** Use `run_computation` with the
   appropriate `command` and `args` from the figure row. For example:
   ```
   run_computation(
       command="filling_with_int_cluster_sizes",
       args={"L": 48, "chi": 32, "U_values": [0, 0.1, 0.2, ..., 1.0],
             "int_sep_ratios_by_Nc": "2:1,2|1,4|1,8|1,12|1,24;3:1,3|1,6|1,12|1,24;..."},
       timeout=7200
   )
   ```
   Monitor stdout for progress. If the computation is long, increase timeout.

5. **Sync results back.**
   ```
   sync_from_remote(
       remote_dir="~/aah/large_files/plots",
       local_dir="<project_root>/aah_code/cluster_model/large_files/plots"
   )
   ```

6. **Generate publication plots locally.**
   ```
   make_plots()
   ```
   Or call a specific plot function if only one figure changed.

7. **Update the CSV.** Change `status` from `needs_data` to `needs_review`
   (or `needs_plot` if plot code doesn't exist yet). Fill in `data_files`
   with the path to the new pickle. Regenerate the markdown:
   ```bash
   python -m aah_code.currently_used_data.figure_registry
   ```

### Skill 2: Run Locally

Use this when the computation is fast enough to run on the local machine
(small L, few parameter points, or replotting from existing data).

**Steps:**

1. **Read the figure registry.** Identify which rows to process.

2. **Run the computation locally.** Either:
   - Call `cluster_runner.py` directly:
     ```bash
     python -m aah_code.cluster_model.cluster_runner filling_with_int_cluster_sizes \
       --L=48 --chi=32 --U_values=0,0.1,0.2,...,1.0 \
       --int_sep_ratios_by_Nc="2:1,2|1,4|1,8|1,12|1,24;..." \
       --save_data --plot_relative_error
     ```
   - Or use `replot_from_pickle` if only replotting existing data:
     ```
     replot_from_pickle(
         pickle_path="large_files/plots/filling_int_cluster_comparison_L48_*.pkl",
         plot_relative_error=False
     )
     ```

3. **Generate publication plots.**
   ```
   make_plots()
   ```

4. **Update the CSV.** Same as Skill 1 step 7.

### Skill 3: Review Plots

Use this after plots have been generated (rows with `status=needs_review`).

**Steps:**

1. **Read the figure registry.** Find all rows with `status=needs_review`.

2. **Open the generated PDFs.** The output path is in the `output_pdf`
   column. Check each plot for:
   - Correct data (right number of lines, correct labels).
   - Sensible axis ranges (no clipping, no excessive whitespace).
   - Style consistency with the main-text figures (YlOrRd gradient,
     L-shaped spines, Computer Modern font, panel labels).
   - Legend readability and placement.
   - Correct filling annotations (half/quarter labels on right side).

3. **Flag issues or approve.** If the plot looks correct, update `status`
   to `done` in the CSV. If there are issues, note them in the `notes`
   column and keep `status=needs_review`.

4. **Regenerate the markdown.**
   ```bash
   python -m aah_code.currently_used_data.figure_registry
   ```

### Skill 4: Paper Integration

Use this after a plot has been reviewed and approved (`status=done`).
This skill adds the figure to the TeX paper with appropriate surrounding text.

**Steps:**

1. **Copy the PDF to the TeX images directory.**
   ```bash
   cp <paper_plots>/<fig_name>.pdf \
      "<tex_images>/current/<tex_name>"
   ```
   The `tex_name` is specified in the CSV. The TeX images dir is:
   `K_blocking/TeX/paper/AAH_AAHK_Paper/images/`

2. **Add the figure to the TeX source.** Insert a `\begin{figure}` block in
   the appropriate appendix section of `main.tex`. Follow the existing
   figure style in the paper:
   ```latex
   \begin{figure}[htbp]
     \centering
     \includegraphics[width=\linewidth]{images/current/<tex_name>}
     \caption{<Caption describing what the figure shows, referring to
       relevant parameters (L, chi, beta, etc.) and the parent main-text
       figure.>}
     \label{fig:<label>}
   \end{figure}
   ```

3. **Write surrounding narrative text.** The appendix text should:
   - Reference the parent main-text figure ("As a complement to
     Fig.~\ref{fig:hub_comparison}, we show...").
   - Briefly explain what the appendix plot adds (raw energies, filling
     curves, zoomed regime).
   - Highlight any notable features visible in the data.
   - Keep it concise — 2-4 sentences per plot is typical.

4. **Verify TeX renders.** Compile the paper and check:
   ```bash
   cd "<tex_dir>" && latexmk -pdf main.tex
   ```
   Verify no missing figure warnings, correct placement, and that the
   caption renders properly.

5. **Update the CSV.** Confirm `status=done` and fill in `tex_name` if
   not already set. Regenerate the markdown.

---

## Parallelism (`n_jobs`)

`compare_filling_with_int_cluster_sizes` accepts an `n_jobs` parameter
(default 1 = serial). Set to `-1` for all cores, or a specific integer.
Uses `joblib.Parallel` with the `loky` backend (multiprocessing).

Both the cluster ED loop and the finite DMRG loop are parallelized. Each
`(Nc, int_sep, fill_mode, U)` task is independent.

### Local machine benchmarks (8-core Apple Silicon, L=48)

**Cluster ED** (64 tasks: Nc=4, 2 int seps, 16 U values, 2 fillings):

| n_jobs | Time | Speedup |
|--------|------|---------|
| 1 | 65s | 1.0x |
| 2 | 38s | 1.8x |
| 4 | 29s | **2.3x** |
| 8 | 27s | 2.5x |

**Finite DMRG** (6 tasks: 3 U values, 2 fillings):

| n_jobs | Time | Speedup |
|--------|------|---------|
| 1 | 221s | 1.0x |
| 2 | 136s | 1.6x |
| 4 | 109s | **2.0x** |

Speedup saturates around `n_jobs=4` locally due to memory bandwidth
contention (sparse eigensolves and MPS contractions are memory-bound).
The sweet spot is `n_jobs=4` on this machine. On RunPod (96 cores, higher
memory bandwidth), scaling should be significantly better.

### CLI usage

```bash
python -m aah_code.cluster_model.cluster_runner filling_with_int_cluster_sizes \
  --L=48 --chi=32 --U_values=0,0.5,1.0 \
  --int_sep_ratios_by_Nc="4:1,4|1,8" \
  --include_finite_dmrg --save_data --n_jobs=4
```

In YAML config: add `n_jobs: 4` (or `-1`).

Via MCP `run_computation`: add `"n_jobs": 4` to the `args` dict.

---

## Remote Instance Details

- **Host:** Persistent RunPod team CPU instance (96 cores, 503GB RAM)
- **Python:** 3.12.3
- **SSH:** `root@69.30.85.136 -p 22112` via ed25519 key
- **Project on remote:** `~/aah/`
- **Venv on remote:** `~/aah/.venv/`

## Important Notes

- **No GPU needed.** All computation uses CPU (numpy/scipy ED + TeNPy DMRG).
- **DMRG is stochastic.** Expect ~10⁻⁶ differences in DMRG energies between
  runs. Cluster (ED) energies are deterministic and should match exactly.
- **large_files/ is not synced** to the remote by default. Computation outputs
  go to `~/aah/large_files/plots/` on the remote — sync them back after.
- **make_paper_plots.py** currently only has functions for the main text
  figures (hub_comparison, v_convergence, fixed_supercluster). Appendix
  plotting functions need to be added following the same style conventions
  (see `paper_plots/README.md`).
- The `int_sep_ratios_by_Nc` for Fig 2 are:
  `{2: [(1,2),(1,4),(1,8),(1,12),(1,24)], 3: [(1,3),(1,6),(1,12),(1,24)], 4: [(1,4),(1,8),(1,12)], 6: [(1,6),(1,12),(1,24)], 8: [(1,8),(1,16),(1,24)]}`
