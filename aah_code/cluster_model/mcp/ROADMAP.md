# MCP Autonomous Pipeline Roadmap

**Date:** 2026-04-03
**Goal:** A claude.ai web instance that autonomously runs computations on RunPod and generates publication figures.

---

## Phase 1: Demonstrate Efficient RunPod Execution

### 1.1 Serial CPU benchmark (local Mac vs RunPod)
- Run Nc=4, L=48, chi=32, V=0, U=[0,1,5,10] on both machines
- Measure cluster ED and finite DMRG separately
- **Success:** RunPod single-core within 2x of Mac

### 1.2 Add multiprocessing parallelism
- Add `n_jobs` parameter to all 6 computation functions in `plots.py`
- Add `--n_jobs` to `cluster_runner.py` CLI
- Pattern: `multiprocessing.Pool` with module-level worker functions
- Control thread oversubscription: `OMP_NUM_THREADS=2` per worker

### 1.3 Parallel scaling benchmark on RunPod
- Test n_jobs = 1, 4, 16, 32, 48, 96
- Determine sweet spot for 96-core instance

### 1.4 GPU acceleration (conditional)
- Only if CPU path can't match local Mac performance
- Profile first, then decide approach (NOT monkey-patching scipy)
- Consider: batched dense eigensolves, CuPy sparse, or custom kernels
- Key question: cluster Hamiltonian dimensions determine best approach

### 1.5 Real figure computation
- Run `app_fig5_sc_compressibility` (status: needs_data) as validation

---

## Phase 2: Cloud-Based Agent Monitoring

### Architecture
```
claude.ai (web) --MCP connector--> RunPod (HTTP/SSE MCP server)
```

### 2.1 Convert MCP server to HTTP/SSE transport
- FastMCP supports `transport="sse"` — add env var toggle
- Keep stdio mode for local Claude Code usage

### 2.2 Local execution mode for on-RunPod operation
- Add `LocalConnection` class (subprocess instead of SSH)
- Config: `remote.mode: "local"` vs `"ssh"`

### 2.3 Progress monitoring
- Computation functions write `progress.json` (total/completed/ETA)
- New `check_progress` MCP tool

### 2.4 Deploy on RunPod
- Start MCP server on exposed port
- Register as MCP connector at claude.ai/settings/connectors

### 2.5 Async computation support
- `run_computation_async` — nohup + PID return
- Pair with `check_progress` for polling

---

## Phase 3: Fully Autonomous Figure Generation

### 3.1 Agent prompt for claude.ai
- Reads `figures.csv` to identify pending work
- Uses MCP tools end-to-end: compute, monitor, plot, update tracking

### 3.2 End-to-end workflow
1. `check_progress()` — anything running?
2. Read `figures.csv` — what needs doing?
3. `run_computation_async(command, args)` — launch computation
4. Poll `check_progress()` periodically
5. `make_plots()` — generate PDFs when done
6. Update `figures.csv`, report results

### 3.3 Scheduled triggers (optional)
- Daily check for pending figures via remote trigger

---

## Current Status

- [x] MCP server implemented (7 tools, stdio transport)
- [x] Committed and pushed to `pbc-fintie-bethe`
- [ ] Phase 1.1: Serial benchmark
- [ ] Phase 1.2: Parallelism
- [ ] Phase 1.3: Parallel benchmark
- [ ] Phase 2: Cloud agent architecture
- [ ] Phase 3: Autonomous operation
