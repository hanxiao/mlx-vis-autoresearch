# mlx-vis-autoresearch

Autonomous LLM-driven optimization of [mlx-vis](https://github.com/hanxiao/mlx-vis) NNDescent on Apple Silicon.

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for MLX GPU workloads.

## How it works

1. Drop `program.md` into the mlx-vis repo
2. Point Claude Code at it and let it run autonomously
3. It creates a branch, establishes baseline, then loops: modify code, benchmark, keep or revert
4. Results logged to `results.tsv`, progress visualized with `plot_progress.py`

## Quick start

```bash
# Copy program.md into your mlx-vis checkout
cp program.md /path/to/mlx-vis/

# Start Claude Code
cd /path/to/mlx-vis
claude --permission-mode bypassPermissions --print --output-format stream-json \
  'Read program.md and follow it exactly. NEVER STOP.'

# Plot progress (auto-installs deps via uv)
uv run plot_progress.py results.tsv progress.png
```

## Files

- `program.md` - Instructions for the LLM researcher. Defines benchmark, quality gates (recall@15), experiment loop protocol, and optimization ideas.
- `plot_progress.py` - Standalone plotting script (uv inline deps). Reads `results.tsv`, produces a Karpathy-style progress chart.

## Results

NNDescent on Fashion-MNIST 70K x 784, M3 Ultra:

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Time | 11.2s | 2.4s | 4.7x faster |
| VRAM | 2307 MB | 2696 MB | +17% |
| Recall@15 | 0.917 | 0.907 | -1.0% |

Key optimizations found autonomously:
1. Remove reverse-of-reverse candidates (1.5x)
2. Increase chunk size to reduce `mx.eval` calls (1.7x)
3. Adaptive candidate count scaling with convergence (2.5x)
4. Skip reverse candidate computation when nearly converged (2.95x)
