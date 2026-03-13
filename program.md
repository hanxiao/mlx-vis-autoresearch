# mlx-vis autoresearch

This is an experiment to have the LLM do its own research on optimizing NNDescent in mlx-vis, a pure-MLX library for dimensionality reduction and GPU-accelerated visualization on Apple Silicon.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `mlx_vis/nndescent.py` — **THE file you modify.** This is the NNDescent implementation. Everything is fair game: distance computation, candidate generation, update loops, memory layout, RP-tree initialization.
   - `mlx_vis/*.py` — read the other files for context (they call NNDescent), but your modifications should focus on `nndescent.py`.
4. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**: Confirm setup looks good. Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single Apple Silicon Mac (M3 Ultra, 512GB unified memory). You benchmark NNDescent directly:

```bash
cd ~/Documents/mlx-vis && uv run python -c "
import numpy as np, time
from sklearn.datasets import fetch_openml
fm = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='liac-arff')
X = fm.data.astype(np.float32) / 255.0
from mlx_vis.nndescent import NNDescent
# warmup
nn = NNDescent(X, n_neighbors=15)
# timed runs
times = []
for i in range(3):
    t0 = time.time()
    nn = NNDescent(X, n_neighbors=15)
    t = time.time() - t0
    times.append(t)
    print(f'  run {i}: {t:.2f}s')
print(f'NNDescent: {np.mean(times):.2f} +/- {np.std(times):.2f}s (n=3)')
" > run.log 2>&1
```

**What you CAN do:**
- Modify `mlx_vis/nndescent.py` — this is where all the NNDescent logic lives. Everything is fair game: distance computation, RP-tree construction, candidate generation, neighbor update loops, deduplication, memory layout, chunking strategy.

**What you CANNOT do:**
- Change the NNDescent API (constructor signature, return types). Other files depend on it.
- Install new packages or add dependencies. Pure MLX + numpy only.
- Sacrifice KNN quality for speed. The neighbor graph must be equally accurate. To verify: compare recall@15 against a brute-force KNN on a subset (e.g. 10K points). If recall drops more than 1%, discard the change.

**The goal: minimize NNDescent wall-clock time AND peak VRAM usage while maintaining KNN quality (recall@15).**

NNDescent is the bottleneck for all 6 embedding methods (UMAP, t-SNE, PaCMAP, TriMap, DREAMS, CNE). Speeding it up speeds up everything.

**Two metrics to track:**
1. **Time**: wall-clock seconds for `NNDescent(X, n_neighbors=15)` on Fashion-MNIST 70K x 784
2. **VRAM**: peak memory usage during the call. Measure with:
```python
import mlx.core as mx
mx.metal.reset_peak_memory()
nn = NNDescent(X, n_neighbors=15)
peak_mb = mx.metal.get_peak_memory() / 1024**2
```

**Quality gate**: After each change, verify recall hasn't degraded:
```python
from sklearn.neighbors import NearestNeighbors
subset = X[:10000]
nn_sub = NNDescent(subset, n_neighbors=15)
brute = NearestNeighbors(n_neighbors=15).fit(subset).kneighbors(subset)[1]
recall = np.mean([len(set(a) & set(b)) / 15 for a, b in zip(nn_sub.neighbor_indices, brute)])
print(f'recall@15: {recall:.4f}')
```
Baseline recall must be established first. Subsequent changes must stay within 1% of baseline.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing code and getting equal results is a great outcome.

**The first run**: Always establish the baseline first (time, VRAM, recall).

## Logging results

Log to `results.tsv` (tab-separated). Header and 7 columns:

```
commit	time_s	baseline_s	speedup	vram_mb	recall	description
```

1. git commit hash (short, 7 chars)
2. time in seconds (mean of 3 runs)
3. baseline time in seconds
4. speedup factor (e.g. 1.15x) — use 0.00x for crashes
5. peak VRAM in MB
6. recall@15 on 10K subset
7. short text description

Example:

```
commit	time_s	baseline_s	speedup	vram_mb	recall	description
a1b2c3d	6.36	6.36	1.00x	2048	0.965	baseline
b2c3d4e	5.80	6.36	1.10x	2048	0.964	vectorize candidate update loop
c3d4e5f	5.80	6.36	1.10x	1536	0.963	reduce chunk size in _gather_dists
d4e5f6g	0.00	6.36	0.00x	-	-	crash: reshape mismatch
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. **Review history first**: `cat results.tsv` to see ALL previous experiments (kept and discarded). This is critical after context compaction — your in-context memory of past attempts is lost, but results.tsv persists. Do NOT re-try strategies that already failed. Look for patterns in what worked vs what didn't.
2. Look at the git state: the current branch/commit we're on
3. **Propose and reason**: Before editing code, write a brief hypothesis: what you're changing, why it might help, and what the expected effect is. Check results.tsv to confirm this direction hasn't been tried.
4. Modify `mlx_vis/nndescent.py` with the experimental idea.
5. git commit
6. Run the NNDescent benchmark (redirect output to run.log, do NOT flood your context)
7. Read out timing, VRAM, and recall results from run.log
8. If crashed, check `tail -n 50 run.log` for the stack trace. Fix if trivial, skip if fundamental.
9. **Recall stability check**: For changes that affect convergence (delta, thresholds, iteration count), run recall measurement at least 10 times and check the MINIMUM, not the mean. A change that passes 9/10 times but fails 1/10 is not acceptable.
10. Record results in results.tsv (do NOT commit results.tsv, leave untracked). Include failure reason in the description for discarded experiments — this is your long-term memory.
11. If time or VRAM improved AND recall is within 1% of baseline: keep the commit
12. If no improvement or recall degraded: `git reset --hard HEAD~1`

**Timeout**: Each benchmark run should take under 2 minutes. If it exceeds 5 minutes, kill and revert.

**Avoiding local optima**: If you've had 5+ consecutive discards, try one of these escape strategies:
- **Combine near-misses**: Two changes that individually didn't help might work together.
- **Radical restructure**: Instead of tweaking parameters, change the algorithmic approach (e.g., different candidate generation, different distance computation strategy).
- **Profile first**: Run with verbose=True or add timing instrumentation to find the actual bottleneck before proposing a fix. Don't guess.
- **Read the literature**: Re-read NNDescent paper (Dong et al. 2011), study PyNNDescent, look at FAISS or Annoy for tricks.

**Key optimization areas for NNDescent:**
- `_gather_dists`: the distance computation is called every iteration. Chunking, compiled kernels, memory layout all matter.
- `_rp_tree_init`: RP-tree construction is currently numpy-heavy. Can parts move to MLX?
- The main update loop: Python-level iteration over candidates. Can it be vectorized?
- `_dedup_sorted`: deduplication of neighbor candidates. Already partially optimized.
- Reduce `mx.eval()` calls — batch lazy ops, let the graph grow before evaluating.
- `mx.compile` on hot functions (but scatter ops limit fusion).
- Eliminate numpy<->MLX round-trips in the hot path.
- Consider whether the number of RP-trees or NNDescent iterations can be adaptively reduced without quality loss.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask if you should continue. The human might be asleep. You are autonomous. If you run out of ideas, re-read the NNDescent paper (Dong et al. 2011), study PyNNDescent's implementation for tricks, try combining previous near-misses. The loop runs until interrupted.
