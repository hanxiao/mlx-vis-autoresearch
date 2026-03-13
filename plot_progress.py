#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas", "matplotlib", "numpy"]
# ///
"""Plot NNDescent autoresearch progress in Karpathy's analysis.ipynb style.

Usage:
    uv run plot_progress.py [results.tsv] [output.png]
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

tsv_path = sys.argv[1] if len(sys.argv) > 1 else "results.tsv"
out_path = sys.argv[2] if len(sys.argv) > 2 else "progress.png"

df = pd.read_csv(tsv_path, sep="\t")
df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
df["recall"] = pd.to_numeric(df["recall"], errors="coerce")
df["vram_mb"] = pd.to_numeric(df["vram_mb"], errors="coerce")


def get_status(row):
    desc = str(row["description"]).lower()
    if "baseline" in desc:
        return "KEEP"
    if "reverted" in desc or "crash" in desc:
        return "DISCARD"
    sp = str(row["speedup"]).replace("x", "")
    try:
        if float(sp) > 1.0:
            return "KEEP"
    except ValueError:
        pass
    return "DISCARD"


df["status"] = df.apply(get_status, axis=1)

n_total = len(df)
n_kept = len(df[df["status"] == "KEEP"])
baseline_time = df.iloc[0]["time_s"]

fig, ax = plt.subplots(figsize=(16, 8))

# For discarded experiments with no time data, place them at baseline level
# so they still appear on the chart as gray dots
df.loc[(df["status"] == "DISCARD") & (df["time_s"].isna() | (df["time_s"] <= 0)), "time_s"] = baseline_time
valid = df[df["time_s"] > 0].copy().reset_index(drop=True)

# Discarded: faint background dots
disc = valid[valid["status"] == "DISCARD"]
ax.scatter(
    disc.index, disc["time_s"],
    c="#cccccc", s=12, alpha=0.5, zorder=2, label="Discarded",
)

# Kept: prominent green dots
kept_v = valid[valid["status"] == "KEEP"]
ax.scatter(
    kept_v.index, kept_v["time_s"],
    c="#2ecc71", s=50, zorder=4, label="Kept",
    edgecolors="black", linewidths=0.5,
)

# Running minimum step line
kept_mask = valid["status"] == "KEEP"
kept_idx = valid.index[kept_mask]
kept_times = valid.loc[kept_mask, "time_s"]
running_min = kept_times.cummin()
ax.step(
    kept_idx, running_min, where="post", color="#27ae60",
    linewidth=2, alpha=0.7, zorder=3, label="Running best",
)

# Label each kept experiment
for idx, t in zip(kept_idx, kept_times):
    desc = str(valid.loc[idx, "description"]).strip()
    if len(desc) > 45:
        desc = desc[:42] + "..."
    ax.annotate(
        desc, (idx, t),
        textcoords="offset points", xytext=(6, 6), fontsize=8.0,
        color="#1a7a3a", alpha=0.9, rotation=30, ha="left", va="bottom",
    )

best = running_min.min()
ax.set_xlabel("Experiment #", fontsize=12)
ax.set_ylabel("NNDescent Time (seconds, lower is better)", fontsize=12)
ax.set_title(
    f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements",
    fontsize=14,
)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.2)

margin = (baseline_time - best) * 0.15
ax.set_ylim(best - margin, baseline_time + margin)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved {out_path}: {n_total} experiments, {n_kept} kept, "
      f"baseline {baseline_time:.2f}s -> best {best:.2f}s ({baseline_time / best:.1f}x)")
