"""
Generate research-paper graphs for Experiment 1 and Experiment 3.

Usage:
    python make_graphs.py

Reads:
    results.csv             — output of experiment_1.py
    specificity_results.csv — output of experiment_3.py

Writes:
    graph_exp1.png
    graph_exp3.png

Requirements: pip install matplotlib numpy pandas
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

LEVEL_LABELS = {
    "L1_vague":       "L1\n(Vague)",
    "L2_rough":       "L2\n(Rough)",
    "L3_formatted":   "L3\n(Formatted)",
    "L4_constrained": "L4\n(Constrained)",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Run the corresponding experiment first."
        )
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def abbreviate(name: str, max_len: int = 14) -> str:
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


# ── Experiment 1: grouped bar chart ──────────────────────────────────────────

def graph_exp1(path: str = "results.csv", out: str = "graph_exp1.png"):
    rows = read_csv(path)

    tasks         = [abbreviate(r["task"]) for r in rows]
    score_without = [float(r["score_without"]) for r in rows]
    score_with    = [float(r["score_with"])    for r in rows]

    x     = np.arange(len(tasks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))

    bars_without = ax.bar(x - width / 2, score_without, width,
                          label="Without format instructions",
                          color="#b0c4de", edgecolor="black", linewidth=0.6)
    bars_with    = ax.bar(x + width / 2, score_with, width,
                          label="With format instructions",
                          color="#2c5f8a", edgecolor="black", linewidth=0.6)

    avg_without = np.mean(score_without)
    avg_with    = np.mean(score_with)
    ax.axhline(avg_without, color="#7a9dbb", linewidth=1.2, linestyle="--",
               label=f"Avg without ({avg_without:.3f})")
    ax.axhline(avg_with, color="#1a3d5c", linewidth=1.2, linestyle="--",
               label=f"Avg with ({avg_with:.3f})")

    ax.set_xlabel("Task")
    ax.set_ylabel("Consistency Score (TF-IDF cosine, 0–1)")
    ax.set_title("Experiment 1: Prompt Format Instructions vs. Output Consistency")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ── Experiment 3: line chart ──────────────────────────────────────────────────

def graph_exp3(path: str = "specificity_results.csv", out: str = "graph_exp3.png"):
    rows = read_csv(path)

    tasks  = list(dict.fromkeys(r["task"] for r in rows))
    levels = sorted(set(int(r["level"]) for r in rows))

    data: dict[str, list[float]] = {}
    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task]
        task_rows.sort(key=lambda r: int(r["level"]))
        data[task] = [float(r["mean_quality"]) for r in task_rows]

    averages = [
        np.mean([data[t][i] for t in tasks]) for i in range(len(levels))
    ]

    level_tick_labels = ["L1\n(Vague)", "L2\n(Rough)", "L3\n(Formatted)", "L4\n(Constrained)"]

    cmap   = plt.cm.get_cmap("tab10", len(tasks))
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for idx, task in enumerate(tasks):
        ax.plot(levels, data[task], marker="o", markersize=5,
                linewidth=1.4, color=cmap(idx), label=task, alpha=0.75)

    ax.plot(levels, averages, marker="D", markersize=6,
            linewidth=2.4, color="black", label="Average", zorder=5)

    ax.set_xlabel("Prompt Specificity Level")
    ax.set_ylabel("Mean Quality Score (LLM-as-judge, 1–10)")
    ax.set_title("Experiment 3: Prompt Specificity vs. Output Quality")
    ax.set_xticks(levels)
    ax.set_xticklabels(level_tick_labels)
    ax.set_ylim(0, 10.5)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    errors = []

    try:
        graph_exp1()
    except FileNotFoundError as e:
        errors.append(str(e))
        print(f"Skipping exp1 graph: {e}")

    try:
        graph_exp3()
    except FileNotFoundError as e:
        errors.append(str(e))
        print(f"Skipping exp3 graph: {e}")

    if errors:
        print("\nRun the missing experiments first, then re-run make_graphs.py.")
    else:
        print("\nDone. Graphs saved as graph_exp1.png and graph_exp3.png.")
