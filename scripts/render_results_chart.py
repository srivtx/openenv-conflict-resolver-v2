"""Render the static results bar chart for README + blog.

The numbers below are the documented holdout / adversarial scores from the
real Colab T4 run referenced in README.md and blog.md. Re-running the Colab
notebook will regenerate equivalent charts in ./assets/ via the plot cells
at the end of `notebooks/train_grpo_colab.ipynb`.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "assets")
os.makedirs(OUT_DIR, exist_ok=True)

scores = {
    "Untrained 3B\n(holdout, n=40)": 0.5454,
    "SFT\n(holdout, n=40)": 0.9877,
    "SFT + GRPO\n(holdout, n=40)": 0.9876,
    "SFT + GRPO\n(adversarial, n=10)": 0.9885,
}

labels = list(scores.keys())
values = list(scores.values())
colors = ["#FF5630", "#FFAB00", "#36B37E", "#0052CC"]

fig, ax = plt.subplots(figsize=(8.5, 4.6))
bars = ax.bar(labels, values, color=colors, width=0.6)
ax.set_ylim(0.0, 1.05)
ax.set_ylabel("avg env reward (0-1)")
ax.set_title("Conflict Resolver — holdout + adversarial scores by training stage")
ax.grid(axis="y", alpha=0.25)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.015,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.setp(ax.get_xticklabels(), fontsize=9)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "results.png")
fig.savefig(out_path, dpi=160)
print(f"wrote {out_path}")
