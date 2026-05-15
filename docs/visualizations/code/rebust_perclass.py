import matplotlib.pyplot as plt
import numpy as np

labels = [
    "CHOL",
    "COAD",
    "ESCA",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "PAAD",
    "READ",
    "SKCM",
    "STAD",
    "UVM",
]
acc = [
    0.5000,
    0.4615,
    0.6667,
    0.5833,
    0.7500,
    0.6000,
    0.6667,
    0.6667,
    0.3333,
    0.7143,
    0.3846,
    0.3333,
]

rare = {"CHOL", "ESCA", "MESO", "PAAD", "READ", "UVM"}  # 6 rarest

sorted_data = sorted(zip(acc, labels), reverse=True)
acc_s, labels_s = zip(*sorted_data)

# Color: orange for rare, steel blue for common
colors = ["#e67e22" if l in rare else "#2e86c1" for l in labels_s]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y_pos = np.arange(len(labels_s))
bars = ax.barh(y_pos, acc_s, height=0.62, color=colors, edgecolor="none", zorder=3)

ax.xaxis.grid(True, color="#dddddd", linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

# Value labels
for bar, a, l in zip(bars, acc_s, labels_s):
    w = bar.get_width()
    ax.text(
        w + 0.008,
        bar.get_y() + bar.get_height() / 2,
        f"{a:.4f}",
        va="center",
        ha="left",
        fontsize=9.5,
        color="#333333",
        fontfamily="monospace",
    )

# Vertical line at mean accuracy
mean_acc = np.mean(acc_s)
ax.axvline(
    mean_acc, color="#888888", linewidth=1.2, linestyle="--", zorder=4, alpha=0.8
)
ax.text(
    mean_acc + 0.005,
    -0.7,
    f"mean = {mean_acc:.3f}",
    fontsize=8.5,
    color="#888888",
    va="top",
)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_s, fontsize=11, color="#222222", fontweight="bold")
ax.set_xlabel("Accuracy", fontsize=11, color="#444444", labelpad=10)
ax.tick_params(colors="#666666", length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xlim(0, 0.95)

# Legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#2e86c1", label="Common class"),
    Patch(facecolor="#e67e22", label="Rare class"),
]
ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=9.5)

plt.tight_layout(pad=2)
plt.savefig("rebust_perclass.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.show()
