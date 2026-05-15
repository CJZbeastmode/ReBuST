import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

classes = [
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
methods = ["ST", "ABMIL", "CLAM", "MAMBA"]
rare = {"CHOL", "ESCA", "MESO", "PAAD", "READ", "UVM"}

data = np.array(
    [
        [0.5000, 0.0000, 0.0000, 0.0000],  # CHOL
        [0.4615, 0.0000, 0.0000, 0.1538],  # COAD
        [0.6667, 0.3333, 0.1667, 0.1667],  # ESCA
        [0.5833, 0.0000, 0.0000, 0.0833],  # LIHC
        [0.7500, 0.0000, 0.1250, 0.3125],  # LUAD
        [0.6000, 0.0000, 0.0000, 0.0000],  # LUSC
        [0.6667, 0.3333, 0.6667, 0.6667],  # MESO
        [0.6667, 0.0000, 0.0000, 0.3333],  # PAAD
        [0.3333, 0.1667, 0.5000, 0.1667],  # READ
        [0.7143, 0.0000, 0.1429, 0.2143],  # SKCM
        [0.3846, 0.3846, 0.0769, 0.0000],  # STAD
        [0.3333, 0.3333, 0.3333, 0.0000],  # UVM
    ]
)

fig, ax = plt.subplots(figsize=(7, 8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1)

# Cell annotations
for i in range(len(classes)):
    for j in range(len(methods)):
        val = data[i, j]
        text_color = "white" if val > 0.55 else "#333333"
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=text_color,
            fontfamily="monospace",
        )

# Axes labels
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=11, fontweight="bold", color="#222222")
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(
    [f"* {c}" if c in rare else f"  {c}" for c in classes],
    fontsize=10,
    color="#222222",
    fontfamily="monospace",
)
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

# Separate rare vs non-rare rows with a horizontal line
# Rare classes: CHOL(0), ESCA(2), MESO(6), PAAD(7), READ(8), UVM(11)
# Draw a visual bracket on the left instead — highlight row backgrounds
for i, cls in enumerate(classes):
    if cls in rare:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (-0.5, i - 0.5),
                len(methods),
                1,
                boxstyle="square,pad=0",
                linewidth=0,
                facecolor="#fff3cd",
                alpha=0.3,
                zorder=0,
            )
        )

# Bold border around ST column
ax.add_patch(
    mpatches.FancyBboxPatch(
        (-0.5, -0.5),
        1,
        len(classes),
        boxstyle="square,pad=0",
        linewidth=2,
        edgecolor="#2e86c1",
        facecolor="none",
        zorder=5,
    )
)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Accuracy", fontsize=9.5, color="#444444")
cbar.ax.tick_params(labelsize=8.5, colors="#666666")
cbar.outline.set_visible(False)

# Legend
from matplotlib.lines import Line2D

legend_elements = [
    mpatches.Patch(facecolor="#fff3cd", edgecolor="none", label="Rare class (*)"),
    mpatches.Patch(
        facecolor="#dbeafe", edgecolor="#2e86c1", linewidth=1.5, label="ST (proposed)"
    ),
]
ax.legend(
    handles=legend_elements,
    loc="upper right",
    bbox_to_anchor=(1.42, 1.02),
    frameon=False,
    fontsize=9,
)

plt.tight_layout(pad=2)
plt.savefig(
    "ablation_classifier_heatmap.png", dpi=180, bbox_inches="tight", facecolor="white"
)
plt.show()
