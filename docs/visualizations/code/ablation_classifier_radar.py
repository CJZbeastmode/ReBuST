import matplotlib.pyplot as plt
import numpy as np

methods = ['ST', 'ABMIL', 'CLAM', 'MAMBA']
metrics = ['Acc', 'B-Acc', 'Macro-F1', 'F1', 'AUC']

means = np.array([
    [0.5780, 0.5550, 0.5549, 0.5810, 0.9158],  # ST
    [0.0917, 0.1293, 0.0563, 0.0432, 0.6366],  # ABMIL
    [0.1101, 0.1676, 0.0856, 0.0813, 0.6254],  # CLAM
    [0.1560, 0.1748, 0.1271, 0.1411, 0.6825],  # MAMBA
])

stds = np.array([
    [0.0475, 0.0631, 0.0598, 0.0488, 0.0143],
    [0.0285, 0.0453, 0.0185, 0.0181, 0.0258],
    [0.0310, 0.0469, 0.0232, 0.0279, 0.0288],
    [0.0355, 0.0434, 0.0312, 0.0353, 0.0242],
])

colors = ['#2e86c1', '#e74c3c', '#27ae60', '#8e44ad']

N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f9f9f9')

# Gridlines styling
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                   fontsize=7.5, color='#aaaaaa')
ax.set_ylim(0, 1)
ax.yaxis.grid(True, color='#dddddd', linewidth=0.8)
ax.xaxis.grid(True, color='#dddddd', linewidth=0.8)
ax.spines['polar'].set_visible(False)

# Metric labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12, color='#222222', fontweight='bold')

# Plot each method
for i, (method, color) in enumerate(zip(methods, colors)):
    values = means[i].tolist() + means[i][:1].tolist()
    std_vals = stds[i].tolist() + stds[i][:1].tolist()

    ax.plot(angles, values, color=color, linewidth=2.2,
            linestyle='solid', zorder=3)
    ax.fill(angles, values, color=color, alpha=0.08, zorder=2)

    # Std shading
    upper = np.clip(np.array(values) + np.array(std_vals), 0, 1)
    lower = np.clip(np.array(values) - np.array(std_vals), 0, 1)
    ax.fill_between(angles, lower, upper, color=color, alpha=0.12, zorder=2)

    # Dots at each metric vertex
    ax.scatter(angles[:-1], means[i], color=color, s=40, zorder=4)

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=c, linewidth=2.2, label=m)
                   for m, c in zip(methods, colors)]
ax.legend(handles=legend_elements, loc='upper right',
          bbox_to_anchor=(1.28, 1.12), frameon=False, fontsize=10.5)

plt.tight_layout()
plt.savefig('ablation_classifier_radar.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()