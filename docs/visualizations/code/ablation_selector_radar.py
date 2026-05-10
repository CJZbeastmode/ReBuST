import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

methods = ['ReBuST', 'EvoPS', 'SASHA', 'DA']
metrics = ['Acc', 'B-Acc', 'Macro-F1', 'F1', 'AUC']

means = np.array([
    [0.5780, 0.5550, 0.5549, 0.5810, 0.9158],  # ReBuST
    [0.2752, 0.2175, 0.1876, 0.2306, 0.7345],  # EvoPS
    [0.3303, 0.2575, 0.2343, 0.2894, 0.7824],  # SASHA
    [0.1193, 0.1217, 0.0913, 0.0860, 0.5126],  # DA
])

stds = np.array([
    [0.0475, 0.0631, 0.0598, 0.0488, 0.0143],
    [0.0436, 0.0391, 0.0358, 0.0443, 0.0273],
    [0.0461, 0.0401, 0.0398, 0.0470, 0.0268],
    [0.0311, 0.0362, 0.0296, 0.0284, 0.0312],
])

colors = ['#2e86c1', '#e74c3c', '#27ae60', '#8e44ad']

N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f9f9f9')

ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                   fontsize=7.5, color='#aaaaaa')
ax.set_ylim(0, 1)
ax.yaxis.grid(True, color='#dddddd', linewidth=0.8)
ax.xaxis.grid(True, color='#dddddd', linewidth=0.8)
ax.spines['polar'].set_visible(False)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12, color='#222222', fontweight='bold')

for i, (method, color) in enumerate(zip(methods, colors)):
    values  = means[i].tolist() + means[i][:1].tolist()
    std_vals = stds[i].tolist() + stds[i][:1].tolist()

    ax.plot(angles, values, color=color, linewidth=2.2, linestyle='solid', zorder=3)
    ax.fill(angles, values, color=color, alpha=0.08, zorder=2)

    upper = np.clip(np.array(values) + np.array(std_vals), 0, 1)
    lower = np.clip(np.array(values) - np.array(std_vals), 0, 1)
    ax.fill_between(angles, lower, upper, color=color, alpha=0.12, zorder=2)
    ax.scatter(angles[:-1], means[i], color=color, s=40, zorder=4)

legend_elements = [Line2D([0], [0], color=c, linewidth=2.2, label=m)
                   for m, c in zip(methods, colors)]
ax.legend(handles=legend_elements, loc='upper right',
          bbox_to_anchor=(1.28, 1.12), frameon=False, fontsize=10.5)

plt.tight_layout()
plt.savefig('ablation_selector_radar.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.show()