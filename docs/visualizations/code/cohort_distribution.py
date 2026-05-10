import matplotlib.pyplot as plt
import numpy as np

labels = ['CHOL', 'COAD', 'ESCA', 'LIHC', 'LUAD', 'LUSC',
          'MESO', 'PAAD', 'READ', 'SKCM', 'STAD', 'UVM']
counts = [14, 132, 52, 107, 148, 143, 25, 52, 48, 133, 125, 23]
percents = [1.40, 13.17, 5.19, 10.68, 14.77, 14.27, 2.50, 5.19, 4.79, 13.27, 12.48, 2.30]

sorted_data = sorted(zip(counts, percents, labels), reverse=True)
counts_s, percents_s, labels_s = zip(*sorted_data)

# Rare threshold — classes with < 6% (ESCA, PAAD, READ, MESO, UVM, CHOL)
n_rare = 6

cmap = plt.colormaps.get_cmap('RdYlBu_r').resampled(len(labels))
colors = [cmap(i / (len(labels) - 1)) for i in range(len(labels))]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

y_pos = np.arange(len(labels_s))

# --- Rare region shading (behind everything) ---
rare_y_min = len(labels_s) - n_rare - 0.5  # = 5.5
rare_y_max = len(labels_s) - 0.5           # = 11.5
ax.axhspan(rare_y_min, rare_y_max, color='#fff3cd', alpha=0.6, zorder=0)

# --- Dashed separator line ---
ax.axhline(y=rare_y_max, color='#e67e22', linewidth=1.2,
           linestyle='--', zorder=4, alpha=0.8)

bars = ax.barh(y_pos, percents_s, height=0.62, color=colors,
               edgecolor='none', zorder=3)

ax.xaxis.grid(True, color='#dddddd', linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

for bar, pct, cnt in zip(bars, percents_s, counts_s):
    w = bar.get_width()
    ax.text(w + 0.18, bar.get_y() + bar.get_height() / 2,
            f'{pct:.2f}%  (n={cnt})',
            va='center', ha='left', fontsize=9.5,
            color='#333333', fontfamily='monospace')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_s, fontsize=11, color='#222222', fontweight='bold')
ax.set_xlabel('Percentage of Cohort (%)', fontsize=11, color='#444444', labelpad=10)
ax.tick_params(colors='#666666', length=0)
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xlim(0, 24)

# --- Rare class bracket + label on the right ---
bracket_x = 21.8
ax.annotate('', xy=(bracket_x, rare_y_min + 0.1),
            xytext=(bracket_x, rare_y_max - 0.1),
            arrowprops=dict(arrowstyle=']-[', color='#e67e22',
                            lw=1.4, mutation_scale=6))
ax.text(bracket_x + 0.3, (rare_y_min + rare_y_max) / 2,
        'Rare\nclasses\n(21.4%)',
        va='center', ha='left', fontsize=8.5,
        color='#e67e22', fontweight='bold', linespacing=1.5)

ax.text(0.98, 0.02, 'Total N = 1,002', transform=ax.transAxes,
        fontsize=8.5, color='#999999', ha='right', va='bottom', style='italic')

plt.tight_layout(pad=2)
plt.savefig('cohort_distribution.png', dpi=180, bbox_inches='tight',
            facecolor='white')
plt.show()