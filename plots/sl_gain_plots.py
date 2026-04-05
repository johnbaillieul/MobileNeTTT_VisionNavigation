import matplotlib.pyplot as plt
import numpy as np

# Data for all corridor widths
gain = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
max_lateral_deviation_1_0 = np.array([0.079, 0.1189, 0.1322, 0.162, 0.22])
max_lateral_deviation_1_5 = np.array([0.0628, 0.0666, 0.0738, 0.086, 0.1016])
max_lateral_deviation_2_0 = np.array([0.0494, 0.0592, 0.068, 0.1012, 0.1128])

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# Create grid: 2 rows, 3 columns for top, 1 full-width for bottom
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3, wspace=0.3)

# ===== First plot: Width = 1.0 m =====
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(gain, max_lateral_deviation_1_0, 'o-', linewidth=2, markersize=8, 
        color='#2E86AB', markerfacecolor='#A23B72', markeredgewidth=2, 
        markeredgecolor='#2E86AB')
ax1.set_xlabel('Gain', fontsize=12, fontweight='bold')
ax1.set_ylabel('Max Lateral Deviation (m)', fontsize=12, fontweight='bold')
ax1.set_title('Width = 1.0 m', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.tick_params(axis='both', labelsize=11)

# ===== Second plot: Width = 1.5 m =====
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(gain, max_lateral_deviation_1_5, 'o-', linewidth=2, markersize=8, 
        color='#E63946', markerfacecolor='#F77F00', markeredgewidth=2, 
        markeredgecolor='#E63946')
ax2.set_xlabel('Gain', fontsize=12, fontweight='bold')
ax2.set_ylabel('Max Lateral Deviation (m)', fontsize=12, fontweight='bold')
ax2.set_title('Width = 1.5 m', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(axis='both', labelsize=11)

# ===== Third plot: Width = 2.0 m =====
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(gain, max_lateral_deviation_2_0, 'o-', linewidth=2, markersize=8, 
        color='#06A77D', markerfacecolor='#F4D35E', markeredgewidth=2, 
        markeredgecolor='#06A77D')
ax3.set_xlabel('Gain', fontsize=12, fontweight='bold')
ax3.set_ylabel('Max Lateral Deviation (m)', fontsize=12, fontweight='bold')
ax3.set_title('Width = 2.0 m', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.tick_params(axis='both', labelsize=11)

# ===== Fourth plot: Combined comparison (spans full bottom) =====
ax4 = fig.add_subplot(gs[1, :])
ax4.plot(gain, max_lateral_deviation_1_0, 'o-', linewidth=2.5, markersize=9, 
        label='Width = 1.0 m', color='#2E86AB', markerfacecolor='#A23B72', 
        markeredgewidth=2, markeredgecolor='#2E86AB')
ax4.plot(gain, max_lateral_deviation_1_5, 's-', linewidth=2.5, markersize=9, 
        label='Width = 1.5 m', color='#E63946', markerfacecolor='#F77F00', 
        markeredgewidth=2, markeredgecolor='#E63946')
ax4.plot(gain, max_lateral_deviation_2_0, '^-', linewidth=2.5, markersize=9, 
        label='Width = 2.0 m', color='#06A77D', markerfacecolor='#F4D35E', 
        markeredgewidth=2, markeredgecolor='#06A77D')
ax4.set_xlabel('Gain', fontsize=12, fontweight='bold')
ax4.set_ylabel('Max Lateral Deviation (m)', fontsize=12, fontweight='bold')
ax4.set_title('Max Lateral Deviation vs Gain - Corridor Width Comparison', 
             fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.tick_params(axis='both', labelsize=11)
ax4.legend(fontsize=11, loc='upper left', framealpha=0.9)

# Save as high-resolution figure (uncomment to save)
# plt.savefig('corridor_width_comparison_all.png', dpi=300, bbox_inches='tight')

plt.show()
