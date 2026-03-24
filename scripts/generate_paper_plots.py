"""Generate plots for IntRob 2026 paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

SEEDS = [42, 43, 44]
OUT_DIR = Path("paper_figures")
OUT_DIR.mkdir(exist_ok=True)

# Colors
C_OURS = '#2166ac'
C_RAND = '#b2182b'
C_RETRAIN = '#4daf4a'
C_NOFILTER = '#ff7f00'
C_ORIG = '#777777'
C_FINETUNE = '#984ea3'


def load_metrics(path_template):
    all_data = []
    for seed in SEEDS:
        with open(path_template.format(seed=seed)) as f:
            all_data.append(json.load(f))
    return all_data


def extract_series(all_data, key):
    ref_steps = np.array([entry["step"] for entry in all_data[0]])
    n = len(ref_steps)
    vals = np.full((len(all_data), n), np.nan)
    for si, data in enumerate(all_data):
        for i in range(min(n, len(data))):
            vals[si, i] = data[i].get(key, np.nan)
    return ref_steps, np.nanmean(vals, axis=0), np.nanstd(vals, axis=0)


# Load all data
ours_data = load_metrics(
    "experiments/outputs/cartpole_left_only_trajectory_selective_seed{seed}/unlearning_metrics_trajectory_selective.json"
)
rand_data = load_metrics(
    "experiments/outputs/cartpole_left_only_random_forget_seed{seed}/unlearning_metrics_trajectory_selective.json"
)
nofilter_data = load_metrics(
    "experiments/outputs/cartpole_left_only_no_step_filter_seed{seed}/unlearning_metrics_trajectory_selective.json"
)
retrain_data = load_metrics(
    "experiments/outputs/cartpole_left_only_full_retrain_seed{seed}/retrain_metrics.json"
)


# ============================================================
# Figure 1: Forget-region fraction over unlearning steps
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))

steps_o, mean_o, std_o = extract_series(ours_data, "forget_region_fraction")
steps_n, mean_n, std_n = extract_series(nofilter_data, "forget_region_fraction")
steps_r, mean_r, std_r = extract_series(rand_data, "forget_region_fraction")

ax.plot(steps_o, mean_o, color=C_OURS, linewidth=1.8, label='Ours (full)', zorder=4)
ax.fill_between(steps_o, mean_o - std_o, mean_o + std_o, alpha=0.15, color=C_OURS, zorder=3)

ax.plot(steps_n, mean_n, color=C_NOFILTER, linewidth=1.8, label='No step filter', linestyle='-.', zorder=3)
ax.fill_between(steps_n, mean_n - std_n, mean_n + std_n, alpha=0.15, color=C_NOFILTER, zorder=2)

ax.plot(steps_r, mean_r, color=C_RAND, linewidth=1.8, label='Random forget', linestyle='--', zorder=2)
ax.fill_between(steps_r, mean_r - std_r, mean_r + std_r, alpha=0.12, color=C_RAND, zorder=1)

ax.axhline(y=0.365, color=C_ORIG, linestyle=':', linewidth=1, label='Original agent', zorder=1)

ax.set_xlabel('Unlearning Step')
ax.set_ylabel('Forget-Region Fraction')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2,
          framealpha=0.9, edgecolor='none', fontsize=7.5)
ax.set_ylim(-0.02, 0.85)

fig.savefig(OUT_DIR / "unlearning_curves.pdf")
fig.savefig(OUT_DIR / "unlearning_curves.png")
plt.close(fig)
print("Saved unlearning_curves")


# ============================================================
# Figure 2: Full retrain learning curve (2-panel)
# ============================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True,
                                gridspec_kw={'hspace': 0.08})

steps_rt, mean_frf, std_frf = extract_series(retrain_data, "forget_region_fraction")
_, mean_ret, std_ret = extract_series(retrain_data, "mean_return")

# Top: forget-region fraction
ax1.plot(steps_rt / 1000, mean_frf, color=C_RETRAIN, linewidth=1.8, label='Full retrain')
ax1.fill_between(steps_rt / 1000, mean_frf - std_frf, mean_frf + std_frf,
                  alpha=0.2, color=C_RETRAIN)
ax1.axhline(y=0.038, color=C_OURS, linestyle='--', linewidth=1, label='Ours (500 steps)')
ax1.axhline(y=0.365, color=C_ORIG, linestyle=':', linewidth=1, label='Original agent')
ax1.set_ylabel('Forget-Reg. Frac.')
ax1.set_ylim(-0.02, 0.5)
ax1.tick_params(labelbottom=False)

# Bottom: return (no separate legend — shared with top panel)
ax2.plot(steps_rt / 1000, mean_ret, color=C_RETRAIN, linewidth=1.8)
ax2.fill_between(steps_rt / 1000, mean_ret - std_ret, mean_ret + std_ret,
                  alpha=0.2, color=C_RETRAIN)
ax2.axhline(y=500, color=C_OURS, linestyle='--', linewidth=1)
ax2.set_xlabel('Training Step (×1000)')
ax2.set_ylabel('Mean Return')
ax2.set_ylim(0, 560)

# Single shared legend below the bottom panel
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, framealpha=0.9, edgecolor='none', fontsize=7)

fig.savefig(OUT_DIR / "retrain_curves.pdf")
fig.savefig(OUT_DIR / "retrain_curves.png")
plt.close(fig)
print("Saved retrain_curves")


# ============================================================
# Figure 3: Per-seed scatter (with no-step-filter ablation)
# ============================================================

fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))

# Per-seed data
data = {
    'Ours (full)':      {'frf': [0.031, 0.041, 0.043], 'ret': [500, 500, 500],
                          'c': C_OURS, 'm': 'o'},
    'No step filter':   {'frf': [0.032, 0.049, 0.048], 'ret': [500, 500, 500],
                          'c': C_NOFILTER, 'm': 'D'},
    'Random forget':    {'frf': [0.051, 0.762, 0.468], 'ret': [500, 500, 500],
                          'c': C_RAND, 'm': '^'},
    'Fine-tuning':      {'frf': [0.034, 0.055, 0.041], 'ret': [500, 500, 500],
                          'c': C_FINETUNE, 'm': 'P'},
    'Full retrain':     {'frf': [0.052, 0.087, 0.033], 'ret': [71.7, 55.8, 109.7],
                          'c': C_RETRAIN, 'm': 's'},
}

# Ideal region
ax.axvspan(0, 0.1, alpha=0.06, color='green', zorder=0)
ax.axhspan(450, 560, alpha=0.06, color='green', zorder=0)

for label, d in data.items():
    ax.scatter(d['frf'], d['ret'], c=d['c'], s=55, marker=d['m'],
               label=label, zorder=5, edgecolors='black', linewidth=0.4)

ax.set_xlabel('Forget-Region Fraction')
ax.set_ylabel('Mean Return')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2,
          framealpha=0.9, edgecolor='none', fontsize=7.5)
ax.set_xlim(-0.03, 0.85)
ax.set_ylim(0, 560)

fig.savefig(OUT_DIR / "per_seed_scatter.pdf")
fig.savefig(OUT_DIR / "per_seed_scatter.png")
plt.close(fig)
print("Saved per_seed_scatter")


# ============================================================
# Figure 4: Bar chart — final metrics comparison (5 methods)
#            Now full-width (figure*)
# ============================================================

methods = ['Original', 'Ours (full)', 'No step filter', 'Random forget', 'Fine-tuning', 'Full retrain']
colors = [C_ORIG, C_OURS, C_NOFILTER, C_RAND, C_FINETUNE, C_RETRAIN]

frf_means = [0.365, 0.038, 0.043, 0.427, 0.043, 0.057]
frf_stds  = [0.175, 0.005, 0.008, 0.291, 0.009, 0.022]

ret_means = [500.0, 500.0, 500.0, 500.0, 500.0, 79.1]
ret_stds  = [0.0,   0.0,   0.0,   0.0,   0.0,   22.3]

pos_means = [-0.046, -0.655, -0.407, -0.023, None, None]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(3.4, 5.2),
                                     gridspec_kw={'hspace': 0.55})

x = np.arange(len(methods))
width = 0.6

# Top: forget-region fraction
bars1 = ax1.bar(x, frf_means, width, yerr=frf_stds, capsize=3,
                color=colors, edgecolor='black', linewidth=0.4, alpha=0.85)
ax1.set_ylabel('Forget-Reg.\nFrac.')
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=6, rotation=25, ha='right')
ax1.set_ylim(0, 0.85)

# Middle: return
bars2 = ax2.bar(x, ret_means, width, yerr=ret_stds, capsize=3,
                color=colors, edgecolor='black', linewidth=0.4, alpha=0.85)
ax2.set_ylabel('Mean Return')
ax2.set_xticks(x)
ax2.set_xticklabels(methods, fontsize=6, rotation=25, ha='right')
ax2.set_ylim(0, 580)

# Bottom: cart position mean (N/A for fine-tuning and full retrain)
pos_vals = [-0.046, -0.655, -0.407, -0.023]
pos_indices = [0, 1, 2, 3]  # Original, Ours, No step filter, Random forget
pos_colors_sub = [colors[i] for i in pos_indices]
bars3 = ax3.bar(pos_indices, pos_vals, width, color=pos_colors_sub,
                edgecolor='black', linewidth=0.4, alpha=0.85)
ax3.set_ylabel('Cart Position Mean')
ax3.set_xticks(x)
ax3.set_xticklabels(methods, fontsize=6, rotation=25, ha='right')
ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
ax3.set_ylim(-0.8, 0.15)
ax3.annotate('← more left', xy=(0.02, 0.02), xycoords='axes fraction',
             fontsize=6, color='gray')
# Mark N/A for methods without cart position data
for i in [4, 5]:  # Fine-tuning, Full retrain
    ax3.text(i, -0.02, 'N/A', ha='center', va='top', fontsize=6, color='gray')
fig.savefig(OUT_DIR / "bar_comparison.pdf")
fig.savefig(OUT_DIR / "bar_comparison.png")
plt.close(fig)
print("Saved bar_comparison")

print("\nAll figures saved to paper_figures/")
