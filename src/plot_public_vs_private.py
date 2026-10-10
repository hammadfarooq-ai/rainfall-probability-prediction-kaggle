"""Public vs private AUC for every scored submission -> assets/public_vs_private.png"""
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'

s = pd.read_csv('results/submissions.csv').dropna(subset=['publicScore', 'privateScore'])
s = s[s.publicScore > 0.8]
fm = s.description.fillna('').str.contains('TabICL|TabPFN')
rho = s.publicScore.corr(s.privateScore, method='spearman')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160, facecolor=SURFACE); ax.set_facecolor(SURFACE)
for mask, col, lab in [(~fm, BLUE, 'Classical models and stacks (CPU)'), (fm, ORANGE, 'Blends containing TabPFN / TabICL (Kaggle GPU)')]:
    ax.scatter(s.publicScore[mask], s.privateScore[mask], s=52, color=col, edgecolor=SURFACE, linewidth=1.4, label=lab, zorder=3)


def note(fname, text, dx, dy):
    r = s[s.fileName == fname].iloc[0]
    ax.annotate(text, (r.publicScore, r.privateScore), xytext=(dx, dy), textcoords='offset points', color=INK2, fontsize=8.5,
                arrowprops=dict(arrowstyle='-', color=AXIS, lw=0.8), ha='left' if dx > 0 else 'right', va='center')


note('sub_x_icl_m_old.csv', 'final score: 0.9033\n(selected by best public)', -8, -98)
note('sub_x_icl_m_lr.csv', 'best private: 0.9048', -60, 2)
note('submission_lgb.csv', 'single LightGBM', 14, -4)
ax.set_xlabel('Public leaderboard AUC (about 25% of test rows)', color=INK2)
ax.set_ylabel('Private leaderboard AUC (about 75%)', color=INK2)
ax.set_title('TabPFN / TabICL blends led on both leaderboards', loc='left', color=INK, fontsize=12.5, pad=28)
ax.text(0, 1.035, f'{len(s)} scored submissions; Spearman correlation between public and private = {rho:.2f}', transform=ax.transAxes, color=MUTED, fontsize=9)
ax.grid(True, color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
for sp in ('left', 'bottom'): ax.spines[sp].set_color(AXIS)
ax.tick_params(colors=MUTED, length=0)
ax.legend(loc='lower right', frameon=False, labelcolor=INK2, fontsize=9, handletextpad=0.3)
fig.tight_layout(); fig.savefig('assets/public_vs_private.png', facecolor=SURFACE)
print('saved', len(s), 'points; corr', round(s.publicScore.corr(s.privateScore, method='spearman'), 3))
