# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from utils import DATA_PATH, FIGURES_PATH

solvers = {
    "ot": "Optimal Transport",
    "Procrustes": "Procrustes",
    "SRM_20": "Shared Response",
}
n_movies, n_subjects = 4, 100

df = pd.read_csv(DATA_PATH / "../data/hcp/sub-100610_labels.csv")
df.sort_values(
    ["condition", "task"], kind="stable", inplace=True, ignore_index=True
)
order = df.sort_values(["task", "condition"], kind="stable").index.values


def confusion_matrices(order):
    return [
        np.load(
            DATA_PATH
            / f"HCP/hcp_{n_subjects}_{n_movies}/{s}/template_in_sample/confusion_matrix.npy"
        )[np.ix_(order, order)]
        for s in solvers
    ]


def draw(ax, cm, labels, vmax, fs, colors=None, aspect="equal"):
    n = len(labels)
    im = ax.imshow(cm, cmap="gist_earth_r", vmin=0, vmax=vmax, aspect=aspect)
    if n <= 35:
        for i, j in np.ndindex(n, n):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=fs,
                color="white" if cm[i, j] > vmax / 10 else "black",
            )
    ax.set_xticks(range(n), labels, rotation=90, fontsize=fs)
    ax.set_yticks(range(n), labels, fontsize=fs)
    for tick_labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for label, c in zip(tick_labels, colors or ["black"] * n):
            label.set_color(c)
            label.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


# %% MAIN FIGURE
motor = order[df["task"].values[order] == "MOTOR"]
conditions = df["condition"].values[motor]
cms = confusion_matrices(motor)
vmax = max(cm.max() for cm in cms)

fig, axes = plt.subplots(
    1, 3, figsize=(7.5, 3.4), sharey=True, layout="constrained"
)
for k, (ax, cm, name) in enumerate(zip(axes, cms, solvers.values())):
    im = draw(ax, cm, conditions, vmax, fs=8)
    ax.tick_params(labelleft=k == 0)
    ax.set_title(name, fontsize=10, fontweight="bold")

fig.colorbar(
    im, ax=axes, location="bottom", shrink=0.4, aspect=30, pad=0.05
).set_label("Number of subjects", fontsize=8)
fig.savefig(
    FIGURES_PATH / "hcp_confusion_matrices.png", dpi=300, bbox_inches="tight"
)
plt.show()

# %% SUPPLEMENTARY FIGURE
conditions, tasks = df["condition"].values[order], df["task"].values[order]
unique_tasks = list(dict.fromkeys(tasks))
boundaries = np.cumsum([0] + [np.sum(tasks == t) for t in unique_tasks])
palette = plt.get_cmap("tab10" if len(unique_tasks) <= 10 else "tab20").colors
color = {t: palette[i % len(palette)] for i, t in enumerate(unique_tasks)}
n = len(conditions)
off = 0.06 * n
cms = confusion_matrices(order)
vmax = max(cm.max() for cm in cms)

fig = plt.figure(figsize=(7, 17), layout="constrained")
gs = fig.add_gridspec(4, 1, height_ratios=[0.15, 1, 1, 1])
header = fig.add_subplot(gs[0])
header.axis("off")
axes = [fig.add_subplot(gs[i]) for i in (1, 2, 3)]

top = blended_transform_factory(axes[0].transData, header.transAxes)
for t, start, end in zip(unique_tasks, boundaries[:-1], boundaries[1:]):
    narrow = end - start < 3
    header.add_patch(
        Rectangle(
            (start - 0.5, 0),
            end - start,
            0.25,
            facecolor=color[t],
            clip_on=False,
            transform=top,
        )
    )
    header.annotate(
        t,
        ((start + end - 1) / 2, 0.25),
        xycoords=top,
        xytext=(0, 2),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        color=color[t],
        va="bottom",
        ha="left" if narrow else "center",
        rotation=30 if narrow else 0,
        rotation_mode="anchor",
    )
header.set_title(
    list(solvers.values())[0], fontsize=12, fontweight="bold", pad=15, x=0.57
)

for k, (ax, cm, name) in enumerate(zip(axes, cms, solvers.values())):
    im = draw(
        ax,
        cm,
        conditions,
        vmax,
        fs=8,
        colors=[color[t] for t in tasks],
        aspect="equal",
    )
    ax.tick_params(labelbottom=k == 2, length=2)

    for b in boundaries[1:-1]:
        ax.axhline(b - 0.5, color="black", linewidth=1)
        ax.axvline(b - 0.5, color="black", linewidth=1)

    for t, start, end in zip(unique_tasks, boundaries[:-1], boundaries[1:]):
        ax.add_patch(
            Rectangle(
                (-0.5 - off, start - 0.5),
                0.6 * off,
                end - start,
                facecolor=color[t],
                clip_on=False,
            )
        )

    ax.set_xlim(-0.5 - 1.6 * off, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    if k:
        ax.set_title(name, fontsize=12, fontweight="bold", pad=4)

fig.colorbar(
    im, ax=axes, location="right", shrink=0.4, aspect=30, pad=0.02
).set_label("Number of subjects", fontsize=8)
fig.savefig(
    FIGURES_PATH / "annex_hcp_confusion_matrices.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
