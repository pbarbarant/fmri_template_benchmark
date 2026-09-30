# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from neuromaps.datasets import fetch_atlas
from neuromaps.images import load_gifti
from nilearn.plotting import plot_surf
from utils import DATA_PATH, FIGURES_PATH

solvers = {
    "Anatomical": "Anatomical",
    "ot": "Optimal Transport",
    "Procrustes": "Procrustes",
}
n_movies, n_subjects = 1, 100

df = pd.read_csv(DATA_PATH.parent / "data/hcp/sub-100610_labels.csv")
order = df.sort_values(["task", "condition"], kind="stable").index.values
conditions, tasks = df["condition"].values[order], df["task"].values[order]

unique_tasks = list(dict.fromkeys(tasks))

parcels = np.load(
    DATA_PATH.parent / "data/hcp/schaefer_400_parcellation.npy"
).flatten()
atlas = fetch_atlas("fsLR", "32k")
mask = load_gifti(atlas["medial"][0]).agg_data().astype(bool)
geometry = load_gifti(atlas["inflated"][0]).agg_data()


def load_left(solver):
    template = np.load(
        DATA_PATH
        / f"HCP/hcp_{n_subjects}_{n_movies}/{solver}/template_in_sample/template.npy"
    )
    full = np.full((len(template), parcels.size), np.nan)
    full[:, parcels != 0] = template
    left = np.full((len(template), mask.size), np.nan, dtype=np.float32)
    left[:, mask] = full[:, : mask.sum()]
    return left[order]


data = {s: load_left(s) for s in solvers}
vmax = np.nanpercentile(np.abs(np.stack(list(data.values()))), 99)


def plot(rows, figsize, location, pad, path, fs):
    fig, axes = plt.subplots(
        len(rows),
        len(solvers),
        figsize=figsize,
        layout="constrained",
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    for i, r in enumerate(rows):
        for j, s in enumerate(solvers):
            ax = axes[i, j]
            plot_surf(
                geometry,
                data[s][r],
                hemi="left",
                view="lateral",
                axes=ax,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                colorbar=False,
                bg_on_data=False,
            )
            if i == 0:
                ax.set_title(solvers[s], fontsize=fs + 3, fontweight="bold")
            if j == 0:
                ax.text2D(
                    0,
                    0.5,
                    conditions[r],
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=fs,
                    fontweight="bold",
                    color="black",
                )

    sm = ScalarMappable(cmap="coolwarm", norm=Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=axes, location=location, shrink=0.4, aspect=30, pad=pad)
    fig.savefig(
        FIGURES_PATH / path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.show()


# %% MAIN FIGURE
plot(
    np.flatnonzero(tasks == "MOTOR"),
    (4.5, 7),
    "bottom",
    0.05,
    "hcp_templates_motor.png",
    fs=8,
)

# %% SUPPLEMENTARY FIGURE
plot(
    np.arange(len(conditions)),
    (4.27, 11.69),
    "right",
    0.02,
    "annex_hcp_templates_by_task.png",
    fs=6,
)
