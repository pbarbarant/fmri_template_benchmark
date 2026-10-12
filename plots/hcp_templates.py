# %%
import hcp_utils as hcp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from neuromaps.datasets import fetch_atlas
from neuromaps.images import load_gifti
from nilearn.plotting import plot_surf, plot_surf_contours
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
mask_l = load_gifti(atlas["medial"][0]).agg_data().astype(bool)
mask_r = load_gifti(atlas["medial"][1]).agg_data().astype(bool)
n_vert = mask_l.size  # 32492 per hemisphere


# --- Meshes ---
def load_mesh(path):
    g = load_gifti(path)
    return g.agg_data("pointset"), g.agg_data("triangle")


(coords_l, faces_l) = load_mesh(atlas["inflated"][0])
(coords_r, faces_r) = load_mesh(atlas["inflated"][1])

# Left lateral view (original behaviour, used for the supplementary figure)
geometry_left = (coords_l, faces_l)

# Both hemispheres in a single mesh, separated along x so they don't overlap
gap = 20.0
cl, cr = coords_l.copy(), coords_r.copy()
cl[:, 0] -= cl[:, 0].max() + gap / 2
cr[:, 0] -= cr[:, 0].min() - gap / 2
geometry_both = (
    np.vstack([cl, cr]),
    np.vstack([faces_l, faces_r + n_vert]),
)

# --- Glasser (HCP-MMP1.0) motor cortex outline, fsLR 32k ---
# Area 4 = primary motor cortex. To include the rest of Glasser's
# "somatosensory and motor" group, extend to e.g. {"4", "3a", "3b", "1", "2"}.
MOTOR_AREAS = {"4", "3a", "3b", "1", "2"}


def motor_roi_for(prefix, getter):
    mmp = getter(hcp.mmp.map_all, fill=0).astype(int)
    ids = [
        k
        for k, name in hcp.mmp.labels.items()
        if name.startswith(prefix)
        and name[len(prefix) :].replace("_ROI", "") in MOTOR_AREAS
    ]
    return np.isin(mmp, ids).astype(int)


motor_l = motor_roi_for("L_", hcp.left_cortex_data)
motor_r = motor_roi_for("R_", hcp.right_cortex_data)
assert motor_l.size == motor_r.size == n_vert, "Glasser/surface size mismatch"
motor_roi_both = np.concatenate([motor_l, motor_r])


def load_hemis(solver):
    template = np.load(
        DATA_PATH
        / f"HCP/hcp_{n_subjects}_{n_movies}/{solver}/template_in_sample/template.npy"
    )
    full = np.full((len(template), parcels.size), np.nan)
    full[:, parcels != 0] = template
    n_l, n_r = mask_l.sum(), mask_r.sum()

    left = np.full((len(template), n_vert), np.nan, dtype=np.float32)
    left[:, mask_l] = full[:, :n_l]
    right = np.full((len(template), n_vert), np.nan, dtype=np.float32)
    right[:, mask_r] = full[:, n_l : n_l + n_r]
    return left[order], right[order]


hemis = {s: load_hemis(s) for s in solvers}
data_left = {s: hemis[s][0] for s in solvers}
data_both = {s: np.hstack(hemis[s]) for s in solvers}
vmax = np.nanpercentile(np.abs(np.stack(list(data_both.values()))), 99)


def plot(
    rows,
    figsize,
    location,
    pad,
    path,
    fs,
    outline=False,
    dorsal=False,
    horizontal=False,
):
    geometry = geometry_both if dorsal else geometry_left
    data = data_both if dorsal else data_left
    view = "dorsal" if dorsal else "lateral"

    n_s = len(solvers)
    shape = (n_s, len(rows)) if horizontal else (len(rows), n_s)
    fig, axes = plt.subplots(
        *shape,
        figsize=figsize,
        layout="constrained",
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    for i, r in enumerate(rows):
        for j, s in enumerate(solvers):
            ax = axes[j, i] if horizontal else axes[i, j]
            plot_surf(
                geometry,
                data[s][r],
                hemi="left",  # only affects camera; both hemis are in the mesh
                view=view,
                axes=ax,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                colorbar=False,
                bg_on_data=False,
            )
            if outline:
                plot_surf_contours(
                    geometry,
                    motor_roi_both if dorsal else motor_l,
                    levels=[1],
                    colors=["green"],
                    hemi="left",
                    view=view,
                    axes=ax,
                    figure=fig,
                    linewidths=1.0,
                )

            # Column titles / row labels depend on the orientation
            on_top = (j == 0) if horizontal else (i == 0)
            on_left = (i == 0) if horizontal else (j == 0)
            title = conditions[r] if horizontal else solvers[s]
            side = solvers[s] if horizontal else conditions[r]

            if on_top:
                ax.set_title(title, fontsize=fs, fontweight="bold")
            if on_left:
                ax.text2D(
                    0,
                    0.5,
                    side,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90 if horizontal else 0,
                    fontsize=fs,
                    fontweight="bold",
                    color="black",
                )
            if dorsal:
                ax.view_init(elev=-270, azim=90)

    sm = ScalarMappable(cmap="coolwarm", norm=Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=axes, location=location, shrink=0.4, aspect=30, pad=pad)
    fig.savefig(
        FIGURES_PATH / path, dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.show()


# %% MAIN FIGURE
plot(
    np.flatnonzero(tasks == "MOTOR"),
    (7, 5.5),  # width and height swapped
    "bottom",
    0.05,
    "hcp_templates_motor.png",
    fs=12,
    outline=True,
    dorsal=True,
    horizontal=True,
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

# %%
