# %%
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting
from nilearn.image import (
    math_img,
    mean_img,
)
from nilearn.plotting import cm
from nilearn.surface import SurfaceImage

# Setup
plt.rcParams.update(
    {
        "figure.dpi": 200,
        "figure.figsize": [7, 3],
        "font.size": 10,
        "text.usetex": False,
        "font.family": "sans-serif",
        "savefig.dpi": 300,
    }
)

# Paths
data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

# Constants
# Glasser (HCP-MMP1) labels on the 164k fsaverage mesh
FFC_LABELS = {
    "left": (
        Path(__file__).parent.parent
        / "fsaverage.label.L.164k_fsavg_L.label.gii",
        "L_FFC_ROI",
    ),
    "right": (
        Path(__file__).parent.parent
        / "fsaverage.label.R.164k_fsavg_R.label.gii",
        "R_FFC_ROI",
    ),
}
N_PARCELS = 400
DATASET = f"IBC_{N_PARCELS}/FaceBody"
METHODS = ["Anatomical", "Procrustes", "Optimal Transport", "Ridge"]
METHOD_PATHS = {
    "Anatomical": data_path / DATASET / "Anatomical",
    "Procrustes": data_path / DATASET / "Procrustes",
    "Optimal Transport": data_path / DATASET / "ot",
    "Ridge": data_path / DATASET / "Ridge",
}

# Load meshes
mesh = "fsaverage5"
fsaverage_meshes = datasets.load_fsaverage(mesh=mesh)
curv_sign = datasets.load_fsaverage_data(mesh=mesh, data_type="curvature")


def project_to_surface(img, interpolation="linear"):
    """Project volumetric image to surface."""
    return SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=img,
        interpolation=interpolation,
    )


def plot_surface(ax, surface_image, cmap):
    """Plot surface map."""
    plotting.plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes["inflated"],
        hemi="both",
        colorbar=False,
        cmap=cmap,
        bg_on_data=True,
        bg_map=curv_sign,
        axes=ax,
        vmin=0,
        vmax=0.6,
        threshold=0,
        alpha=0.5,
    )
    ax.view_init(elev=270, azim=-90)


cmap = LinearSegmentedColormap.from_list(
    "cmap_pos",
    cm.cold_white_hot(np.linspace(0.5, 1, 256)),
)


def draw_zoom_box(
    ax, xmin, xmax, ymin, ymax, z, color="k", linestyle="--", linewidth=2
):
    lines = [
        [(xmin, ymax, z), (xmax, ymax, z)],
        [(xmin, ymin, z), (xmax, ymin, z)],
        [(xmax, ymin, z), (xmax, ymax, z)],
        [(xmin, ymax, z), (xmin, ymin, z)],
    ]
    for line in lines:
        ax.plot3D(
            *zip(*line),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=1e10,
            clip_on=False,
        )


# Create figure
fig = plt.figure()
grid = gridspec.GridSpec(
    2,
    len(METHODS) + 1,
    figure=fig,
    wspace=0.2,
    hspace=0.01,
    width_ratios=[1] * len(METHODS) + [0.05],
)

# Zoom regions for weights plots (right column)
# Format: elevation, azimuth, zoom level, position, focus region

xmin = -85
ymin = -75
xmax = xmin + 50
ymax = ymin + 50
width, height = 0.15, 0.15


def load_weights_volume(method):
    """Average fold weights and normalize by their max."""
    img_paths = sorted(
        (METHOD_PATHS[method] / "template_in_sample").glob(
            "*/coefs_Faces_adult.nii.gz"
        )
    )
    return math_img("img/img.max()", img=mean_img(img_paths))


def load_ffc_surface():
    """FFC labels resampled to fsaverage5 (its vertices are the first ones of
    the 164k mesh) as a SurfaceImage."""
    n_vertices = fsaverage_meshes["pial"].parts["left"].n_vertices
    data = {}
    for hemi, (path, name) in FFC_LABELS.items():
        label_img = nib.load(path)
        keys = {label.label: label.key for label in label_img.labeltable.labels}
        in_label = label_img.darrays[0].data[:n_vertices] == keys[name]
        data[hemi] = in_label.astype(np.uint8)
    return SurfaceImage(mesh=fsaverage_meshes["pial"], data=data)


def mean_weight_in_label(weights_surf, ffc_surf):
    """Mean weight (positive and negative) over the FFC label, per hemisphere."""
    return {
        hemi: weights_surf.data.parts[hemi][
            ffc_surf.data.parts[hemi] > 0
        ].mean()
        for hemi in FFC_LABELS
    }


def draw_panel(ax, surface_img, roi_surf=None):
    """Draw weights on a 3D axis, with the ROI contour if given."""
    plot_surface(ax, surface_img, cmap)
    if roi_surf is not None:
        plotting.plot_surf_contours(
            surf_mesh=fsaverage_meshes["inflated"],
            roi_map=roi_surf,
            hemi="both",
            levels=[1],
            legend=False,
            colors=["royalblue"],
            view="ventral",
            axes=ax,
        )


def score_label(method):
    """Mean weight in the left and right FFC labels."""
    mean = mean_weights[method]
    return f"Mean weight\nL: {mean['left']:.2f}  R: {mean['right']:.2f}"


ffc_surf = load_ffc_surface()

# Load all methods once, in parallel, then project and score them
weight_imgs = dict(
    zip(
        METHODS,
        Parallel(n_jobs=len(METHODS), prefer="threads")(
            delayed(load_weights_volume)(m) for m in METHODS
        ),
    )
)
surface_imgs = {m: project_to_surface(w) for m, w in weight_imgs.items()}

mean_weights = {
    m: mean_weight_in_label(surf, ffc_surf) for m, surf in surface_imgs.items()
}

# Plot each method
for i, method in enumerate(METHODS):
    surface_img = surface_imgs[method]

    # Full view (top row)
    ax_weights = fig.add_subplot(grid[0, i], projection="3d")
    ax_weights.set_title(method)
    draw_panel(ax_weights, surface_img, ffc_surf)

    # Zoomed view (bottom row)
    ax_inset = fig.add_subplot(grid[1, i], projection="3d")
    draw_panel(ax_inset, surface_img, ffc_surf)
    ax_inset.set_xlim(xmin, xmax)
    ax_inset.set_ylim(ymin, ymax)
    ax_inset.set_axis_off()
    ax_inset.text2D(
        0.5,
        -0.02,
        score_label(method),
        transform=ax_inset.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )

    zmin, _ = ax_inset.get_zlim()

    # Zoom box on the inset and the corresponding box on the main plot
    draw_zoom_box(
        ax_inset,
        xmin,
        xmax + 5,
        ymin - 4.5,
        ymax + 3.5,
        zmin,
        color="black",
        linestyle="-",
        linewidth=1,
    )
    draw_zoom_box(
        ax_weights,
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        color="black",
        linestyle="-",
        linewidth=1,
    )

# Add colorbar
# Colorbar axis (the extra column)
cax = fig.add_subplot(grid[:, -1])  # span both rows

# Remove ticks for cleaner look (optional)
cax.tick_params(size=0, labelsize=8)

# Create colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=0.6)
cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax,
)

cax.set_ylabel("Weight", rotation=270, labelpad=15)

plt.show()

# Save figure
fig.savefig(figures_path / "surf_comparison.png", bbox_inches="tight")
