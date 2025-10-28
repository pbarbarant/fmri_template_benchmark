# %%
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from nilearn import datasets, plotting
from nilearn.surface import SurfaceImage
from nilearn.plotting import cm

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
data_path = Path(__file__).parent.parent / "outputs_no_folds"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

# Constants
N_PARCELS = 400
DATASET = f"IBC_{N_PARCELS}/FaceBody"
VMIN_WEIGHTS, VMAX_WEIGHTS = -0.003, 0.003
VMIN_CONTRAST, VMAX_CONTRAST = -1, 1
IDX_WEIGHTS, IDX_CONTRAST = 4, 19
METHODS = ["Euclidean", "Procrustes", "Optimal Transport", "Ridge"]
METHOD_PATHS = {
    "Euclidean": data_path / DATASET / "Anatomical",
    "Procrustes": data_path / DATASET / "Procrustes",
    "Optimal Transport": data_path / DATASET / "ot",
    "Ridge": data_path / DATASET / "Ridge",
}

# Load meshes
mesh = "fsaverage3"
cache_dir = (
    "/home/mind/pbarbara/.paths/pbarbara/fmri_template_benchmark/memory_cache"
)
fsaverage_meshes = datasets.load_fsaverage(mesh=mesh, data_dir=cache_dir)
curv_sign = datasets.load_fsaverage_data(
    mesh=mesh, data_type="curvature", data_dir=cache_dir
)


def project_to_surface(img, interpolation="linear"):
    """Project volumetric image to surface."""
    return SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=img,
        interpolation=interpolation,
    )


def plot_surface(ax, surface_image, cmap, vmin, vmax, threshold):
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
        vmin=vmin,
        vmax=vmax,
        threshold=threshold,
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
        )


# Create figure
fig = plt.figure()
grid = gridspec.GridSpec(2, len(METHODS), figure=fig, wspace=0.2, hspace=0.01)

# Zoom regions for weights plots (right column)
# Format: elevation, azimuth, zoom level, position, focus region

xmin = -85
ymin = -70
xmax = xmin + 50
ymax = ymin + 50
width, height = 0.15, 0.15

# Plot each method
for i, method in enumerate(METHODS):
    # Template map (left column)
    # Classifier weights (right column)
    ax_weights = fig.add_subplot(grid[0, i], projection="3d")
    ax_weights.set_title(METHODS[i])
    surface_img = project_to_surface(
        METHOD_PATHS[method] / "template_in_sample/coefs_Faces_adult.nii.gz",
    )
    plot_surface(
        ax_weights,
        surface_img,
        cmap,
        0,
        None,
        0,
    )

    # Add zoomed inset for right column plots
    # Plot the same surface in the inset with the same parameters
    ax_inset = fig.add_subplot(grid[1, i], projection="3d")
    plot_surface(
        ax_inset,
        surface_img,
        cmap,
        # VMIN_WEIGHTS,
        # VMAX_WEIGHTS,
        0,
        None,
        0,
    )

    # Apply zoom by setting the limits based on focus region
    ax_inset.set_xlim(xmin, xmax)
    ax_inset.set_ylim(ymin, ymax)

    # To adjust the zoom level, you can set the limits to a smaller range
    ax_inset.set_axis_off()

    zmin, _ = ax_inset.get_zlim()

    # Draw box on the inset
    draw_zoom_box(
        ax_inset,
        xmin,
        xmax + 5,
        ymin - 5,
        ymax + 3.5,
        zmin,
        color="black",
        linestyle="-",
        linewidth=1,
    )

    # Draw corresponding box on the main plot
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


plt.show()

# Save figure
fig.savefig(figures_path / "surf_comparison.pdf", bbox_inches="tight")
