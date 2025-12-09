# %%
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from nilearn import datasets, plotting
from nilearn.surface import SurfaceImage
from nilearn.plotting import cm
from nilearn.image import mean_img, math_img

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
N_PARCELS = 400
DATASET = f"IBC_{N_PARCELS}/FaceBody"
METHODS = ["Euclidean", "Procrustes", "Optimal Transport", "Ridge"]
METHOD_PATHS = {
    "Euclidean": data_path / DATASET / "Anatomical",
    "Procrustes": data_path / DATASET / "Procrustes",
    "Optimal Transport": data_path / DATASET / "ot",
    "Ridge": data_path / DATASET / "Ridge",
}

# Load meshes
mesh = "fsaverage5"
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


# Plot each method
for i, method in enumerate(METHODS):
    # Classifier weights (top row)
    ax_weights = fig.add_subplot(grid[0, i], projection="3d")
    ax_weights.set_title(METHODS[i])
    img_paths = list(
        (METHOD_PATHS[method] / "template_in_sample").glob(
            "*/coefs_Faces_adult.nii.gz"
        )
    )
    surface_img = project_to_surface(
        math_img(
            "img/img.max()", img=mean_img(img_paths)
        ),  # Average the results across folds
    )
    plot_surface(
        ax_weights,
        surface_img,
        cmap,
    )

    # Add zoomed inset
    # Plot the same surface in the inset with the same parameters
    ax_inset = fig.add_subplot(grid[1, i], projection="3d")
    plot_surface(
        ax_inset,
        surface_img,
        cmap,
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
        ymin - 4.5,
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
