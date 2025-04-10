# %%
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from nilearn import image, plotting, datasets
from nilearn.surface import SurfaceImage
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.mplot3d import proj3d
import scienceplots
import glob
import numpy as np

# Setup
plt.style.use(["science", "nature", "no-latex"])
plt.rcParams.update({"figure.dpi": 500, "font.size": 10})

# Paths
data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

# Constants
N_PARCELS = 400
DATASET = f"IBC_FaceBody_{N_PARCELS}"
VMIN_WEIGHTS, VMAX_WEIGHTS = -0.003, 0.003
VMIN_CONTRAST, VMAX_CONTRAST = -1, 1
IDX_WEIGHTS, IDX_CONTRAST = 4, 19
METHODS = ["Euclidean", "Procrustes", "Optimal Transport"]
METHOD_PATHS = {
    "Euclidean": data_path / DATASET / "Anatomical",
    "Procrustes": data_path / DATASET / "Procrustes",
    "Optimal Transport": data_path / DATASET / "ot"
}

# Load meshes
mesh = "fsaverage3"
cache_dir = "/home/mind/pbarbara/.paths/pbarbara/fmri_template_benchmark/memory_cache"
fsaverage_meshes = datasets.load_fsaverage(mesh=mesh, data_dir=cache_dir)
curv_sign = datasets.load_fsaverage_data(mesh=mesh, data_type="curvature", data_dir=cache_dir)
for hemi, data in curv_sign.data.parts.items():
    curv_sign.data.parts[hemi] = np.sign(data)

def average_subjects_weights(weights_path):
    """Average subjects' weights."""
    imgs = [image.load_img(nii_file) for nii_file in glob.glob(str(weights_path / "*.nii.gz"))]
    data = np.mean([img.get_fdata() for img in imgs], axis=0)
    return image.new_img_like(imgs[0], data)

def project_to_surface(img, idx):
    """Project volumetric image to surface."""
    return SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=image.index_img(img, idx)
    )
    
def get_template_img(dataset, method):
    dataset_folder = data_path / dataset
    template_img_path = dataset_folder / method / "template.nii.gz"
    return image.load_img(template_img_path)

def get_avg_weights_img(dataset, method):
    method_path = data_path / dataset / method / "template"
    weights_files = sorted(list(method_path.glob("*_weights.nii.gz")))
    data = None
    for weights_file in weights_files:
        weights_img = image.load_img(weights_file)
        if data is None:
            data = weights_img.get_fdata()
        else:
            data += weights_img.get_fdata()
    data /= len(weights_files)
    return image.new_img_like(weights_img, data)


def get_threshold(imgs, quantile=0.90):
    # Get the threshold for each image
    data = np.concatenate([img.get_fdata() for img in imgs], axis=-1)
    # Return the quantile while discarding zeros
    return np.quantile(data[np.abs(data) > 0], quantile)

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
        darkness=0.25,
        axes=ax,
        vmin=vmin,
        vmax=vmax,
        threshold=threshold
    )
    ax.view_init(elev=270, azim=-90)
    
    
THRESHOLD_CONTRAST = get_threshold(
    [
        get_template_img(DATASET, METHOD_PATHS[method].name)
        for method in METHODS
    ],
    quantile=0.90
)

THRESHOLD_WEIGHTS = get_threshold(
    [
        get_avg_weights_img(DATASET, METHOD_PATHS[method].name)
        for method in METHODS
    ],
    quantile=0.80
)
# %%
# Create figure
fig = plt.figure(figsize=(4.5, 6))
grid = gridspec.GridSpec(3, 2, figure=fig, wspace=0.2, hspace=0.05)

# Zoom regions for weights plots (right column)
# Format: elevation, azimuth, zoom level, position, focus region
zoom_regions = [
    {"elev": 270, "azim": -90, "pos": (0.42, 0.68), "focus": (-5, 15, -5, 15, -5, 15)},  # Euclidean
    {"elev": 270, "azim": -90, "pos": (0.42, 0.42), "focus": (-5, 15, -5, 15, -5, 15)},  # Procrustes
    {"elev": 270, "azim": -90, "pos": (0.42, 0.15), "focus": (-5, 15, -5, 15, -5, 15)}   # Optimal Transport
]

# Plot each method
for i, method in enumerate(METHODS):
    # Template map (left column)
    ax_contrast = fig.add_subplot(grid[i, 0], projection="3d")
    surface_img = project_to_surface(METHOD_PATHS[method] / "template.nii.gz", IDX_CONTRAST)
    plot_surface(ax_contrast, surface_img, "coolwarm", VMIN_CONTRAST, VMAX_CONTRAST, THRESHOLD_CONTRAST)
    
    if i == 0:
        ax_contrast.set_title("Template Map ")
    
    # Label with method name
    y_pos = 0.4 if method != "Optimal Transport" else 0.2
    ax_contrast.text2D(-0.05, y_pos, method, transform=ax_contrast.transAxes, rotation=90)
    
    # Classifier weights (right column)
    ax_weights = fig.add_subplot(grid[i, 1], projection="3d")
    weights = average_subjects_weights(METHOD_PATHS[method] / "template/")
    surface_img = project_to_surface(weights, IDX_WEIGHTS)
    plot_surface(ax_weights, surface_img, "cold_hot", VMIN_WEIGHTS, VMAX_WEIGHTS, THRESHOLD_WEIGHTS)
    
    if i == 0:
        ax_weights.set_title("Classifier Weights ")
    
    # Add zoom region indicator using 3D line plots instead of 2D Rectangle
    zoom_settings = zoom_regions[i]
    xmin, xmax, ymin, ymax, zmin, zmax = zoom_settings["focus"]
    
    # Add zoomed inset for right column plots
    # Create a new axis for the zoomed region
    width, height = 0.15, 0.15
    ax_inset = fig.add_axes([zoom_settings["pos"][0], zoom_settings["pos"][1], width, height], projection='3d')
    
    # Plot the same surface in the inset with the same parameters
    plot_surface(ax_inset, surface_img, "cold_hot", VMIN_WEIGHTS, VMAX_WEIGHTS, THRESHOLD_WEIGHTS)
    
    # Adjust the view to focus on relevant brain region
    ax_inset.view_init(elev=zoom_settings["elev"], azim=zoom_settings["azim"] - 40)
    
    # Apply zoom by setting the limits based on focus region
    ax_inset.set_xlim(xmin, xmax)
    ax_inset.set_ylim(ymin, ymax)
    ax_inset.set_zlim(zmin, zmax)
    
    # Remove axes for cleaner look
    ax_inset.set_axis_off()

# Add colorbars
for j, (vmin, vmax, cmap, x_pos) in enumerate([
    (VMIN_CONTRAST, VMAX_CONTRAST, "coolwarm", 0.085),
    (VMIN_WEIGHTS, VMAX_WEIGHTS, "cold_hot", 0.54)
]):
    ax = fig.add_subplot(grid[:, j])
    ax.axis("off")
    cax = fig.add_axes([x_pos, 0.03, 0.3, 0.01])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap),
        cax=cax,
        orientation="horizontal"
    )

plt.tight_layout()
plt.show()

# Save figure
# fig.savefig(figures_path / f"surf_comparison.pdf", bbox_inches="tight")
