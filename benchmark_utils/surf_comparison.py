# %%
from pathlib import Path
import scienceplots  # noqa: F401
import matplotlib.pyplot as plt
from nilearn import image, plotting, datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nilearn.surface import SurfaceImage
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import glob
import numpy as np

plt.rcParams["figure.dpi"] = 500
# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
# Increase the font size
plt.rcParams.update({"font.size": 10})

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

DATASET = "IBC_FaceBody"
HEMI = "right"
VMIN_WEIGHTS = -0.003
VMAX_WEIGHTS = 0.003
THRESHOLD_WEIGHTS = 0.0015
VMIN_CONTRAST = -1
VMAX_CONTRAST = 1
THRESHOLD_CONTRAST = 0.25

IDX_WEIGHTS = 4
IDX_CONTRAST = 19

mesh = "fsaverage7"
fsaverage_meshes = datasets.load_fsaverage(mesh=mesh)

euclidean_path = data_path / DATASET / "Anatomical"
procrustes_path = data_path / DATASET / "Procrustes"
ot_path = data_path / DATASET / "ot"


def average_subjects_weights(weights_path):
    """Util function for averaging subjects' weights."""
    # Glob all nii.gz files
    nii_files = glob.glob(str(weights_path / "*.nii.gz"))
    # Load all nii files
    imgs = [image.load_img(nii_file) for nii_file in nii_files]
    data = np.mean([img.get_fdata() for img in imgs], axis=0)
    return image.new_img_like(imgs[0], data)


def load_images_and_project_to_surface(img, idx):
    """Util function for loading and projecting volumetric images."""
    surface_image = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=image.index_img(img, idx),
    )
    return surface_image


def plot_surface_map(surface_image, cmap, **kwargs):
    """Util function for plotting surfaces."""
    plotting.plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes["inflated"],
        hemi=HEMI,
        view="lateral",
        colorbar=False,
        cmap=cmap,
        bg_on_data=False,
        darkness=0.25,
        **kwargs,
    )


fig = plt.figure(figsize=(4, 6))
grid_spec = gridspec.GridSpec(3, 2, figure=fig, wspace=0.00, hspace=0.00)
ax0 = fig.add_subplot(grid_spec[0, 0], projection="3d")
plot_surface_map(
    load_images_and_project_to_surface(
        euclidean_path / "template.nii.gz", IDX_CONTRAST
    ),
    cmap="coolwarm",
    axes=ax0,
    vmin=VMIN_CONTRAST,
    vmax=VMAX_CONTRAST,
    threshold=THRESHOLD_CONTRAST,
)
ax0.view_init(elev=270, azim=-90)
ax0.set_title("Template Map")
ax0.text2D(
    0.05,
    0.4,
    "Euclidean",
    transform=ax0.transAxes,
    rotation=90,
)

# Add method vertically on the left
ax1 = fig.add_subplot(grid_spec[0, 1], projection="3d")
weights_euclidean = average_subjects_weights(euclidean_path / "template/")
plot_surface_map(
    load_images_and_project_to_surface(weights_euclidean, IDX_WEIGHTS),
    cmap="cold_hot",
    axes=ax1,
    vmin=VMIN_WEIGHTS,
    vmax=VMAX_WEIGHTS,
    threshold=THRESHOLD_WEIGHTS,
)
ax1.view_init(elev=270, azim=-90)
ax1.set_title("Classifier Weights")

ax2 = fig.add_subplot(grid_spec[1, 0], projection="3d")
plot_surface_map(
    load_images_and_project_to_surface(
        procrustes_path / "template.nii.gz", IDX_CONTRAST
    ),
    cmap="coolwarm",
    axes=ax2,
    vmin=VMIN_CONTRAST,
    vmax=VMAX_CONTRAST,
    threshold=THRESHOLD_CONTRAST,
)
ax2.view_init(elev=270, azim=-90)
ax2.text2D(
    0.05,
    0.4,
    "Procrustes",
    transform=ax2.transAxes,
    rotation=90,
)

ax3 = fig.add_subplot(grid_spec[1, 1], projection="3d")
weights_procrustes = average_subjects_weights(procrustes_path / "template/")
plot_surface_map(
    load_images_and_project_to_surface(weights_procrustes, IDX_WEIGHTS),
    cmap="cold_hot",
    axes=ax3,
    vmin=VMIN_WEIGHTS,
    vmax=VMAX_WEIGHTS,
    threshold=THRESHOLD_WEIGHTS,
)
ax3.view_init(elev=270, azim=-90)

ax4 = fig.add_subplot(grid_spec[2, 0], projection="3d")
plot_surface_map(
    load_images_and_project_to_surface(
        ot_path / "template.nii.gz", IDX_CONTRAST
    ),
    cmap="coolwarm",
    axes=ax4,
    vmin=VMIN_CONTRAST,
    vmax=VMAX_CONTRAST,
    threshold=THRESHOLD_CONTRAST,
)
ax4.view_init(elev=270, azim=-90)
# Set square aspect ratio
ax4.text2D(
    0.05,
    0.2,
    "Optimal Transport",
    transform=ax4.transAxes,
    rotation=90,
)

ax5 = fig.add_subplot(grid_spec[2, 1], projection="3d")
weights_ot = average_subjects_weights(ot_path / "template/")
plot_surface_map(
    load_images_and_project_to_surface(weights_ot, IDX_WEIGHTS),
    cmap="cold_hot",
    axes=ax5,
    vmin=VMIN_WEIGHTS,
    vmax=VMAX_WEIGHTS,
    threshold=THRESHOLD_WEIGHTS,
)
# Zoom in to see the weights
ax5.view_init(elev=270, azim=-90)

# Add colorbar for contrasts
ax_contrast = fig.add_subplot(grid_spec[:, 0])
ax_contrast.axis("off")
divider = make_axes_locatable(ax_contrast)
cax = fig.add_axes([0.15, 0.03, 0.3, 0.01])
fig.add_axes(cax)
fig.colorbar(
    mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=VMIN_CONTRAST, vmax=VMAX_CONTRAST),
        cmap="coolwarm",
    ),
    cax=cax,
    orientation="horizontal",
)
# Add colorbar for contrasts
ax_weights = fig.add_subplot(grid_spec[:, 1])
ax_weights.axis("off")
divider = make_axes_locatable(ax_weights)
cax = fig.add_axes([0.6, 0.03, 0.3, 0.01])
fig.add_axes(cax)
fig.colorbar(
    mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=VMIN_WEIGHTS, vmax=VMAX_WEIGHTS),
        cmap="cold_hot",
    ),
    cax=cax,
    orientation="horizontal",
)

plt.tight_layout()
plt.show()

# Save as PDF
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
fig.savefig(figures_path / f"surf_comparison_{HEMI}.pdf", bbox_inches="tight")
