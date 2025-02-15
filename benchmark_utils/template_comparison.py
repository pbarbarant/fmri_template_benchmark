# %%
from pathlib import Path
import scienceplots  # noqa: F401
import matplotlib.pyplot as plt
from nilearn import image, plotting, datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nilearn.surface import SurfaceImage
import matplotlib as mpl
import matplotlib.gridspec as gridspec

plt.rcParams["figure.dpi"] = 500
# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
# Increase the font size
plt.rcParams.update({"font.size": 10})

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

DATASET = "IBC_FaceBody"
VMIN = -1
VMAX = 1
THRESHOLD = 0.25
HEMI = "right"
CONTRAST_IDX = 19

mesh = "fsaverage5"
fsaverage_meshes = datasets.load_fsaverage(mesh=mesh)

euclidean_path = data_path / DATASET / "Anatomical" / "template.nii.gz"
procrustes_path = data_path / DATASET / "Procrustes" / "template.nii.gz"
ot_path = data_path / DATASET / "ot" / "template.nii.gz"


def load_images_and_project_to_surface(image_path, idx):
    """Util function for loading and projecting volumetric images."""
    surface_image = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=image.index_img(image_path, idx),
    )
    return surface_image


def plot_surface_map(surface_image, cmap="coolwarm", **kwargs):
    """Util function for plotting surfaces."""
    plotting.plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes["inflated"],
        hemi=HEMI,
        view="ventral",
        colorbar=False,
        cmap=cmap,
        bg_on_data=False,
        darkness=0.25,
        **kwargs,
    )


fig = plt.figure(figsize=(5, 3))
grid_spec = gridspec.GridSpec(1, 3, figure=fig, wspace=0.00)
ax0 = fig.add_subplot(grid_spec[0, 0], projection="3d")
surf_euclidean = load_images_and_project_to_surface(
    euclidean_path, CONTRAST_IDX
)
plot_surface_map(
    surf_euclidean,
    axes=ax0,
    vmin=VMIN,
    vmax=VMAX,
    threshold=THRESHOLD,
)
# Add title
ax0.set_title("Euclidean")
ax0.view_init(elev=270, azim=-90)

ax1 = fig.add_subplot(grid_spec[0, 1], projection="3d")
surf_procrustes = load_images_and_project_to_surface(
    procrustes_path, CONTRAST_IDX
)
plot_surface_map(
    surf_procrustes,
    axes=ax1,
    vmin=VMIN,
    vmax=VMAX,
    threshold=THRESHOLD,
)
ax1.set_title("Procrustes")
ax1.view_init(elev=270, azim=-90)

ax2 = fig.add_subplot(grid_spec[0, 2], projection="3d")
surf_ot = load_images_and_project_to_surface(ot_path, CONTRAST_IDX)
plot_surface_map(
    surf_ot,
    axes=ax2,
    vmin=VMIN,
    vmax=VMAX,
    threshold=THRESHOLD,
)
ax2.set_title("Optimal Transport")
ax2.view_init(elev=270, azim=-90)

# Add colorbar
ax = fig.add_subplot(grid_spec[0, 1])
ax.axis("off")
divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%")
fig.add_axes(cax)
fig.colorbar(
    mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=VMIN, vmax=VMAX),
        cmap="coolwarm",
    ),
    cax=cax,
    orientation="horizontal",
)
plt.tight_layout()
plt.show()

# Save as PDF
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
fig.savefig(
    figures_path / f"templates_comparison_{HEMI}.pdf", bbox_inches="tight"
)
