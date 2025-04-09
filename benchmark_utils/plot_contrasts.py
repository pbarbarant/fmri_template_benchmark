# %%
from pathlib import Path
from tqdm import tqdm
import scienceplots  # noqa: F401
import matplotlib.pyplot as plt
from nilearn import image, plotting, datasets
from nilearn.surface import SurfaceImage
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
HEMI = "right"
VMIN_WEIGHTS = -0.003
VMAX_WEIGHTS = 0.003
THRESHOLD_WEIGHTS = 0.0015
VMIN_CONTRAST = -1
VMAX_CONTRAST = 1
THRESHOLD_CONTRAST = 0.25

IDX_WEIGHTS = 4
IDX_CONTRAST = 0
mesh = "fsaverage5"
fsaverage_meshes = datasets.load_fsaverage(
    mesh=mesh, data_dir=data_path.parent / "memory_cache"
)

euclidean_path = data_path / DATASET / "Anatomical"
ot_path = data_path / DATASET / "ot"


def load_images_and_project_to_surface(img):
    """Util function for loading and projecting volumetric images."""
    surface_image = SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=img,
    )
    return surface_image


def plot_surface_map(surface_image, cmap, **kwargs):
    """Util function for plotting surfaces."""
    plotting.plot_surf_stat_map(
        stat_map=surface_image,
        surf_mesh=fsaverage_meshes["pial"],
        hemi=HEMI,
        view="lateral",
        colorbar=False,
        cmap=cmap,
        bg_on_data=False,
        darkness=0.25,
        **kwargs,
    )


n_contrasts = 14

fig = plt.figure(figsize=(4, n_contrasts * 2))
grid_spec = gridspec.GridSpec(
    n_contrasts, 2, figure=fig, wspace=0.00, hspace=0.00
)
for i in tqdm(range(n_contrasts)):
    ax = fig.add_subplot(grid_spec[i, 0], projection="3d")
    img_euclidean = load_images_and_project_to_surface(
        image.index_img(euclidean_path / "template.nii.gz", i)
    )
    img_ot = load_images_and_project_to_surface(
        image.index_img(ot_path / "template.nii.gz", i)
    )
    plot_surface_map(
        img_euclidean,
        cmap="coolwarm",
        axes=ax,
        vmin=VMIN_CONTRAST,
        vmax=VMAX_CONTRAST,
        threshold=THRESHOLD_CONTRAST,
    )
    ax = fig.add_subplot(grid_spec[i, 1], projection="3d")
    plot_surface_map(
        img_ot,
        cmap="coolwarm",
        axes=ax,
        vmin=VMIN_CONTRAST,
        vmax=VMAX_CONTRAST,
        threshold=THRESHOLD_CONTRAST,
    )

plt.tight_layout()
plt.show()

# Save as PDF
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
fig.savefig(figures_path / "contrasts.pdf", bbox_inches="tight")
