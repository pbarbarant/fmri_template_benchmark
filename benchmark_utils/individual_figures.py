# %%
from pathlib import Path
import matplotlib.pyplot as plt

from nilearn import plotting

from nilearn import image

plt.rcParams["figure.dpi"] = 500

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "Simulated_MNI152",
    "IBC_Audio",
    "IBC_FaceBody",
    "IBC_Mario",
    "IBC_RSVPLanguage",
    "IBC_MathLanguage",
    "NSD",
    "Forrest",
    "HCP_Wm",
]

SOLVERS = [
    "Anatomical",
    "Diagonal",
    "OptimalTransport",
    "Procrustes",
    "Ridge",
]

for dataset in DATASETS:
    fig, axs = plt.subplots(1, len(SOLVERS), figsize=(50, 5))
    for solver in SOLVERS:
        template_path = data_path / dataset / solver / "template.nii.gz"
        # Load the template
        if template_path.exists():
            template = image.index_img(template_path, 0)
            plotting.plot_stat_map(
                template,
                title=f"{solver}",
                display_mode="ortho",
                cut_coords=(0, 0, 0),
                draw_cross=False,
                axes=axs[SOLVERS.index(solver)],
                vmax=2,
                vmin=-2,
            )
        else:
            axs[SOLVERS.index(solver)].axis("off")
            axs[SOLVERS.index(solver)].set_title(f"{solver} not available")
    fig.suptitle(dataset)
    print(f"Saving {dataset}")
    plt.savefig(
        figures_path / f"{dataset}_individual_figures.png",
        dpi=500,
        bbox_inches="tight",
    )
