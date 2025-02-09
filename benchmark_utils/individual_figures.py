# %%
from pathlib import Path

import matplotlib.pyplot as plt
from nilearn import image, plotting

plt.rcParams["figure.dpi"] = 100

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

DATASETS = [
    # "Budapest_run-01",
    # "Budapest_run-02",
    # "Budapest_run-03",
    # "Budapest_run-04",
    # "Budapest_run-05",
    # "Simulated_MNI152",
    "IBC_Audio",
    # "IBC_FaceBody",
    # "IBC_Mario",
    # "IBC_RSVPLanguage",
    # "IBC_MathLanguage",
    # "NSD",
    # "Forrest",
    # "HCP_Wm",
]

SOLVERS = [
    "Anatomical",
    # "Diagonal",
    # "OptimalTransport",
    "Procrustes",
    # "SparseOT",
]

for dataset in DATASETS:
    vmax, vmin = 1, -1
    for solver in SOLVERS:
        template_path = data_path / dataset / solver / "template.nii.gz"
        # Load the template
        if template_path.exists():
            template = image.index_img(template_path, 0)
            plotting.plot_img_on_surf(
                template,
                title=f"{solver}",
                # axes=axs[SOLVERS.index(solver)],
                # vmax=vmax,
                # vmin=vmin,
            )
            plotting.show()
        else:
            print(f"{solver} not available for {dataset}")
    # fig.suptitle(dataset)
    # print(f"Saving {dataset}")
    # plt.savefig(
    #     figures_path / f"{dataset}_individual_figures.png",
    #     dpi=500,
    #     bbox_inches="tight",
    # )
