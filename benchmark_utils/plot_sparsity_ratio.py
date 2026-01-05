# %%
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn.maskers import NiftiMasker
from nilearn.image import concat_imgs

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

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

METHODS = ["Anatomical", "ot", "Procrustes", "Ridge"]
masker = NiftiMasker()


def sparsity_ratio(img):
    data = masker.fit_transform(img)
    return np.linalg.norm(data, axis=1, ord=2) / np.linalg.norm(
        data, axis=1, ord=1
    )


dict_results = {}
for method in METHODS:
    method_ = "Optimal Transport" if method == "ot" else method
    coef_list = sorted(
        data_path.rglob(f"{method}/template_out_of_sample*/**/*.nii.gz")
    )
    coef_list_filtered = [
        img
        for img in coef_list
        if "THINGS" not in str(img) and "Simulated" not in str(img)
    ]
    img = concat_imgs(coef_list_filtered)
    dict_results[method_] = sparsity_ratio(img)

dict_palette = {
    "Anatomical": (0.192, 0.510, 0.741),
    "Optimal Transport": (0.902, 0.333, 0.051),
    "Optimal Transport\nIn Sample": (0.992, 0.553, 0.235),
    "Optimal Transport\nPairwise": (0.992, 0.682, 0.420),
    "Procrustes": (0.192, 0.639, 0.329),
    "Procrustes\nIn Sample": (0.455, 0.769, 0.463),
    "Procrustes\nPairwise": (0.631, 0.851, 0.608),
    "Ridge": (0.459, 0.420, 0.694),
    "Ridge\nIn Sample": (0.620, 0.604, 0.784),
    "Ridge\nPairwise": (0.737, 0.741, 0.863),
    "Shared Response": (0.855, 0.855, 0.922),
    "Shared Response\nIn Sample": (0.388, 0.388, 0.388),
}

df = pd.DataFrame(dict_results).melt(var_name="Method", value_name="Sparsity")

fig, ax = plt.subplots()
sns.histplot(
    data=df,
    x="Sparsity",
    hue="Method",
    palette=dict_palette,
    bins=300,
    element="step",
    linewidth=1.5,
    ax=ax,
)
sns.despine(ax=ax)
ax.set_xlabel("Sparsity Ratio (L2 / L1)")
ax.set_ylabel("Count")

x1, x2 = 0.0065, 0.01
axins = inset_axes(
    ax, width="70%", height="70%", loc="upper right", borderpad=1.2
)
sns.histplot(
    data=df,
    x="Sparsity",
    hue="Method",
    palette=dict_palette,
    bins=300,
    element="step",
    linewidth=1.2,
    ax=axins,
    legend=False,
)
axins.set_xlim(x1, x2)
axins.set_ylim(0, ax.get_ylim()[1] * 0.6)
axins.set_xticks([])
axins.set_yticks([])
axins.set_xlabel("")
axins.set_ylabel("")

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", ls="--")

legend = ax.get_legend()
legend.set_title("Alignment Method")
legend.set_bbox_to_anchor((1.02, 0.5))
legend._loc = 6
legend.set_frame_on(False)

plt.tight_layout()
plt.show()
fig.savefig(figures_path / "sparsity_ratio.pdf", bbox_inches="tight")
