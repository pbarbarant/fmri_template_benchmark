# %%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import DATA_PATH, FIGURES_PATH, create_palette, get_results_dataframe

sns.set_theme(
    context="paper",
    style="ticks",
    rc={
        "figure.figsize": [7, 6],
        "text.usetex": False,
        "font.family": "sans-serif",
        "savefig.dpi": 300,
    },
)


def parcellation_influence(
    data: pd.DataFrame,
    palette: dict,
    anat_level: float,
):
    """Show influence of number of parcels on CV scores."""
    fig, ax = plt.subplots()

    # Anatomical level line
    ax.axhline(
        anat_level,
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label="Anatomical",
    )

    sns.pointplot(
        data=data,
        x="n_parcels",
        y="cv_scores",
        hue="solver_target",
        palette=palette,
        markers="o",
        linestyles="-",
        ax=ax,
    )

    ax.set_xlabel("Number of parcels", fontsize=12, fontweight="bold")
    ax.set_ylabel("Averaged score", fontsize=12, fontweight="bold")

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Gridlines
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        title="Alignment method",
        title_fontsize=11,
        fontsize=10,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=3,
    )

    sns.despine(left=True)
    plt.tight_layout()

    return fig, ax


df = get_results_dataframe(DATA_PATH)
anat_level = df[df["solver_name"] == "Anatomical"]["cv_scores"].mean()
dict_palette = create_palette(df)

# Delete Anatomical from dataframe to avoid duplication in plot
df = df[df["solver_name"] != "Anatomical"]

fig, ax = parcellation_influence(
    df, palette=dict_palette, anat_level=anat_level
)
fig.savefig(
    FIGURES_PATH / "parcellation_influence.pdf",
    bbox_inches="tight",
)
plt.show()
