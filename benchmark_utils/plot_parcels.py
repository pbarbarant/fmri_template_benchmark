# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from joblib import load
from tqdm import tqdm
from typing import Optional

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
# Configuration
data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)


def get_results_dataframe(
    data_path: Path, n_parcels: Optional[int] = None
) -> pd.DataFrame:
    """Load and preprocess all decoding results."""
    results_paths = glob.glob(
        str(data_path / "**" / "decoding_results.pkl"), recursive=True
    )

    res_list = []
    for path in tqdm(results_paths):
        path = Path(path)
        results = load(path)
        res_list.append(pd.DataFrame(results))

    df = pd.concat(res_list)

    # Extract n_parcels into a new column
    df["n_parcels"] = df["dataset_name"].str.extract(r"_(\d+)$").astype("Int64")

    if n_parcels is not None:
        df = df[df["n_parcels"] == n_parcels]

    # Clean dataset names
    df = df[~df["dataset_name"].str.contains("Simulated")]
    df["dataset_name"] = df["dataset_name"].str.replace(
        r"_[0-9]+$", "", regex=True
    )

    # Clean solver and task names
    df["solver_name"] = (
        df["solver_name"]
        .str.replace("ot", "Optimal Transport")
        .str.replace("SRM", "Shared Response")
        .str.replace("_", " ")
    )
    df["task_name"] = df["task_name"].str.replace("RSVPLanguage", "Language")

    # Handle Anatomical solver - only keep template_out_of_sample
    mask_anat = df["solver_name"] == "Anatomical"
    df_anat = df[mask_anat & (df["target"] == "template_out_of_sample")].copy()
    df_non_anat = df[~mask_anat].copy()

    # Create solver_target labels
    df_anat["solver_target"] = df_anat["solver_name"]
    df_non_anat["solver_target"] = np.select(
        [
            df_non_anat["target"] == "template_in_sample",
            df_non_anat["target"] == "template_out_of_sample",
        ],
        [
            df_non_anat["solver_name"] + "\nIn Sample",
            df_non_anat["solver_name"],
        ],
        default=df_non_anat["solver_name"] + "\nPairwise",
    )

    df = pd.concat([df_anat, df_non_anat])

    # Add subject counts to task names
    anat_counts = df.groupby("task_name")["subject"].nunique().to_dict()
    df["task_name"] = df["task_name"].apply(
        lambda x: f"{x} (N={anat_counts.get(x, 0)})"
    )

    return df.sort_values(["task_name", "solver_target"])


def create_palette(df: pd.DataFrame) -> dict:
    """Create color palette for solvers."""
    solvers_keys = sorted(df.solver_target.unique())
    palette = sns.color_palette("tab20c", n_colors=18)
    palette = [
        c for i, c in enumerate(palette) if i not in (1, 2, 3, 7, 11, 15)
    ]
    result = []
    for i, color in enumerate(palette):
        result.append(color)

    return {k: v for k, v in zip(solvers_keys, palette)}


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
        dodge=True,
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


df = get_results_dataframe(data_path)
anat_level = df[df["solver_name"] == "Anatomical"]["cv_scores"].mean()
dict_palette = create_palette(df)

# Delete Anatomical from dataframe to avoid duplication in plot
df = df[df["solver_name"] != "Anatomical"]

fig, ax = parcellation_influence(
    df, palette=dict_palette, anat_level=anat_level
)
fig.savefig(
    figures_path / "parcellation_influence.pdf",
    bbox_inches="tight",
)
plt.show()
