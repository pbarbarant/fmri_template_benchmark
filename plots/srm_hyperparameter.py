# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from statannotations.Annotator import Annotator
from tqdm import tqdm
from utils import DATA_PATH, FIGURES_PATH

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


def get_srm_results(
    data_path: Path, n_parcels: int | None = None
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
    df = df[~df["dataset_name"].str.startswith(("Simulated", "HCP"))]
    df["dataset_name"] = df["dataset_name"].str.replace(
        r"_[0-9]+$", "", regex=True
    )

    # Drop any solver apart from SRM
    df = df[(df["solver_name"].str.startswith("SRM"))]

    # Clean solver and task names
    df["solver_name"] = (
        df["solver_name"]
        .str.replace("SRM", "Shared Response K =")
        .str.replace("_", " ")
    )

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


df = get_srm_results(DATA_PATH, n_parcels=400)


plot_df = df.loc[df["target"] == "template_out_of_sample"].copy()
plot_df["n_components"] = (
    plot_df["solver_name"].str.extract(r"K\s*=\s*(\d+)")[0].astype(int)
)

plot_df = (
    plot_df.groupby(
        ["dataset_name", "task_name", "subject", "n_components"],
        as_index=False,
    )["cv_scores"]
    .mean()
    .sort_values(["task_name"])
)

g = sns.catplot(
    data=plot_df,
    x="n_components",
    y="cv_scores",
    col="task_name",
    col_wrap=2,
    kind="point",
    order=sorted(plot_df["n_components"].unique()),
    color=sns.color_palette("tab20c", n_colors=18)[16],
    errorbar=("ci", 95),
    capsize=0.15,
    markers="o",
    linestyles="-",
    height=2.8,
    aspect=1.2,
    sharey=True,
)

g.set_axis_labels("", "Accuracy")
g.set_titles("")


for ax, task in zip(g.axes.flat, g.col_names):
    task_df = plot_df.loc[plot_df["task_name"] == task]
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel(
        f"Number of components (K)\n{task}",
        fontsize=10,
        labelpad=8,
    )
    ax.xaxis.label.set_visible(True)

    # Retain complete subjects and ensure identical pairing order.
    wide = task_df.pivot(
        index=["dataset_name", "subject"],
        columns="n_components",
        values="cv_scores",
    ).dropna()

    paired_df = wide.reset_index().melt(
        id_vars=["dataset_name", "subject"],
        var_name="n_components",
        value_name="cv_scores",
    )

    pairs = [(20, k) for k in sorted(wide.columns) if k != 20]

    annotator = Annotator(
        ax,
        pairs,
        data=paired_df,
        x="n_components",
        y="cv_scores",
        order=sorted(plot_df["n_components"].unique()),
    )
    annotator.configure(
        test="t-test_paired",
        comparisons_correction="BH",
        text_format="star",
        correction_format="replace",
        hide_non_significant=False,
        loc="inside",
        verbose=0,
    )
    annotator.apply_and_annotate()

sns.despine()
g.figure.tight_layout()

g.figure.savefig(FIGURES_PATH / "annex_srm_components.pdf", bbox_inches="tight")
plt.show()
