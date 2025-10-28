# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from joblib import load

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


def get_results_dataframe(data_path: Path) -> pd.DataFrame:
    """Load and preprocess all decoding results."""
    results_paths = glob.glob(
        str(data_path / "**" / "decoding_results.pkl"), recursive=True
    )

    res_list = []
    for path in results_paths:
        path = Path(path)
        results = load(path)
        res_list.append(pd.DataFrame(results))

    df = pd.concat(res_list)

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

    # Handle Anatomical solver - only keep template_in_sample
    mask_anat = df["solver_name"] == "Anatomical"
    df_anat = df[mask_anat & (df["target"] == "template_in_sample")].copy()
    df_non_anat = df[~mask_anat].copy()

    # Create solver_target labels
    df_anat["solver_target"] = df_anat["solver_name"]
    df_non_anat["solver_target"] = np.select(
        [
            df_non_anat["target"] == "template_in_sample",
            df_non_anat["target"] == "template_out_of_sample",
        ],
        [
            df_non_anat["solver_name"],
            df_non_anat["solver_name"] + "\nOut of sample",
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
    palette = sns.color_palette("tab20", n_colors=12)
    del palette[1]
    del palette[-3:]

    result = []
    for i, color in enumerate(palette):
        result.append(color)
        if i >= 2 and (i - 2) % 2 == 0:
            result.append(color)

    return {k: v for k, v in zip(solvers_keys, result)}


def sum_time(df: pd.DataFrame):
    df = df.drop(
        ["subject", "cv_scores", "chance_level", "fold"], axis=1
    ).drop_duplicates()
    df = df.groupby(
        [
            "dataset_name",
            "task_name",
            "solver_name",
            "solver_target",
        ],
        as_index=False,
    )["time"].sum()

    return df.sort_values(["task_name", "solver_target"])


def create_barplot(data: pd.DataFrame, palette: dict, y: str = "cv_scores"):
    """Create a styled barplot."""
    fig, ax = plt.subplots()
    sns.barplot(
        data=data,
        x="task_name",
        y=y,
        hue="solver_target",
        dodge=True,
        palette=palette,
        edgecolor="k",
        ax=ax,
        legend=True,
    )

    return fig, ax


def time_comparison(
    data: pd.DataFrame,
    palette: dict,
):
    """Compare times for in-sample template alignment."""
    data = data[data.target != "template_out_of_sample"].copy()
    data = sum_time(data)
    fig, ax = create_barplot(data, palette, y="time")

    # Add hatching to Anatomical bars
    _, labels = ax.get_legend_handles_labels()
    anatomical_idx = labels.index("Anatomical")
    for container_idx, container in enumerate(ax.containers):
        if container_idx == anatomical_idx:
            for bar in container:
                bar.set_hatch("///")

    # Add rectangles for separation
    for i, task in enumerate(data["task_name"].unique()):
        plt.axvspan(
            i - 0.5,
            i + 0.5,
            facecolor="gray",
            alpha=[0.05 if i % 2 == 1 else 0][0],
        )

    ax.set_xlabel("Task (N Subjects)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Total Time (s)", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Add gridlines
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(
        title="Alignment method",
        title_fontsize=11,
        fontsize=10,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.4),
        ncol=4,
    )

    # Add hatching to legend
    legend = ax.get_legend()
    for patch, label in zip(legend.get_patches(), legend.get_texts()):
        if label.get_text() == "Anatomical":
            patch.set_hatch("///")

    plt.tight_layout()
    sns.despine(left=True)
    return fig


# Main execution
df = get_results_dataframe(data_path)
dict_palette = create_palette(df)

# Generate and save all plots
fig = time_comparison(df, palette=dict_palette)
fig.savefig(figures_path / "time_comparison.pdf", bbox_inches="tight")
plt.show()
