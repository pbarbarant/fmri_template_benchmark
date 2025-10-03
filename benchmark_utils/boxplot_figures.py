# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import scienceplots  # noqa: F401
import seaborn as sns
from joblib import load
from statannotations.Annotator import Annotator

plt.rcParams["figure.dpi"] = 300

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

N_PARCELS = 400


def get_results_dataframe(data_path: Path, n_parcels=400) -> pd.DataFrame:
    # Glob recursively all the decoding_results.pkl files
    results_paths = glob.glob(
        str(data_path / "**" / "decoding_results.pkl"), recursive=True
    )
    # Build a dataframe with all the results
    res_list = []
    for path in results_paths:
        path = Path(path)
        solver = path.parent.parent.name
        results = load(path)
        # Add the solver name to the results
        results["solver_name"] = solver
        df_one_solver = pd.DataFrame(results)
        res_list.append(df_one_solver)

    df = pd.concat(res_list)

    # Remove the simulated data
    df = df[~df["dataset_name"].str.contains("Simulated")]

    # Keep only the results for the specified number of parcels
    # df = df[df["dataset_name"].str.contains(f"{n_parcels}")]

    # Remove parcels numbers from the dataset names
    df["dataset_name"] = df["dataset_name"].str.replace(
        r"_[0-9]+$", "", regex=True
    )

    # Rename ot to Optimal Transport
    df["solver_name"] = df["solver_name"].str.replace(
        "ot", "Optimal Transport"
    )
    df["solver_name"] = df["solver_name"].str.replace("SRM", "Shared Response")
    # Sort alphabetically by dataset name and solver name
    df.sort_values(by=["task_name", "solver_name"], inplace=True)

    # Fix underscores in the solver names
    df["solver_name"] = df["solver_name"].str.replace("_", " ")

    # If solver is "Anatomical", keep only template_in_sample rows
    mask_anat = df["solver_name"] == "Anatomical"
    df = pd.concat(
        [
            df[mask_anat & (df["target"] == "template_in_sample")],
            df[~mask_anat],
        ]
    )

    # Separate Anatomical solver
    mask_anat = df["solver_name"] == "Anatomical"
    mask_non_anat = ~mask_anat

    # Keep only template_in_sample for Anatomical
    df_anat = df[mask_anat & (df["target"] == "template_in_sample")].copy()
    df_non_anat = df[mask_non_anat].copy()

    # Add solver_target only for non-Anatomical solvers
    df_non_anat["solver_target"] = np.select(
        [
            df_non_anat["target"] == "template_in_sample",
            df_non_anat["target"] == "template_out_of_sample",
        ],
        [
            df_non_anat["solver_name"] + " (In sample template)",
            df_non_anat["solver_name"] + " (Out of sample template)",
        ],
        default=df_non_anat["solver_name"] + " (Pairwise)",
    )

    # For Anatomical, keep solver_target same as solver_name
    df_anat["solver_target"] = df_anat["solver_name"]

    # Combine back
    df = pd.concat([df_anat, df_non_anat])

    return df.sort_values(["dataset_name", "task_name", "solver_name"])


df = get_results_dataframe(data_path, n_parcels=N_PARCELS)


# Set the style and font scale for better readability
# plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)


#######################################################################
def anat_vs_template(data: pd.DataFrame, figsize: tuple = (14, 7)):
    data = data.copy()
    data = data[data.target == "template_in_sample"]
    fig, ax = plt.subplots(figsize=figsize)
    # Use a better color palette
    palette = sns.color_palette(
        "Set1", n_colors=len(data["solver_target"].unique())
    )
    # Add error bars and improve styling
    ax = sns.barplot(
        data=data,
        x="task_name",
        y="cv_scores",
        hue="solver_target",
        palette=palette,
        errorbar="se",  # or "sd" for standard deviation
        capsize=0.1,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add hatching to Anatomical bars - iterate through all bars
    # Get the order of solvers as they appear in the legend
    _, labels = ax.get_legend_handles_labels()

    anatomical_idx = labels.index("Anatomical")

    # Iterate through containers (each container is one hue/solver)
    for container_idx, container in enumerate(ax.containers):
        if container_idx == anatomical_idx:
            for bar in container:
                bar.set_hatch("///")

    ax.set_xlabel("Task", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Improve legend with hatching for Anatomical
    legend = ax.legend(
        title="Alignment method",
        title_fontsize=11,
        fontsize=10,
        frameon=True,
        shadow=False,
        loc="best",
    )

    # Add hatching to Anatomical legend entry
    for patch, label in zip(legend.get_patches(), legend.get_texts()):
        if label.get_text() == "Anatomical (Pai)":
            patch.set_hatch("///")

    # Add chance levels
    for i, task in enumerate(data["task_name"].unique()):
        chance = data[data["task_name"] == task]["chance_level"].iloc[0]
        ax.hlines(
            chance,
            i - 0.4,
            i + 0.4,
            colors="k",
            linestyles="--",
            alpha=0.7,
            linewidth=2,
            label="Chance level" if i == 0 else "",
        )

    # Add horizontal gridlines
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)

    # Statistical annotations
    pairs = [
        ((task, "Anatomical"), (task, solver))
        for task in data["task_name"].unique()
        for solver in data["solver_target"].unique()
        if solver != "Anatomical"
    ]

    annotator = Annotator(
        ax,
        pairs=pairs,
        data=data,
        x="task_name",
        y="cv_scores",
        hue="solver_target",
    )
    annotator.configure(
        test="Wilcoxon",
        text_format="star",
        loc="inside",
        verbose=0,
    )
    annotator.apply_and_annotate()

    plt.tight_layout()
    return fig


fig = anat_vs_template(df, figsize=(11, 5))
fig.savefig(
    figures_path / f"boxplot_task_accuracy_{N_PARCELS}.pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()


# %% #######################################################################
def template_vs_pairwise(data: pd.DataFrame, figsize: tuple = (14, 7)):
    data = data.copy()
    data = data[data.target != "template_out_of_sample"]
    fig, ax = plt.subplots(figsize=figsize)
    # Use a better color palette
    palette = sns.color_palette(
        "tab20", n_colors=len(data["solver_target"].unique())
    )
    palette = palette[1:] + palette[:1]
    # Add error bars and improve styling
    ax = sns.barplot(
        data=data,
        x="task_name",
        y="cv_scores",
        hue="solver_target",
        palette=palette,
        errorbar="se",  # or "sd" for standard deviation
        capsize=0.1,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add hatching to Anatomical bars - iterate through all bars
    # Get the order of solvers as they appear in the legend
    _, labels = ax.get_legend_handles_labels()

    anatomical_idx = labels.index("Anatomical")

    # Iterate through containers (each container is one hue/solver)
    for container_idx, container in enumerate(ax.containers):
        if container_idx == anatomical_idx:
            for bar in container:
                bar.set_hatch("///")

    ax.set_xlabel("Task", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Improve legend with hatching for Anatomical
    legend = ax.legend(
        title="Alignment method",
        title_fontsize=11,
        fontsize=10,
        frameon=False,
        shadow=False,
        loc="best",  # anchor point of legend relative to bbox
        bbox_to_anchor=(1.02, 1),  # place it just outside on the right
    )

    # Add hatching to Anatomical legend entry
    for patch, label in zip(legend.get_patches(), legend.get_texts()):
        if label.get_text() == "Anatomical (Pai)":
            patch.set_hatch("///")

    # Add chance levels
    for i, task in enumerate(data["task_name"].unique()):
        chance = data[data["task_name"] == task]["chance_level"].iloc[0]
        ax.hlines(
            chance,
            i - 0.4,
            i + 0.4,
            colors="k",
            linestyles="--",
            alpha=0.7,
            linewidth=2,
            label="Chance level" if i == 0 else "",
        )

    # Add horizontal gridlines
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)

    # Statistical annotations
    from itertools import combinations

    pairs = []

    for task in data["task_name"].unique():
        for solver in data["solver_name"].unique():
            flavors = data.loc[
                (data["task_name"] == task) & (data["solver_name"] == solver),
                "solver_target",
            ].unique()
            if len(flavors) < 2:
                continue  # need at least 2 flavors to form a pair
            pairs.extend(
                [
                    ((task, f1), (task, f2))
                    for f1, f2 in combinations(flavors, 2)
                ]
            )

    annotator = Annotator(
        ax,
        pairs=pairs,
        data=data,
        x="task_name",
        y="cv_scores",
        hue="solver_target",
    )
    annotator.configure(
        test="Wilcoxon",
        text_format="star",
        loc="inside",
        verbose=0,
    )
    annotator.apply_and_annotate()

    plt.tight_layout()
    return fig


fig = template_vs_pairwise(df, figsize=(14, 5))

plt.show()


# %% #######################################################################
def in_vs_out_of_sample(data: pd.DataFrame, figsize: tuple = (14, 7)):
    data = data.copy()
    data = data[
        (data.target == "template_out_of_sample")
        | (data.target == "template_in_sample")
    ]
    fig, ax = plt.subplots(figsize=figsize)
    # Use a better color palette
    palette = sns.color_palette(
        "tab20", n_colors=len(data["solver_target"].unique())
    )
    palette = palette[1:] + palette[:1]
    # Add error bars and improve styling
    ax = sns.barplot(
        data=data,
        x="task_name",
        y="cv_scores",
        hue="solver_target",
        palette=palette,
        errorbar="se",  # or "sd" for standard deviation
        capsize=0.1,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add hatching to Anatomical bars - iterate through all bars
    # Get the order of solvers as they appear in the legend
    _, labels = ax.get_legend_handles_labels()

    anatomical_idx = labels.index("Anatomical")

    # Iterate through containers (each container is one hue/solver)
    for container_idx, container in enumerate(ax.containers):
        if container_idx == anatomical_idx:
            for bar in container:
                bar.set_hatch("///")

    ax.set_xlabel("Task", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Improve legend with hatching for Anatomical
    legend = ax.legend(
        title="Alignment method",
        title_fontsize=11,
        fontsize=10,
        frameon=False,
        shadow=False,
        loc="best",  # anchor point of legend relative to bbox
        bbox_to_anchor=(1.02, 1),  # place it just outside on the right
    )

    # Add hatching to Anatomical legend entry
    for patch, label in zip(legend.get_patches(), legend.get_texts()):
        if label.get_text() == "Anatomical (Pai)":
            patch.set_hatch("///")

    # Add chance levels
    for i, task in enumerate(data["task_name"].unique()):
        chance = data[data["task_name"] == task]["chance_level"].iloc[0]
        ax.hlines(
            chance,
            i - 0.4,
            i + 0.4,
            colors="k",
            linestyles="--",
            alpha=0.7,
            linewidth=2,
            label="Chance level" if i == 0 else "",
        )

    # Add horizontal gridlines
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)

    # Statistical annotations
    from itertools import combinations

    pairs = []

    for task in data["task_name"].unique():
        for solver in data["solver_name"].unique():
            flavors = data.loc[
                (data["task_name"] == task) & (data["solver_name"] == solver),
                "solver_target",
            ].unique()
            if len(flavors) < 2:
                continue  # need at least 2 flavors to form a pair
            pairs.extend(
                [
                    ((task, f1), (task, f2))
                    for f1, f2 in combinations(flavors, 2)
                ]
            )

    annotator = Annotator(
        ax,
        pairs=pairs,
        data=data,
        x="task_name",
        y="cv_scores",
        hue="solver_target",
    )
    annotator.configure(
        test="Wilcoxon",
        text_format="star",
        loc="inside",
        verbose=0,
    )
    annotator.apply_and_annotate()

    plt.tight_layout()
    return fig


fig = in_vs_out_of_sample(df, figsize=(14, 5))

plt.show()
