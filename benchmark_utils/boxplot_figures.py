# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from joblib import load

plt.rcParams["figure.dpi"] = 300

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

N_PARCELS = 400


def get_results_dataframe(
    data_path: Path, score="cv_scores_classif", n_parcels=400
) -> pd.DataFrame:
    # Glob recursively all the decoding_results.pkl files
    results_paths = glob.glob(
        str(data_path / "**" / "decoding_results.pkl"), recursive=True
    )
    # Build a dataframe with all the results
    res_list = []
    for path in results_paths:
        path = Path(path)
        solver = path.parent.name
        results = load(path)
        # Add the solver name to the results
        results["solver_name"] = solver
        res_list.append(results)

    df = pd.DataFrame(res_list)

    # Remove the simulated data
    df = df[~df["dataset_name"].str.contains("Simulated")]

    # Keep only the results for the specified number of parcels
    # df = df[df["dataset_name"].str.contains(f"{n_parcels}")]

    # Remove parcels numbers from the dataset names
    df["dataset_name"] = df["dataset_name"].str.replace(
        r"_[0-9]+$", "", regex=True
    )

    # Rename Wm by WM
    # df["task_name"] = df["task_name"].str.replace("Wm", "WM")

    # Sort alphabetically by dataset name and solver name
    df.sort_values(by=["task_name", "solver_name"], inplace=True)

    # Fix underscores in the solver names
    df["solver_name"] = df["solver_name"].str.replace("_", " ")

    df = df.explode("cv_scores")

    return df


df = get_results_dataframe(data_path)


# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)


def create_accuracy_plot(data, figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    original_palette = sns.color_palette("Paired")
    shifted_palette = original_palette[1:] + original_palette[:1]
    sns.barplot(
        data=data,
        y="cv_scores",
        x="task_name",
        hue="solver_name",
        ax=ax,
        palette=shifted_palette,
    )

    ax.set_xlabel("Task", fontweight="bold")
    ax.set_ylabel("Accuracy", fontweight="bold")

    for i in range(len((data["task_name"].unique()))):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.05)

    # Add a vertical line at chance level
    chance_levels = (
        data.groupby("task_name")["chance_level"]
        .unique()
        .apply(lambda x: x[0])
    ).to_list()
    for i, chance_level in enumerate(chance_levels):
        ax.hlines(
            chance_level,
            i - 0.5,
            i + 0.5,
            color="k",
            linestyles="--",
            alpha=0.5,
        )

    ax.set_ylim(0, 1.05)
    ax.legend(title="Alignment method", title_fontsize="large")
    sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    return fig


fig = create_accuracy_plot(df, figsize=(11, 3))
# fig1.savefig(figures_path / "boxplot_movie_accuracy.pdf", bbox_inches="tight")
fig.savefig(
    figures_path / f"boxplot_task_accuracy_{N_PARCELS}.pdf",
    bbox_inches="tight",
    dpi=300,
)

plt.show()
