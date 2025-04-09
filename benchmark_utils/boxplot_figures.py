# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from joblib import load

plt.rcParams["figure.dpi"] = 500

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
        dataset = path.parent.parent.parent.name
        solver = path.parent.parent.name
        target = path.parent.name
        results = load(path)
        # Add the dataset name and solver name to the results
        results["data_name"] = dataset
        results["solver_name"] = solver
        results["target"] = target
        res_list.append(results)

    df = pd.DataFrame(res_list)

    # Remove the simulated data
    df = df[~df["data_name"].str.contains("Simulated")]
    
    # Keep only the results for the specified number of parcels
    df = df[df["data_name"].str.contains(f"{n_parcels}")]

    # Remove parcels numbers from the dataset names
    df["data_name"] = df["data_name"].str.replace(
        r"_[0-9]+$", "", regex=True
    )

    # Remove the SparseOT solver
    df = df[~df["solver_name"].str.contains("Sparse")]

    # For anatomical keep only the template target
    df = df[
        ~((df["solver_name"] == "Anatomical") & (df["target"] == "template"))
    ]

    # Remove the "IBC " prefix on the dataset names
    df["data_name"] = df["data_name"].str.replace("IBC", "")

    # Add a type column to indicate tasks or movie
    df["type"] = df["data_name"].apply(
        lambda x: "movie"
        if x.lower().startswith("raiders") or x.lower().startswith("budapest")
        else "task"
    )

    # For datasets split in runs, remove the "_run-0*" suffix
    df["data_name"] = df["data_name"].str.replace(r"_run-\d+", "", regex=True)

    # Add (template) to the solver name if target is template
    df["solver_name"] = df.apply(
        lambda x: x["solver_name"] + " (template)"
        if x["target"] == "template"
        else x["solver_name"] + " (pairwise)",
        axis=1,
    )

    # Remove the (template) suffix for the anatomical alignment
    df["solver_name"] = df["solver_name"].str.replace(
        "Anatomical (pairwise)", "Anatomical"
    )

    # Rename ot by Optimal Transport
    df["solver_name"] = df["solver_name"].str.replace(
        "ot", "Optimal Transport"
    )

    # Rename Wm by WM
    df["data_name"] = df["data_name"].str.replace("Wm", "WM")

    # Sort alphabetically by dataset name and solver name
    df.sort_values(by=["data_name", "solver_name"], inplace=True)

    # Fix underscores in the solver names
    df["solver_name"] = df["solver_name"].str.replace("_", " ")

    # Expand the lists in df[score]
    df = df.explode(score)

    return df


df = get_results_dataframe(data_path, score="cv_scores_classif", n_parcels=N_PARCELS)

# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)


def create_accuracy_plot(data, figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    original_palette = sns.color_palette("Paired")
    shifted_palette = original_palette[1:] + original_palette[:1]
    sns.barplot(
        data=data,
        x="cv_scores_classif",
        y="data_name",
        hue="solver_name",
        ax=ax,
        palette=shifted_palette,
    )

    ax.set_xlabel("Accuracy", fontweight="bold")
    ax.set_ylabel("Dataset", fontweight="bold")

    for i in range(len((data["data_name"].unique()))):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.05)

    # Add a vertical line at chance level
    chance_levels = (
        data.groupby("data_name")["chance_level"]
        .unique()
        .apply(lambda x: x[0])
    ).to_list()
    for i, chance_level in enumerate(chance_levels):
        ax.vlines(
            chance_level,
            i - 0.5,
            i + 0.5,
            color="k",
            linestyles="--",
            alpha=0.5,
        )

    ax.set_yticklabels(
        [name.replace("_", " ") for name in data["data_name"].unique()]
    )
    ax.legend(title="Alignment method", title_fontsize="large")
    sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    return fig


# Create separate plots
# movie_data = df[df["type"] == "movie"]
task_data = df[df["type"] == "task"]

# fig1 = create_accuracy_plot(
#     movie_data,
#     "Movie Prediction Accuracies",
# )
fig2 = create_accuracy_plot(task_data, figsize=(3.15, 8))
# fig1.savefig(figures_path / "boxplot_movie_accuracy.pdf", bbox_inches="tight")
fig2.savefig(figures_path / f"boxplot_task_accuracy_{N_PARCELS}.pdf", bbox_inches="tight")

plt.show()
# %%
# Do the same for the Pearson correlation
df = get_results_dataframe(data_path, score="pearson_corrs")

# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)

# Create the figure and axes with a specific size
fig, ax = plt.subplots(
    figsize=(4.25, 6), constrained_layout=True
)  # Increased height

# Create the box plot
sns.boxplot(
    data=df,
    x="pearson_corrs",
    y="data_name",
    hue="solver_name",
    showfliers=False,
    ax=ax,
    fill=False,
    legend=False,
    palette="dark:k",
)

original_palette = sns.color_palette("Paired")
shifted_palette = original_palette[1:] + original_palette[:1]

# Create the scatter plot
sns.stripplot(
    data=df,
    x="pearson_corrs",
    y="data_name",
    size=4,
    hue="solver_name",
    dodge=True,
    jitter=True,
    palette=shifted_palette,
)

# Customize the plot
ax.set_xlabel("Pearson Correlation", fontweight="bold")
ax.set_ylabel("Dataset", fontweight="bold")

# Set and move the legend ABOVE the figure (not overlapping)
legend = ax.legend(
    title="Alignment method",
    title_fontsize="large",
    loc="lower center",
    bbox_to_anchor=(0.3, -0.7),
    frameon=False,
)

# Add gray rectangles to separate the datasets
for i, data_name in enumerate(df["data_name"].unique()):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.05)

# Fix underscores in the dataset names
ax.set_yticklabels(
    [name.replace("_", " ") for name in df["data_name"].unique()]
)

# Save the figure with high resolution
plt.savefig(
    figures_path / f"boxplot_correlation_{N_PARCELS}.pdf", dpi=500, bbox_inches="tight"
)

# Display the plot
plt.show()
