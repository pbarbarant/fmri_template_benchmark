# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import get_results_dataframe, create_palette, DATA_PATH, FIGURES_PATH

sns.set_theme(
    context="paper",
    style="ticks",
    rc={
        "figure.figsize": [7, 6.5],
        "text.usetex": False,
        "font.family": "sans-serif",
        "savefig.dpi": 300,
    },
)


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
    plt.setp(ax.get_xticklabels(), ha="right")
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
df = get_results_dataframe(DATA_PATH, n_parcels=400)
dict_palette = create_palette(df)

# Generate and save all plots
fig = time_comparison(df, palette=dict_palette)
fig.savefig(FIGURES_PATH / "time_comparison.pdf", bbox_inches="tight")
plt.show()
