# %%
from itertools import combinations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t, ttest_1samp
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
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


def average_folds(df: pd.DataFrame):
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = df.select_dtypes(include="number").columns.tolist()

    # Remove 'fold' from the grouping columns if it's categorical
    categorical_cols = [c for c in categorical_cols if c != "fold"]

    # Group by all categorical columns except 'fold' and average the numerical ones
    df = df.groupby(categorical_cols, as_index=False)[numerical_cols].mean()
    df = df.drop("fold", axis=1)
    return df.sort_values(["task_name", "solver_target"])


def add_common_plot_elements(
    ax, data: pd.DataFrame, add_legend: bool = True, ncols: int = 3
):
    """Add common elements to plots (chance levels, grid, labels)."""
    ax.set_xlabel("Task (N Subjects)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Decoding Accuracy", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Add chance levels and rectangles for separation
    for i, task in enumerate(data["task_name"].unique()):
        chance = data[data["task_name"] == task]["chance_level"].iloc[0]
        ax.hlines(
            chance,
            i - 0.45,
            i + 0.45,
            colors="k",
            linestyles="--",
            alpha=0.75,
            linewidth=2,
        )
        plt.axvspan(
            i - 0.5,
            i + 0.5,
            facecolor="gray",
            alpha=[0.05 if i % 2 == 1 else 0][0],
        )

    # Add gridlines
    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)

    # Add legend
    if add_legend:
        ax.legend(
            title="Alignment method",
            title_fontsize=11,
            fontsize=10,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.4),
            ncol=ncols,
        )


def corrected_dependent_ttest(data1, data2):
    n = len(data1)
    differences = np.array(data1) - np.array(data2)
    sd = np.std(differences)
    divisor = 1 / n * sum(differences)
    test_training_ratio = 1 / n
    denominator = np.sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    df = n - 1
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, p


class CorrectedDependentTTest(StatTest):
    def __init__(self):
        super().__init__(
            func=corrected_dependent_ttest,
            test_long_name="Corrected Dependent t-test",
            test_short_name="Corrected t-test",
            stat_name="t",
            alpha=0.05,
        )


class OneSampleTTest(StatTest):
    @staticmethod
    def statannotations_to_scipy_ttest_1samp(group1, group2, **stats_params):
        return ttest_1samp(group1, popmean=0, **stats_params)

    def __init__(self):
        super().__init__(
            func=self.statannotations_to_scipy_ttest_1samp,
            test_long_name="One-sample t-test",
            test_short_name="One-sample t-test",
            stat_name="t",
            alpha=0.05,
        )


def add_statistical_annotations(
    ax, data: pd.DataFrame, pairs: list, hide_ns: bool = False
):
    """Add statistical significance annotations."""
    annotator = Annotator(
        ax,
        pairs=pairs,
        data=data,
        x="task_name",
        y="cv_scores",
        hue="solver_target",
    )
    annotator._pvalue_format.pvalue_thresholds = [
        [0.001, "***"],
        [0.01, "**"],
        [0.05, "*"],
        [1, "ns"],
    ]
    annotator.configure(
        test=CorrectedDependentTTest(),
        text_format="star",
        loc="inside",
        verbose=0,
    )
    annotator.apply_and_annotate()


def create_barplot(
    data: pd.DataFrame,
    palette: dict,
    y: str = "cv_scores",
    hue: str = "solver_target",
):
    """Create a styled barplot."""
    fig, ax = plt.subplots()
    sns.boxplot(
        data=data,
        x="task_name",
        y=y,
        hue=hue,
        showmeans=True,
        dodge=True,
        palette=palette,
        meanline=True,
        meanprops={"color": "k", "ls": "-", "lw": 1},
        medianprops={"visible": False},
        whiskerprops={"visible": False},
        zorder=10,
        showfliers=False,
        showbox=False,
        showcaps=False,
        linewidth=1,
        fill=False,
        ax=ax,
        legend=False,
    )
    sns.stripplot(
        data=data,
        x="task_name",
        y=y,
        hue=hue,
        dodge=True,
        jitter=False,
        size=4,
        palette=palette,
        alpha=1,
        ax=ax,
        linewidth=0.5,
    )

    return fig, ax


def anat_vs_template(
    data: pd.DataFrame,
    palette: dict,
):
    """Compare Anatomical alignment vs template-based methods."""
    data = data[data.target == "template_out_of_sample"].copy()
    data = average_folds(data)
    fig, ax = create_barplot(data, palette)

    add_common_plot_elements(ax, data)

    # Statistical annotations: Anatomical vs all others
    pairs = [
        ((task, "Anatomical"), (task, solver))
        for task in data["task_name"].unique()
        for solver in data["solver_target"].unique()
        if solver != "Anatomical"
    ]
    add_statistical_annotations(ax, data, pairs, hide_ns=True)

    plt.tight_layout()
    sns.despine(left=True)
    return fig


def template_vs_pairwise(
    data: pd.DataFrame,
    palette: dict,
):
    """Compare template-based vs pairwise alignment."""
    data = data[
        ~data.solver_name.isin(["Anatomical", "Shared Response"])
        & (data.target != "template_in_sample")
    ].copy()
    data = average_folds(data)

    fig, ax = create_barplot(data, palette)
    add_common_plot_elements(ax, data)

    # Statistical annotations: within-solver comparisons
    pairs = [
        ((task, f1), (task, f2))
        for task in data["task_name"].unique()
        for solver in data["solver_name"].unique()
        for f1, f2 in combinations(
            data.loc[
                (data["task_name"] == task) & (data["solver_name"] == solver),
                "solver_target",
            ].unique(),
            2,
        )
    ]
    add_statistical_annotations(ax, data, pairs)

    sns.despine(left=True)
    plt.tight_layout()
    return fig


def in_vs_out_of_sample(
    data: pd.DataFrame,
    palette: dict,
):
    """Compare in-sample vs out-of-sample template alignment."""
    data = data[
        ~data.solver_name.isin(["Anatomical"])
        & data.target.isin(["template_out_of_sample", "template_in_sample"])
    ].copy()
    data = average_folds(data)

    fig, ax = create_barplot(data, palette)
    add_common_plot_elements(ax, data, ncols=4)

    # Statistical annotations: within-solver comparisons
    pairs = [
        ((task, f1), (task, f2))
        for task in data["task_name"].unique()
        for solver in data["solver_name"].unique()
        for f1, f2 in combinations(
            data.loc[
                (data["task_name"] == task) & (data["solver_name"] == solver),
                "solver_target",
            ].unique(),
            2,
        )
    ]
    add_statistical_annotations(ax, data, pairs)

    plt.tight_layout()
    sns.despine(left=True)
    return fig


def bias_diff(
    data: pd.DataFrame,
    palette: dict,
):
    """Compare in-sample vs out-of-sample template alignment."""
    data = data[
        ~data.solver_name.isin(["Anatomical"])
        & data.target.isin(["template_out_of_sample", "template_in_sample"])
    ].copy()
    data = average_folds(data)

    # Pivot so each target becomes its own column
    pivot = data.pivot_table(
        index=["subject", "task_name", "solver_name"],
        columns="target",
        values="cv_scores",
        aggfunc="mean",  # in case of duplicates
    ).reset_index()

    pivot.columns.name = None  # clean up column name

    # Compute the difference: in_sample - out_of_sample
    pivot["cv_score_diff"] = (
        pivot["template_in_sample"] - pivot["template_out_of_sample"]
    )

    # Sort
    pivot = pivot.sort_values(["task_name", "solver_name"])
    solvers = sorted(pivot["solver_name"].unique().tolist())
    tasks = pivot["task_name"].unique().tolist()

    fig, ax = plt.subplots()
    sns.pointplot(
        data=pivot,
        x="task_name",
        y="cv_score_diff",
        hue="solver_name",
        palette=palette,
        dodge=0.6,
        join=False,
        markers="D",
        markersize=2,
        errwidth=1.5,
        capsize=0.15,
        ax=ax,
        legend=True,
    )

    pairs = [
        ((task, solver), (task, solver)) for task in tasks for solver in solvers
    ]

    annotator = Annotator(
        ax,
        pairs=pairs,
        data=pivot,
        x="task_name",
        y="cv_score_diff",
        hue="solver_name",
    )
    annotator.configure(
        test=OneSampleTTest(),
        text_format="star",
        loc="inside",
        verbose=0,
    )
    annotator._pvalue_format.pvalue_thresholds = [
        [0.001, "***"],
        [0.01, "**"],
        [0.05, "*"],
        [1, "ns"],
    ]
    annotator.apply_and_annotate()

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Task (N Subjects)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bias", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelsize=10)

    # Add chance levels and rectangles for separation
    for i, task in enumerate(data["task_name"].unique()):
        plt.axvspan(
            i - 0.5,
            i + 0.5,
            facecolor="gray",
            alpha=[0.05 if i % 2 == 1 else 0][0],
        )

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

    plt.tight_layout()
    sns.despine(left=True)
    return fig, annotator


# Main execution
df = get_results_dataframe(DATA_PATH, n_parcels=400)
dict_palette = create_palette(df)

# Generate and save all plots
plots = [
    (anat_vs_template, "anat_vs_template.pdf"),
    (template_vs_pairwise, "template_vs_pairwise.pdf"),
    (in_vs_out_of_sample, "in_vs_out_of_sample.pdf"),
    (bias_diff, "bias_diff.pdf"),
]

for plot_func, filename in plots:
    fig, annot = plot_func(df, palette=dict_palette)
    fig.savefig(FIGURES_PATH / filename, bbox_inches="tight")
    plt.show()
