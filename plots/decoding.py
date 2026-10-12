# %%
import itertools

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
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    },
)

STAR_THRESHOLDS = [[1.0, "ns"], [0.05, "*"], [0.01, "**"], [0.001, "***"]]


# --------------------------------------------------------------------------
# Statistical tests
# --------------------------------------------------------------------------
def corrected_dependent_ttest(data1, data2):
    """Paired t-test with Nadeau & Bengio variance correction."""
    data1, data2 = np.asarray(data1), np.asarray(data2)
    n = len(data1)
    differences = data2 - data1
    sd = np.std(differences, ddof=1)
    test_training_ratio = 1 / n
    denominator = np.sqrt(1 / n + test_training_ratio) * sd
    t_stat = differences.mean() / denominator
    p = 2.0 * t.sf(abs(t_stat), n - 1)
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


def annotate_stats(
    ax, data, pairs, test, y, hue, order, hue_order, plot, loc="inside"
):
    """Run `test` on `pairs`, BH-correct, draw stars, return a stats table.

    Within each (task, group) the rows must be sorted by subject so that
    paired tests compare the same subjects.
    """
    annotator = Annotator(
        ax,
        pairs,
        data=data,
        x="task_name",
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
        plot=plot,
    )
    annotator.configure(
        test=test,
        comparisons_correction="BH",
        hide_non_significant=False,
        correction_format="replace",
        loc=loc,
        verbose=0,
    )

    annotator._pvalue_format.pvalue_thresholds = STAR_THRESHOLDS
    annotator.apply_test().annotate()


def within_solver_pairs(data):
    """Pairs of solver_target labels belonging to the same solver, per task."""
    return [
        ((task, g1), (task, g2))
        for (task, _), sub in data.groupby(["task_name", "solver_name"])
        for g1, g2 in itertools.combinations(
            sorted(sub["solver_target"].unique()), 2
        )
    ]


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def average_folds(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "fold"]
    df = df.groupby(categorical_cols, as_index=False)[numerical_cols].mean()
    df = df.drop("fold", axis=1)
    return df.sort_values(["task_name", "solver_target", "subject"])


def add_common_plot_elements(
    ax, data: pd.DataFrame, add_legend: bool = True, ncols: int = 3
):
    ax.set_xlabel("Task (N Subjects)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Decoding Accuracy", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

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

    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)

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


def create_pointplot(
    data, palette, order, hue_order, y="cv_scores", hue="solver_target"
):
    fig, ax = plt.subplots()
    sns.boxplot(
        data=data,
        x="task_name",
        y=y,
        hue=hue,
        order=order,
        hue_order=hue_order,
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
        order=order,
        hue_order=hue_order,
        dodge=True,
        jitter=False,
        size=4,
        palette=palette,
        alpha=1,
        ax=ax,
        linewidth=0.5,
    )
    return fig, ax


def _orders(data):
    return sorted(data["task_name"].unique()), sorted(
        data["solver_target"].unique()
    )


def _absolute_plot(data, palette, pairs, ncols=3):
    """Absolute accuracies; Nadeau-Bengio corrected paired t-tests."""
    order, hue_order = _orders(data)
    fig, ax = create_pointplot(data, palette, order, hue_order)
    add_common_plot_elements(ax, data, ncols=ncols)
    annotate_stats(
        ax,
        data,
        pairs,
        CorrectedDependentTTest(),
        y="cv_scores",
        hue="solver_target",
        order=order,
        hue_order=hue_order,
        plot="boxplot",
    )
    plt.tight_layout()
    sns.despine(left=True)
    return fig


def _diff_plot(data, pivot, palette, ylabel):
    """Per-subject differences; one-sample t-test of each solver against 0."""
    pivot = pivot.sort_values(["task_name", "solver_name", "subject"])
    solvers = sorted(pivot["solver_name"].unique().tolist())
    tasks = sorted(pivot["task_name"].unique().tolist())
    n = len(solvers)

    fig, ax = plt.subplots()
    sns.pointplot(
        data=pivot,
        x="task_name",
        y="cv_score_diff",
        hue="solver_name",
        order=tasks,
        hue_order=solvers,
        palette=palette,
        # centers match the boxplot slots statannotations assumes
        dodge=0.8 * (n - 1) / n,
        linestyle="none",
        markers="D",
        markersize=2,
        err_kws={"linewidth": 1.5},
        capsize=0.15,
        ax=ax,
        legend=True,
    )

    pairs = [((t_, s), (t_, s)) for t_ in tasks for s in solvers]
    annotate_stats(
        ax,
        pivot,
        pairs,
        OneSampleTTest(),
        y="cv_score_diff",
        hue="solver_name",
        order=tasks,
        hue_order=solvers,
        plot="boxplot",
    )

    reference = "Optimal Transport"
    pairs_ot = [
        ((task, reference), (task, solver))
        for task in tasks
        for solver in solvers
        if solver != reference
    ]

    annotate_stats(
        ax,
        pivot,
        pairs_ot,
        test=CorrectedDependentTTest(),
        y="cv_score_diff",
        hue="solver_name",
        order=tasks,
        hue_order=solvers,
        plot="boxplot",
        loc="outside",
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Task (N Subjects)", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelsize=10)

    for i, _ in enumerate(data["task_name"].unique()):
        plt.axvspan(
            i - 0.5,
            i + 0.5,
            facecolor="gray",
            alpha=[0.05 if i % 2 == 1 else 0][0],
        )

    ax.yaxis.grid(True, linestyle=":", alpha=0.7)
    ax.set_axisbelow(True)
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
    fig.subplots_adjust(bottom=0.3)
    return fig


def anat_vs_template(data, palette):
    data = average_folds(data[data.target == "template_out_of_sample"].copy())
    pairs = [
        ((task, solver), (task, "Anatomical"))
        for task in data["task_name"].unique()
        for solver in data["solver_target"].unique()
        if solver != "Anatomical"
    ]
    return _absolute_plot(data, palette, pairs)


def template_vs_pairwise(data, palette):
    data = data[
        ~data.solver_name.isin(["Anatomical", "Shared Response"])
        & (data.target != "template_in_sample")
    ].copy()
    data = average_folds(data)
    return _absolute_plot(data, palette, within_solver_pairs(data))


def in_vs_out_of_sample(data, palette):
    data = data[
        ~data.solver_name.isin(["Anatomical"])
        & data.target.isin(["template_out_of_sample", "template_in_sample"])
    ].copy()
    data = average_folds(data)
    return _absolute_plot(data, palette, within_solver_pairs(data), ncols=4)


def bias_diff(data, palette):
    data = data[
        ~data.solver_name.isin(["Anatomical"])
        & data.target.isin(["template_out_of_sample", "template_in_sample"])
    ].copy()
    data = average_folds(data)

    pivot = data.pivot_table(
        index=["subject", "task_name", "solver_name"],
        columns="target",
        values="cv_scores",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    pivot["cv_score_diff"] = (
        pivot["template_in_sample"] - pivot["template_out_of_sample"]
    )
    return _diff_plot(data, pivot, palette, "Bias")


def pairwise_diff(data, palette):
    data = data[
        ~data.solver_name.str.startswith(("Anatomical", "Shared Response"))
        & (data.target != "template_in_sample")
    ].copy()
    data["target"] = np.where(
        data["target"] == "template_out_of_sample",
        "template_out_of_sample",
        "pairwise",
    )
    data = average_folds(data)

    pivot = data.pivot_table(
        index=["subject", "task_name", "solver_name"],
        columns="target",
        values="cv_scores",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    pivot["cv_score_diff"] = pivot["pairwise"] - pivot["template_out_of_sample"]
    return _diff_plot(data, pivot, palette, "Accuracy Gap")


if __name__ == "__main__":
    df = get_results_dataframe(DATA_PATH, n_parcels=400)
    dict_palette = create_palette(df)

    plots = [
        (anat_vs_template, "anat_vs_template"),
        (template_vs_pairwise, "template_vs_pairwise"),
        (in_vs_out_of_sample, "in_vs_out_of_sample"),
        (bias_diff, "bias_diff"),
        (pairwise_diff, "pairwise_diff"),
    ]

    for plot_func, name in plots:
        fig = plot_func(df, palette=dict_palette)
        fig.savefig(FIGURES_PATH / f"{name}.pdf", bbox_inches="tight")
