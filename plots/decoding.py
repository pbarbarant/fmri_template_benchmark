# %%
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import false_discovery_control, permutation_test
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

RNG_SEED = 0


def stars_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""  # dont display ns results


def paired_contrasts(data, value, group, contrasts):
    """
    contrasts: list of (task, g1, g2). Per subject, d = value[g1] - value[g2]
    (or d = value[g1] if g2 is None, i.e. a one-sample test against 0).
    The sign-permutation test is run on d; p-values are BH-corrected across
    all contrasts of the figure.
    """
    wide = data.pivot_table(
        index=["subject", "task_name"],
        columns=group,
        values=value,
        aggfunc="mean",
    )
    rows = []
    for task, g1, g2 in contrasts:
        sub = wide.xs(task, level="task_name")
        d = sub[g1] if g2 is None else sub[g1] - sub[g2]
        d = d.dropna().to_numpy()
        if len(d) < 2:
            continue
        result = permutation_test(
            (d,),
            lambda x, axis: x.mean(axis=axis),
            vectorized=True,
            permutation_type="samples",
            n_resamples=np.inf,
            alternative="two-sided",
            rng=RNG_SEED,
        )
        rows.append(
            {
                "task_name": task,
                "g1": g1,
                "g2": g2,
                "n": len(d),
                "mean_diff": d.mean(),
                "p": result.pvalue,
            }
        )
    res = pd.DataFrame(rows)
    if res.empty:
        return res.assign(p_fdr=[], stars=[])
    res["p_fdr"] = false_discovery_control(res["p"].to_numpy(), method="bh")
    res["stars"] = res["p_fdr"].map(stars_from_p)
    return res


def within_solver_contrasts(data):
    return [
        (task, g1, g2)
        for (task, _), sub in data.groupby(["task_name", "solver_name"])
        for g1, g2 in itertools.combinations(
            sorted(sub["solver_target"].unique()), 2
        )
    ]


def annotate_stars(
    ax,
    res,
    data,
    y,
    group,
    order,
    hue_order,
    spread=0.8,
    mode="slots",
    top="raw",
):
    """
    One-sample contrasts (g2 None): stars above the group.
    Pairwise contrasts: bracket + stars above the task's data.
    mode: 'slots' (box/strip dodge) or 'spread' (pointplot dodge)
    top:  'raw' (max of points) or 'ci' (mean + 1.96 SEM, for pointplots)
    """
    res = res[res["stars"] != ""]
    if res.empty:
        return
    n = len(hue_order)
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    step = 0.06 * span

    def xpos(task, g):
        i, j = order.index(task), hue_order.index(g)
        if mode == "slots":
            return i - spread / 2 + spread / n * (j + 0.5)
        return i - spread / 2 + (spread * j / (n - 1) if n > 1 else spread / 2)

    def group_top(task, g):
        v = data.loc[
            (data["task_name"] == task) & (data[group] == g), y
        ].dropna()
        if v.empty:
            return -np.inf
        return (
            v.max()
            if top == "raw"
            else v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        )

    ymax = ax.get_ylim()[1]
    for task in order:
        r = res[res["task_name"] == task]
        if r.empty:
            continue
        tops = [group_top(task, g) for g in hue_order]
        base = max(tops) + 0.05 * span
        level = 0
        for _, row in r.iterrows():
            if pd.isna(row["g2"]):
                y_txt = group_top(task, row["g1"]) + 0.02 * span
                ax.text(
                    xpos(task, row["g1"]),
                    y_txt,
                    row["stars"],
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )
                ymax = max(ymax, y_txt + step)
            else:
                x1, x2 = xpos(task, row["g1"]), xpos(task, row["g2"])
                yb = base + level * step
                tick = 0.015 * span
                ax.plot(
                    [x1, x1, x2, x2],
                    [yb - tick, yb, yb, yb - tick],
                    color="k",
                    lw=1,
                )
                ax.text(
                    (x1 + x2) / 2,
                    yb,
                    row["stars"],
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    fontweight="bold",
                )
                ymax = max(ymax, yb + step)
                level += 1
    ax.set_ylim(top=ymax)


def average_folds(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "fold"]
    df = df.groupby(categorical_cols, as_index=False)[numerical_cols].mean()
    df = df.drop("fold", axis=1)
    return df.sort_values(["task_name", "solver_target"])


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


def _absolute_plot(data, palette, contrasts, ncols=3):
    """Plot absolute accuracies; stats on subject-wise paired differences."""
    order, hue_order = _orders(data)
    fig, ax = create_pointplot(data, palette, order, hue_order)
    add_common_plot_elements(ax, data, ncols=ncols)
    res = paired_contrasts(data, "cv_scores", "solver_target", contrasts)
    annotate_stars(
        ax, res, data, "cv_scores", "solver_target", order, hue_order
    )
    plt.tight_layout()
    sns.despine(left=True)
    return fig, res


def _diff_plot(data, pivot, palette, ylabel):
    """Pointplot of per-subject differences; stats on the differences."""
    pivot = pivot.sort_values(["task_name", "solver_name"])
    solvers = sorted(pivot["solver_name"].unique().tolist())
    tasks = sorted(pivot["task_name"].unique().tolist())

    fig, ax = plt.subplots()
    sns.pointplot(
        data=pivot,
        x="task_name",
        y="cv_score_diff",
        hue="solver_name",
        order=tasks,
        hue_order=solvers,
        palette=palette,
        dodge=0.6,
        linestyle="none",
        markers="D",
        markersize=2,
        err_kws={"linewidth": 1.5},
        capsize=0.15,
        ax=ax,
        legend=True,
    )

    # 1 each solver's difference vs 0; 2 Optimal Transport vs other solvers
    contrasts = [(t, s, None) for t in tasks for s in solvers]
    if "Optimal Transport" in solvers:
        contrasts += [
            (t, "Optimal Transport", s)
            for t in tasks
            for s in solvers
            if s != "Optimal Transport"
        ]
    res = paired_contrasts(pivot, "cv_score_diff", "solver_name", contrasts)
    annotate_stars(
        ax,
        res,
        pivot,
        "cv_score_diff",
        "solver_name",
        tasks,
        solvers,
        spread=0.6,
        mode="spread",
        top="ci",
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
    return fig, res


def anat_vs_template(data, palette):
    data = average_folds(data[data.target == "template_out_of_sample"].copy())
    contrasts = [
        (task, solver, "Anatomical")
        for task in data["task_name"].unique()
        for solver in data["solver_target"].unique()
        if solver != "Anatomical"
    ]
    return _absolute_plot(data, palette, contrasts)


def template_vs_pairwise(data, palette):
    data = data[
        ~data.solver_name.isin(["Anatomical", "Shared Response"])
        & (data.target != "template_in_sample")
    ].copy()
    data = average_folds(data)
    return _absolute_plot(data, palette, within_solver_contrasts(data))


def in_vs_out_of_sample(data, palette):
    data = data[
        ~data.solver_name.isin(["Anatomical"])
        & data.target.isin(["template_out_of_sample", "template_in_sample"])
    ].copy()
    data = average_folds(data)
    return _absolute_plot(data, palette, within_solver_contrasts(data), ncols=4)


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
        fig, stats_table = plot_func(df, palette=dict_palette)
        fig.savefig(FIGURES_PATH / f"{name}.pdf", bbox_inches="tight")
        stats_table.to_csv(FIGURES_PATH / f"{name}_stats.csv", index=False)
