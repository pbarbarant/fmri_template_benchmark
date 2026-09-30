# %%
import urllib.request
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from matplotlib import ticker as mtick
from matplotlib.colors import to_rgb
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch
from scipy.stats import false_discovery_control, permutation_test
from tqdm import tqdm
from utils import DATA_PATH, FIGURES_PATH

sns.set_theme(
    context="paper",
    style="white",
    rc={
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,  # embed TrueType: text stays editable in Illustrator/Inkscape
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    },
)

N_SUBJECTS = [10, 20, 50, 100]
N_MOVIES = [1, 2, 3, 4]
RNG_SEED = 0

BRAIN_ZOOM = 0.36  # was 0.25
BRAIN_SPACING = 0.15  # was 0.12 (scaled with the larger icons)

brain_icon_path = Path("brain_emoji.png")
if not brain_icon_path.exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f9e0.png",
        brain_icon_path,
    )
brain_img = mpimg.imread(brain_icon_path)


# ----------------------------------------------------------------- data
def get_hcp_dataframe(data_path):
    paths = list((data_path / "HCP").glob("hcp_*/**/decoding_results.pkl"))
    df = pd.concat([pd.DataFrame(load(p)) for p in tqdm(paths)])

    df["solver_name"] = (
        df["solver_name"]
        .str.replace(r"_[0-9]+$", "", regex=True)
        .str.replace("ot", "OT")
        .str.replace("_", " ")
    )
    df = df[df["solver_name"] != "Ridge"]
    df = df[df["target"] == "template_in_sample"]

    cfg = df["task_name"].str.extract(r"hcp_([0-9]+)_([0-9]+)").astype(int)
    df["n_subjects"], df["n_movies"] = cfg[0], cfg[1]
    return df


# ----------------------------------------------------------------- stats
def stars_from_p(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def winner_vs_anatomical(df):
    """Per (n_subjects, n_movies) cell: best method by mean accuracy, then a
    paired sign-permutation test of (winner - Anatomical) across subjects
    (same test as anat_vs_template), BH-corrected across the 16 cells."""
    subj = (
        df.groupby(["n_subjects", "n_movies", "solver_name", "subject"])[
            "cv_scores"
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for (ns, nm), g in subj.groupby(["n_subjects", "n_movies"]):
        wide = g.pivot(
            index="subject", columns="solver_name", values="cv_scores"
        )
        means = wide.mean()
        winner = means.idxmax()
        row = dict(
            n_subjects=ns,
            n_movies=nm,
            winner=winner,
            score=means.max(),
            p=np.nan,
        )
        if winner != "Anatomical":
            d = (wide[winner] - wide["Anatomical"]).dropna().to_numpy()
            row["n"] = len(d)
            row["mean_diff"] = d.mean()
            row["p"] = permutation_test(
                (d,),
                lambda x, axis: x.mean(axis=axis),
                vectorized=True,
                permutation_type="samples",
                n_resamples=9999,  # exact when 2**n < 9999
                alternative="two-sided",
                rng=RNG_SEED,
            ).pvalue
        rows.append(row)

    res = pd.DataFrame(rows)
    ok = res["p"].notna()
    res["p_fdr"] = np.nan
    res.loc[ok, "p_fdr"] = false_discovery_control(
        res.loc[ok, "p"], method="bh"
    )
    # "n.s." for tested-but-not-significant cells, blank for the Anatomical winner
    res["stars"] = res["p_fdr"].map(
        lambda p: "" if pd.isna(p) else (stars_from_p(p) or "n.s.")
    )
    return res


# ----------------------------------------------------------------- plot helpers
def add_panel_label(ax, label, x=-0.02, y=1.0):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def add_brain_icon(
    ax, zoom=BRAIN_ZOOM, y_offset=1.015, n_brains=1, spacing=BRAIN_SPACING
):
    start_x = 0.5 - spacing * (n_brains - 1) / 2
    for i in range(n_brains):
        ax.add_artist(
            AnnotationBbox(
                OffsetImage(brain_img, zoom=zoom, interpolation="lanczos"),
                (start_x + i * spacing, y_offset),
                xycoords="axes fraction",
                frameon=False,
                box_alignment=(0.5, 0),
                annotation_clip=False,
            )
        )


def add_zebra_stripes(ax, y_min=0.0, y_max=1.0, step=0.05):
    bands = np.arange(y_min, y_max + step, step)
    for i, y0 in enumerate(bands[:-1]):
        ax.axhspan(
            y0,
            bands[i + 1],
            color="white" if i % 2 == 0 else "lightgray",
            alpha=0.22,
            linewidth=0,
            zorder=0,
        )


def text_color(bg):
    r, g, b = to_rgb(bg)
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


def draw_lineplots(fig, subgs, df, hue_order, palette):
    axes = []
    for i, (ns, spec) in enumerate(zip(N_SUBJECTS, subgs)):
        ax = fig.add_subplot(spec, sharey=axes[0] if axes else None)
        axes.append(ax)
        add_zebra_stripes(ax)
        sns.lineplot(
            data=df[df["n_subjects"] == ns],
            x="n_movies",
            y="cv_scores",
            hue="solver_name",
            hue_order=hue_order,
            palette=palette,
            marker="o",
            markersize=7.5,
            markeredgecolor="white",
            markeredgewidth=1.0,
            linewidth=2.2,
            errorbar=None,
            legend=False,
            ax=ax,
            zorder=3,
        )
        ax.set_title(
            f"N = {ns}",
            fontsize=13,
            fontweight="bold",
            pad=36,  # leaves room for the (larger) brain icons under the title
        )
        add_brain_icon(ax, n_brains=i + 1)
        ax.set_ylim(0.65, 1.0)
        ax.set_xlim(0.5, 4.5)
        ax.set_xticks(N_MOVIES)
        ax.tick_params(axis="x", labelsize=11, length=3)
        ax.set_xlabel("Number of movies", fontsize=12)
        if i == 0:
            ax.set_ylabel("Decoding accuracy", fontsize=12)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            ax.set_yticks([0.65 + 0.05 * k for k in range(8)])
            ax.tick_params(axis="y", labelsize=11, length=3)
            sns.despine(ax=ax, top=True, right=True)
        else:
            ax.set_ylabel("")
            ax.tick_params(left=False, labelleft=False)
            sns.despine(ax=ax, left=True, top=True, right=True)
    return axes


def draw_grid(ax, res, palette):
    ax.set_facecolor("white")
    pad = 0.05
    for _, r in res.iterrows():
        i, j = N_SUBJECTS.index(r.n_subjects), N_MOVIES.index(r.n_movies)
        c = palette[r.winner]
        tc = text_color(c)
        ax.add_patch(
            FancyBboxPatch(
                (j + pad, i + pad),
                1 - 2 * pad,
                1 - 2 * pad,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=c,
                edgecolor="white",
                linewidth=1.5,
                alpha=0.92,
                zorder=2,
                path_effects=[
                    pe.withSimplePatchShadow(offset=(1, -1), alpha=0.12)
                ],
            )
        )
        ax.text(
            j + 0.5,
            i + 0.65,
            r.winner,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=tc,
            zorder=3,
        )
        ax.text(
            j + 0.5,
            i + 0.44,
            f"{r.score:.1%}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=tc,
            alpha=0.9,
            zorder=3,
        )
        is_ns = r.stars == "n.s."
        ax.text(
            j + 0.5,
            i + 0.22,
            r.stars,
            ha="center",
            va="center",
            fontsize=8.5 if is_ns else 13,
            fontweight="normal" if is_ns else "bold",
            color=tc,
            alpha=0.75 if is_ns else 1.0,
            zorder=3,
        )

    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xticks(np.arange(4) + 0.5, N_MOVIES, fontsize=11)
    ax.set_yticks(np.arange(4) + 0.5, N_SUBJECTS, fontsize=11)
    ax.set_xlabel("Number of movies", fontsize=12, labelpad=8)
    ax.set_ylabel("Number of subjects", fontsize=12, labelpad=8)
    ax.set_title(
        "Best-performing alignment method",
        fontsize=13,
        fontweight="bold",
        pad=36,
    )
    ax.text(
        0.5,
        1.035,
        "Stars: winner vs. anatomical (paired permutation test, BH-corrected)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
        style="italic",
    )
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False)
    sns.despine(ax=ax, left=True, bottom=True)


# ----------------------------------------------------------------- main
def main():
    df = get_hcp_dataframe(DATA_PATH)
    res = winner_vs_anatomical(df)

    # palette unchanged: same seaborn default colours, same sorted hue order
    hue_order = sorted(df["solver_name"].unique())
    palette = dict(zip(hue_order, sns.color_palette(n_colors=len(hue_order))))

    fig = plt.figure(figsize=(18, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.17)
    left = gs[0].subgridspec(1, 4, wspace=0.08)

    line_axes = draw_lineplots(
        fig, [left[0, k] for k in range(4)], df, hue_order, palette
    )
    grid_ax = fig.add_subplot(gs[1])
    draw_grid(grid_ax, res, palette)

    add_panel_label(line_axes[0], "a", x=-0.28, y=1.22)
    add_panel_label(grid_ax, "b", x=-0.14, y=1.22)

    handles = [
        plt.Line2D(
            [],
            [],
            color=palette[s],
            marker="o",
            markersize=7.5,
            markeredgecolor="white",
            linewidth=2.2,
            label=s,
        )
        for s in hue_order
    ]
    fig.legend(
        handles=handles,
        title="Alignment method",
        title_fontsize=12,
        fontsize=11,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(hue_order),
        columnspacing=2.0,
    )

    fig.savefig(FIGURES_PATH / "HCP_summary.pdf", dpi=300, bbox_inches="tight")
    res.to_csv(FIGURES_PATH / "HCP_summary_stats.csv", index=False)
    plt.show()


if __name__ == "__main__":
    main()
