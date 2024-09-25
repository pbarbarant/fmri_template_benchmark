# %%
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

plt.rcParams["figure.dpi"] = 500

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
# Parse the latest file
file_list = glob.glob(os.path.join(data_path, "*.parquet"))
latest_file = max(file_list, key=os.path.getmtime)
df = pd.read_parquet(latest_file)

# Remove the simulated data
df.drop(df[df["data_name"].str.contains("Simulated")].index, inplace=True)

# Merge all BOLD5000 folds into one
df.loc[df["data_name"].str.contains("BOLD5000"), "data_name"] = "BOLD5000"

# Expand the lists in df["objective_cv_scores"]
df = df.explode("objective_cv_scores")

# Sort alphabetically by dataset name and solver name
df.sort_values(by=["data_name", "solver_name"], inplace=True)

# Fix underscores in the solver names
df["solver_name"] = df["solver_name"].str.replace("_", " ")

# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)

# Create the figure and axes with a specific size
fig, ax = plt.subplots(figsize=(12, 5))

# Create the box plot
sns.boxplot(
    data=df,
    x="objective_cv_scores",
    y="data_name",
    hue="solver_name",
    showfliers=False,
    ax=ax,
    fill=False,
    legend=False,
    color="k",
)
# Create the scatter plot
sns.stripplot(
    data=df,
    x="objective_cv_scores",
    y="data_name",
    size=4,
    hue="solver_name",
    dodge=True,
    jitter=True,
)

# Customize the plot
ax.set_xlabel("Accuracy", fontweight="bold")
ax.set_ylabel("Dataset", fontweight="bold")
ax.set_title(
    "Prediction accuracies for various template estimators",
    fontweight="bold",
    fontsize="large",
)
# Set the legend title
ax.legend(title="Alignment method", title_fontsize="large")

# Move the legend outside the plot
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

# Add gray rectangles to separate the datasets
for i, data_name in enumerate(df["data_name"].unique()):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.05)

# Fix underscores in the dataset names
ax.set_yticklabels(
    [name.replace("_", " ") for name in df["data_name"].unique()]
)

# Adjust the layout to prevent the legend from being cut off
plt.tight_layout()

# Save the figure with high resolution
plt.savefig(
    figures_path / "boxplot_accuracies.pdf", dpi=500, bbox_inches="tight"
)

# Display the plot
plt.show()


# %%
# Concatenate pdfs per dataset and solver
aligned_dataset_paths = figures_path / "aligned_datasets"
# Get the list of folders
aligned_datasets = [
    folder for folder in aligned_dataset_paths.iterdir() if folder.is_dir()
]

import matplotlib.gridspec as gridspec
import matplotlib as mpl
from nilearn import surface, plotting, datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_contrast(
    ax,
    contrast_path,
    hemi="left",
    cmap="coolwarm",
    **kwargs,
):
    # Load the contrast
    surface_map = surface.load_surf_data(str(contrast_path) + f"_{hemi}.gii")
    fsaverage = datasets.fetch_surf_fsaverage()
    plotting.plot_surf(
        fsaverage.infl_left,
        surface_map,
        cmap=cmap,
        hemi="left",
        axes=ax,
        colorbar=False,
        bg_map=fsaverage.sulc_left,
        bg_on_data=True,
        darkness=0.5,
        **kwargs,
    )


def generate_tiling_figure(dataset_path, vmin=-10, vmax=10):
    # Grab the available solvers
    solvers = [solver for solver in dataset_path.iterdir() if solver.is_dir()]
    # List the contrasts for the first solver in the template folder
    contrasts = [
        contrast
        for contrast in (solvers[0] / "template").iterdir()
        if contrast.is_file()
    ]
    # Split on the last underscore to get the contrast name
    contrast_names = [
        contrast.stem.rsplit("_", 1)[0] for contrast in contrasts
    ]
    contrast_names = list(set(contrast_names))

    # Create the figure and axes with a specific size
    fig = plt.figure(figsize=(3 * len(solvers), 3 * len(contrast_names)))
    grid_spec = gridspec.GridSpec(
        len(contrast_names), len(solvers), figure=fig
    )
    for i, contrast_name in enumerate(contrast_names):
        for j, solver in enumerate(solvers):
            ax = fig.add_subplot(grid_spec[i, j], projection="3d")
            plot_contrast(
                ax,
                solver / "template" / f"{contrast_name}",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
            if i == 0:
                ax.set_title(solver.name)
            if j == 0:
                # Add the contrast name in a separate column
                ax.text2D(
                    0.0,
                    0.3,
                    contrast_name,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                )
    # Add colorbar
    ax = fig.add_subplot(grid_spec[len(contrast_names) // 2, :])
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%")
    fig.add_axes(cax)
    fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap="coolwarm"
        ),
        cax=cax,
    )
    fig.savefig(dataset_path / "templates.pdf", dpi=500, bbox_inches="tight")


for dataset_path in aligned_datasets:
    print(f"Generating template figure for {dataset_path.name}")
    generate_tiling_figure(dataset_path)
