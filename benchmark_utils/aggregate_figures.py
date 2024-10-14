# %%
import glob
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots  # noqa: F401
import numpy as np

from nilearn import surface, plotting, datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["figure.dpi"] = 500

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
# Parse the latest file
file_list = glob.glob(os.path.join(data_path, "*.parquet"))
latest_file = max(file_list, key=os.path.getmtime)

# %%
aligned_dataset_paths = figures_path / "aligned_datasets"
# Get the list of folders
aligned_datasets = [
    folder for folder in aligned_dataset_paths.iterdir() if folder.is_dir()
]


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
        fsaverage.pial_left,
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


def plot_weights(
    ax,
    weights_list,
    hemi="left",
    cmap="coolwarm",
    **kwargs,
):
    # Load all the weights
    surface_maps = np.stack(
        [surface.load_surf_data(str(weights)) for weights in weights_list],
    )
    mean_map = surface_maps.mean(axis=0)
    normalized_map = mean_map / np.abs(mean_map).max()
    fsaverage = datasets.fetch_surf_fsaverage()
    plotting.plot_surf(
        fsaverage.pial_left,
        normalized_map,
        cmap=cmap,
        hemi="left",
        axes=ax,
        colorbar=False,
        bg_map=fsaverage.sulc_left,
        bg_on_data=True,
        threshold=0.5,
        **kwargs,
    )


def generate_template_figure(dataset_path, vmin=-1, vmax=1):
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


def generate_weights_figure(dataset_path, vmin=-1, vmax=1):
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
            # Glob all files recursively that start with coefs_contrast_name
            weights_list = list(
                solver.glob(f"**/coefs_{contrast_name}_*left.gii")
            )
            plot_weights(
                ax,
                weights_list,
                cmap="cold_white_hot",
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
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
            cmap="cold_white_hot",
        ),
        cax=cax,
    )
    fig.savefig(dataset_path / "weights.pdf", dpi=500, bbox_inches="tight")


for dataset_path in aligned_datasets:
    print(f"Generating weights figure for {dataset_path.name}")
    generate_weights_figure(dataset_path)
    print(f"Generating template figure for {dataset_path.name}")
    generate_template_figure(dataset_path)
