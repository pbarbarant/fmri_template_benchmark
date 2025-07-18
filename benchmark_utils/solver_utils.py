from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from benchmark_utils.datasets_utils import Dataset


def compute_pca(dict_subjects: dict) -> np.ndarray:
    data = np.vstack(list(dict_subjects.values()))
    if data.shape[1] == 2:
        return data
    else:
        pca = PCA(n_components=2)
        return pca.fit_transform(data)


def plot_pca(dataset: Dataset) -> None:
    subjects = dataset.subjects
    pca_unaligned = compute_pca(dataset.dict_decoding)
    pca_aligned = compute_pca(dataset.dict_aligned)

    # Create a single figure with 2 subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    legend_subjects = np.repeat(
        subjects, dataset.dict_y[subjects[0]].shape[0]
    )
    legend_condition = np.concatenate(
        [dataset.dict_y[subject] for subject in subjects]
    )

    # Define separate colormaps
    subject_colormap = plt.cm.tab10  # Colormap for subjects
    condition_colormap = plt.cm.viridis  # Colormap for conditions

    # Map subjects to colors
    subject_colors = {
        subject: subject_colormap(i / len(subjects))
        for i, subject in enumerate(subjects)
    }
    condition_colors = {
        condition: condition_colormap(i / len(np.unique(legend_condition)))
        for i, condition in enumerate(np.unique(legend_condition))
    }

    for subject in subjects:
        ax[0, 0].scatter(
            pca_unaligned[legend_subjects == subject, 0],
            pca_unaligned[legend_subjects == subject, 1],
            color=subject_colors[subject],
            label=subject,
        )
        ax[0, 1].scatter(
            pca_aligned[legend_subjects == subject, 0],
            pca_aligned[legend_subjects == subject, 1],
            color=subject_colors[subject],
            label=subject,
        )

    for condition in np.unique(legend_condition):
        ax[1, 0].scatter(
            pca_unaligned[legend_condition == condition, 0],
            pca_unaligned[legend_condition == condition, 1],
            color=condition_colors[condition],
            label=condition,
        )
        ax[1, 1].scatter(
            pca_aligned[legend_condition == condition, 0],
            pca_aligned[legend_condition == condition, 1],
            color=condition_colors[condition],
            label=condition,
        )

    # Add titles and legends
    ax[0, 0].set_title("PCA of unaligned data")
    ax[0, 1].set_title(f"PCA - {dataset.solver} - Target: {dataset.target_name}")
    # Add legends to the right of the rightmost plots
    ax[0, 1].legend(
        loc="center left", bbox_to_anchor=(1, 0.5), title="Subjects"
    )
    ax[1, 1].legend(
        loc="center left", bbox_to_anchor=(1, 0.5), title="Conditions"
    )

    # Adjust layout to fit legends
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Leave space for legends
    output_dir = dataset.output_dir / dataset.target_name
    output_dir.mkdir(exist_ok=True, parents=True)
    # Save the figure
    fig.savefig(
        output_dir / "pca.png",
        bbox_inches="tight",
        dpi=300,
    )


def compute_alignment(
    algo,
    dataset: Dataset,
    solver_name: str,
) -> Dataset:
    dataset.solver = solver_name
    output_dir = Path("outputs") / dataset.name / solver_name
    output_dir.mkdir(exist_ok=True, parents=True)
    dataset.output_dir = output_dir
    
    # Time the alignment process
    start_time = perf_counter()
    algo.fit(list(dataset.dict_alignment.values()))
    aligned_data = algo.transform(list(dataset.dict_decoding.values()), range(dataset.n_subjects))
    dataset.dict_aligned = dict(zip(dataset.subjects, aligned_data))
    dataset.time = perf_counter() - start_time
    
    # Compute the PCA
    print("Computing PCA")
    plot_pca(dataset)
    print("PCA computed")
    return dataset