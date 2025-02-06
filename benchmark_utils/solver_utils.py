from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fmralign.template_alignment import TemplateAlignment
from fmralign.pairwise_alignment import PairwiseAlignment
from fmralign.sparse_template_alignment import SparseTemplateAlignment
from fmralign.sparse_pairwise_alignment import SparsePairwiseAlignment

from benchmark_utils.datasets_utils import Dataset, LabeledImage
from sklearn.decomposition import PCA

from typing import Union


def compute_pca(dict_subjects: dict, subjects, masker) -> np.ndarray:
    imgs = [
        dict_subjects[subject].img for subject in subjects
    ]  # Use subjects passed as parameter
    data = np.concatenate([masker.transform(img) for img in imgs], axis=0)
    if data.shape[1] == 2:
        return data
    else:
        pca = PCA(n_components=2)
        return pca.fit_transform(data)


def plot_pca(dataset: Dataset) -> None:
    subjects = dataset.subjects
    masker = dataset.masker
    pca_unaligned = compute_pca(dataset.dict_decoding, subjects, masker)
    pca_aligned = compute_pca(dataset.dict_aligned, subjects, masker)

    # Create a single figure with 2 subplots
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    legend_subjects = np.repeat(
        subjects, dataset.dict_decoding[subjects[0]].y.shape[0]
    )
    legend_condition = np.concatenate(
        [dataset.dict_decoding[subject].y for subject in subjects]
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
    ax[0, 1].set_title(f"PCA - {dataset.solver} - Target: {dataset.target}")
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
    output_dir = (
        Path("outputs") / dataset.name / dataset.solver / dataset.target
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    # Save the figure
    fig.savefig(
        output_dir / "pca.png",
        bbox_inches="tight",
        dpi=300,
    )


def compute_template(
    algo: TemplateAlignment,
    dataset: Dataset,
) -> Dataset:
    # Get the list of subjects
    subjects = dataset.subjects

    # Get the list of images
    imgs = [dataset.dict_alignment[subject] for subject in subjects]

    # Align the images
    algo.fit(imgs)

    # Initialize the template
    template_data = np.zeros_like(
        dataset.masker.transform(dataset.dict_decoding[subjects[0]].img)
    )
    dict_aligned = dict()
    for i, subject in enumerate(subjects):
        transformed_img = algo.transform(
            dataset.dict_decoding[subject].img, subject_index=i
        )
        dict_aligned[subject] = LabeledImage(
            img=transformed_img,
            y=dataset.dict_decoding[subject].y,
        )
        template_data += dataset.masker.transform(transformed_img) / len(
            subjects
        )

    # Convert the template to a LabeledImage
    template = LabeledImage(
        img=dataset.masker.inverse_transform(template_data),
        y=dataset.dict_decoding[subjects[0]].y,
    )

    # Save the template
    if dataset.is_surf:
        save_template_gii(template, dataset.name, dataset.solver)
    else:
        save_template_nii(template, dataset.name, dataset.solver)

    dataset.template = template
    dataset.dict_aligned = dict_aligned

    # euclidean_template = (
    #     dataset.dict_alignment["sub-01"].get_fdata()
    #     + dataset.dict_alignment["sub-02"].get_fdata()
    # ) / 2
    # assert np.allclose(
    #     euclidean_template,
    #     dataset.template.img.get_fdata(),
    # )

    return dataset


def compute_pairwise(
    target_subject: str,
    algo: PairwiseAlignment,
    dataset: Dataset,
) -> Dataset:
    subjects = dataset.subjects
    dict_aligned = dict()
    for subject in subjects:
        if subject == target_subject:
            dict_aligned[subject] = dataset.dict_decoding[subject]
        else:
            algo.fit(
                dataset.dict_alignment[subject],
                dataset.dict_alignment[target_subject],
            )
            transformed_img = algo.transform(
                dataset.dict_decoding[subject].img
            )
            dict_aligned[subject] = LabeledImage(
                img=transformed_img,
                y=dataset.dict_decoding[subject].y,
            )

    dataset.dict_aligned = dict_aligned
    return dataset


def compute_alignment(
    algo: Union[TemplateAlignment, PairwiseAlignment],
    dataset: Dataset,
    solver_name: str,
) -> Dataset:
    dataset.solver = solver_name
    if isinstance(algo, (TemplateAlignment, SparseTemplateAlignment)):
        dataset = compute_template(algo, dataset)
    elif isinstance(algo, (PairwiseAlignment, SparsePairwiseAlignment)):
        dataset = compute_pairwise(dataset.target, algo, dataset)
    else:
        raise ValueError(
            "algo must be either TemplateAlignment or PairwiseAlignment"
        )

    # Compute the PCA
    print("Computing PCA")
    plot_pca(dataset)
    print("PCA computed")
    return dataset


def save_template_nii(
    template: LabeledImage, dataset_name: str, solver_name: str
) -> None:
    output_dir = Path("outputs") / dataset_name / solver_name
    output_dir.mkdir(exist_ok=True, parents=True)
    template_img = template.img
    template_img.to_filename(output_dir / "template.nii.gz")


def save_template_gii(
    template: LabeledImage, dataset_name: str, solver_name: str
) -> None:
    output_dir = Path("outputs") / dataset_name / solver_name
    output_dir.mkdir(exist_ok=True, parents=True)
    template_img = template.img
    template_img.data.to_filename(output_dir / "template_data.gii")
