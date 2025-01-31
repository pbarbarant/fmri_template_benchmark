from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fmralign.template_alignment import TemplateAlignment
from joblib import dump, load

from benchmark_utils.datasets_utils import Dataset, LabeledImage
from sklearn.decomposition import PCA


def compute_pca(dict_subjects: dict, subjects, masker) -> np.ndarray:
    imgs = [
        dict_subjects[subject].img for subject in subjects
    ]  # Use subjects passed as parameter
    data = np.concatenate([masker.transform(img) for img in imgs], axis=0)
    pca = PCA(n_components=2)
    return pca.fit_transform(data)


def plot_pca(dataset: Dataset, solver_name: str) -> None:
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

    for subject in subjects:
        ax[0, 0].scatter(
            pca_unaligned[legend_subjects == subject, 0],
            pca_unaligned[legend_subjects == subject, 1],
            label=subject,
        )
        ax[0, 1].scatter(
            pca_aligned[legend_subjects == subject, 0],
            pca_aligned[legend_subjects == subject, 1],
            label=subject,
        )

    for condition in np.unique(legend_condition):
        ax[1, 0].scatter(
            pca_unaligned[legend_condition == condition, 0],
            pca_unaligned[legend_condition == condition, 1],
            label=condition,
        )
        ax[1, 1].scatter(
            pca_aligned[legend_condition == condition, 0],
            pca_aligned[legend_condition == condition, 1],
            label=condition,
        )

    # Add titles and legends
    ax[0, 0].set_title("PCA of unaligned data")
    ax[0, 1].set_title(f"PCA of aligned data ({solver_name})")
    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[1, 0].legend()
    ax[1, 1].legend()

    output_dir = Path("outputs") / "figures"
    output_dir.mkdir(exist_ok=True, parents=True)
    # Save the figure
    fig.savefig(
        output_dir / f"{dataset.name}_{solver_name}_pca.png",
        bbox_inches="tight",
        dpi=300,
    )


def _compute_template(
    algo: TemplateAlignment,
    dataset: Dataset,
    solver_name: str,
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
        save_template_gii(template, dataset.name, solver_name)
    else:
        save_template_nii(template, dataset.name, solver_name)

    dataset.template = template
    dataset.dict_aligned = dict_aligned

    return dataset


def compute_template(
    algo: TemplateAlignment,
    dataset: Dataset,
    solver_name: str,
) -> Dataset:
    dataset = _compute_template(
        algo=algo,
        dataset=dataset,
        solver_name=solver_name,
    )
    print("Computing PCA")
    plot_pca(dataset, solver_name)
    print("PCA computed")
    # Save the dataset to the cache
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
