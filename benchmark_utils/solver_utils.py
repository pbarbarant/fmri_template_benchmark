from pathlib import Path

import numpy as np
from joblib import dump, load

from benchmark_utils.datasets_utils import LabeledImage


def _compute_template(
    algo,
    dataset,
    solver_name,
):
    # Get the list of subjects
    subject_list = list(dataset.dict_alignment.keys())

    # Get the list of images
    imgs = [dataset.dict_alignment[subject] for subject in subject_list]

    # Align the images
    algo.fit(imgs)

    # Initialize the template
    template_data = np.zeros_like(
        dataset.masker.transform(dataset.dict_decoding[subject_list[0]].img)
    )
    dict_aligned = dict()
    for i, subject in enumerate(subject_list):
        transformed_img = algo.transform(
            dataset.dict_decoding[subject].img, subject_index=i
        )
        dict_aligned[subject] = LabeledImage(
            img=transformed_img,
            y=dataset.dict_decoding[subject].y,
        )
        template_data += dataset.masker.transform(transformed_img) / len(
            subject_list
        )

    # Convert the template to a LabeledImage
    template = LabeledImage(
        img=dataset.masker.inverse_transform(template_data),
        y=dataset.dict_decoding[subject_list[0]].y,
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
    algo,
    dataset,
    solver_name,
):
    cache_dir = (
        Path(__file__).parent.parent
        / "memory_cache"
        / "alignments"
        / dataset.name
        / solver_name
    )
    if (cache_dir / "dict_aligned.pkl").exists() and (
        cache_dir / "template.pkl"
    ).exists():
        print(f"Loading aligned data from cache: {cache_dir}")
        # Load the dataset from the cache
        template = load(cache_dir / "template.pkl")
        dict_aligned = load(cache_dir / "dict_aligned.pkl")
        dataset.template = template
        dataset.dict_aligned = dict_aligned
        return dataset
    else:
        dataset = _compute_template(
            algo=algo,
            dataset=dataset,
            solver_name=solver_name,
        )
        # Save the dataset to the cache
        cache_dir.mkdir(exist_ok=True, parents=True)
        dump(dataset.template, cache_dir / "template.pkl")
        dump(dataset.dict_aligned, cache_dir / "dict_aligned.pkl")
        return dataset


def save_template_nii(template, dataset_name, solver_name):
    output_dir = Path("outputs") / dataset_name / solver_name
    output_dir.mkdir(exist_ok=True, parents=True)
    template_img = template.img
    template_img.to_filename(output_dir / "template.nii.gz")

def save_template_gii(template, dataset_name, solver_name):
    output_dir = Path("outputs") / dataset_name / solver_name
    output_dir.mkdir(exist_ok=True, parents=True)
    template_img = template.img
    template_img.data.to_filename(output_dir / "template_data.gii")
