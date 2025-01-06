import numpy as np
from benchmark_utils.utils import LabeledImage, Fold
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018, load_mni152_brain_mask
from nilearn.maskers import NiftiMasker


def fetch_clustering_img(masker, n_rois=400, resolution_mm=2):
    clustering_img = (
        masker.transform(
            fetch_atlas_schaefer_2018(
                n_rois=n_rois,
                resolution_mm=resolution_mm,
            )["maps"]
        )
    ).astype(int)
    return image.index_img(masker.inverse_transform(clustering_img), 0)


def _sample_labels(n_samples):
    return np.arange(3).reshape(1, -1).repeat(n_samples // 3, axis=0).flatten()


def _sample_labeled_image(n_samples, masker):
    mask_img = masker.mask_img_
    n_voxels = masker.transform(mask_img).shape[1]
    data = np.random.randn(n_samples, n_voxels)
    img = masker.inverse_transform(data)
    y = _sample_labels(n_samples)
    return LabeledImage(
        img=img,
        y=y,
    )


def sample_fold(
    name,
    masker,
    subjects,
    n_samples_alignement,
    n_samples_decoding,
):
    print(f"Generating fold {name}")
    dict_alignment = dict()
    dict_decoding = dict()
    for subject in subjects:
        # Generate random surface images for each subject.
        dict_alignment[subject] = _sample_labeled_image(
            n_samples_alignement, masker
        )
        dict_decoding[subject] = _sample_labeled_image(
            n_samples_decoding, masker
        )

    return Fold(
        name=name,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
    )


def fit_mni152_masker(resolution=2):
    mask_img = load_mni152_brain_mask(resolution=resolution)
    return NiftiMasker(
        mask_img=mask_img, memory="nilearn_cache", memory_level=1
    ).fit()
