from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn.maskers import NiftiMasker
    from nilearn.datasets import load_mni152_brain_mask
    from benchmark_utils.utils import LabeledImage, Fold


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


def _sample_fold(
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


def fetch_fitted_masker():
    mask_img = load_mni152_brain_mask(resolution=2)
    return NiftiMasker(
        mask_img=mask_img, memory="nilearn_cache", memory_level=1
    ).fit()


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    def __init__(
        self,
    ):
        self.subjects = ["sub-01", "sub-02", "sub-03"]
        self.n_samples_alignement = 20
        self.n_samples_decoding = 15

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`
        self.masker = fetch_fitted_masker()

        folds = [
            _sample_fold(
                f"fold-{i:02d}",
                self.masker,
                self.subjects,
                self.n_samples_alignement,
                self.n_samples_decoding,
            )
            for i in range(2)
        ]

        return dict(
            dataset_name=self.name,
            folds=folds,
            masker=self.masker,
        )
