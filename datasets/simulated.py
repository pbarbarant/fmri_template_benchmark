from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.datasets_utils import (
        sample_fold,
    )
    from nilearn.maskers import NiftiMasker
    from fmralign.tests.utils import random_niimg


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
        _, mask_img = random_niimg((8, 7, 6))
        masker = NiftiMasker(mask_img=mask_img).fit()

        folds = [
            sample_fold(
                f"fold-{i:02d}",
                masker,
                self.subjects,
                self.n_samples_alignement,
                self.n_samples_decoding,
            )
            for i in range(2)
        ]

        return dict(
            dataset_name=self.name,
            folds=folds,
            masker=masker,
            clustering_img=mask_img,
        )
