from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from nilearn.maskers import NiftiMasker
    from fmralign.tests.utils import random_niimg
    from benchmark_utils.datasets_utils import (
        check_init_dataset,
        log_dataset_info,
        sample_dataset,
    )


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

        self.dataset = sample_dataset(
            name=self.name,
            masker=masker,
            subjects=self.subjects,
            n_samples_alignement=self.n_samples_alignement,
            n_samples_decoding=self.n_samples_decoding,
        )

        check_init_dataset(self.dataset)
        log_dataset_info(self.dataset)

        return dict(dataset=self.dataset)
