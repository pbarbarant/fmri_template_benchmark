from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from fmralign.tests.utils import random_niimg
    from nilearn.maskers import NiftiMasker

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

    parameters = {
        "target": ["sub-01", "template"],
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`
        self.subjects = ["sub-01", "sub-02", "sub-03"]
        self.n_samples_alignement = 200
        self.n_samples_decoding = 150

        _, mask_img = random_niimg((5, 4, 3))
        masker = NiftiMasker(mask_img=mask_img).fit()

        self.dataset = sample_dataset(
            name=self.name,
            target=self.target,
            masker=masker,
            clustering_img=mask_img,
            subjects=self.subjects,
            n_samples_alignement=self.n_samples_alignement,
            n_samples_decoding=self.n_samples_decoding,
        )

        check_init_dataset(self.dataset)
        log_dataset_info(self.dataset)

        return dict(dataset=self.dataset)
