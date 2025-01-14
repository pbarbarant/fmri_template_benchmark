from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.datasets_utils import (
        check_init_dataset,
        fetch_hcp,
        log_dataset_info,
    )


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "HCP_Social"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    def __init__(self):
        self.n_subjects = 100
        self.n_parcels = 400

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`

        self.dataset = fetch_hcp(
            name=self.name,
            n_subjects=int(self.n_subjects),
            task="SOCIAL",
            n_parcels=self.n_parcels,
        )

        check_init_dataset(self.dataset)
        log_dataset_info(self.dataset)

        return dict(dataset=self.dataset)
