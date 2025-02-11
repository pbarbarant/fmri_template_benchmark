from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.datasets_utils import (
        check_init_dataset,
        fetch_raiders,
        log_dataset_info,
    )

SUBJECTS = [
    "sub-rid000005",
    "sub-rid000011",
    "sub-rid000014",
    "sub-rid000015",
    "sub-rid000028",
    "sub-rid000029",
    "sub-rid000033",
    "sub-rid000038",
    "sub-rid000042",
    "sub-rid000043",
]


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Raiders"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        "left_out_run": [1, 2, 3, 4, 5, 6, 7, 8],
        "target": ["template"],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = []

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`

        self.dataset = fetch_raiders(
            n_parcels=400,
            target=self.target,
            lo_run=self.left_out_run,
        )

        check_init_dataset(self.dataset)
        log_dataset_info(self.dataset)

        return dict(dataset=self.dataset)
