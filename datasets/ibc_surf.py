from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.datasets_utils import (
        check_init_dataset,
        fetch_ibc_surf,
        log_dataset_info,
    )

SUBJECTS = [
    "sub-01",
    "sub-02",
    "sub-04",
    "sub-05",
    "sub-06",
    "sub-07",
    "sub-08",
    "sub-09",
    "sub-11",
    "sub-12",
    "sub-13",
    "sub-14",
    "sub-15",
]

TASKS = ["Audio", "FaceBody", "Mario", "MathLanguage", "RSVPLanguage", "HcpWm"]

# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "IBC_Surf"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    parameters = {"target_name": ["template"] + SUBJECTS, "task": TASKS}

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`

        self.dataset = fetch_ibc_surf(
            name=self.name,
            target_name=self.target_name,
            subjects=SUBJECTS,
            task=self.task,
        )

        # check_init_dataset(self.dataset)
        # log_dataset_info(self.dataset)

        return dict(dataset=self.dataset)
