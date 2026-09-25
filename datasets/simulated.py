from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.datasets_utils import DatasetParams

SUBJECTS = ["sub-01", "sub-02", "sub-03"]


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    parameters = {
        "target": ["template_in_sample", "template_out_of_sample"] + SUBJECTS
    }

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`

        dataset_params = DatasetParams(
            name=self.name,
            subjects=SUBJECTS,
            target=self.target,
            data_path=None,
            task=self.name,
            n_subjects=len(SUBJECTS),
        )

        return {"dataset_params": dataset_params}
