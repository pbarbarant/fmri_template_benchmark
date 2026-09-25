from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from pathlib import Path

    from benchmark_utils.conf import HCP_CONDITIONS_DIR
    from benchmark_utils.datasets_utils import DatasetParams, parse_subjects


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "HCP"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    data_path = HCP_CONDITIONS_DIR

    parameters = {
        "target": ["template_in_sample"],
        "n_subjects": [10, 20, 50, 100],
        "n_movies": [1, 2, 3, 4],
    }

    def get_data(self, data_path: Path = data_path):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`
        subjects = parse_subjects(data_path)[: self.n_subjects]
        dataset_params = DatasetParams(
            name=self.name,
            subjects=subjects,
            target=self.target,
            data_path=data_path,
            task=self.name,
            n_subjects=len(subjects),
            n_parcels=None,
            n_movies=self.n_movies,
        )

        return {"dataset_params": dataset_params}
