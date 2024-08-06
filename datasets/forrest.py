from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.config import DATA_PATH_FORREST, MEMORY
    from benchmark_utils.datasets_utils import load_dataset, load_mask
    from pathlib import Path


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Forrest"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["nilearn", "pandas"]

    def __init__(self):
        self.subjects = [
            "sub-01",
            "sub-02",
            "sub-03",
            "sub-04",
            "sub-05",
            "sub-06",
            "sub-07",
            "sub-08",
            "sub-10",
            "sub-11",
        ]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        data_path = Path(DATA_PATH_FORREST)

        # Load the masker object
        mask = load_mask(data_path, MEMORY)

        dict_alignment = dict()
        dict_decoding = dict()
        dict_labels = dict()
        for subject in self.subjects:
            (
                data_alignment,
                data_decoding,
                labels,
            ) = load_dataset(subject, data_path, mask)

            dict_alignment[subject] = data_alignment
            dict_decoding[subject] = data_decoding
            dict_labels[subject] = labels

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            dict_alignment=dict_alignment,
            dict_decoding=dict_decoding,
            dict_labels=dict_labels,
            mask=mask,
        )
