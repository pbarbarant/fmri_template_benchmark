from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from benchmark_utils.conf import HCP_CONDITIONS_DIR
    from benchmark_utils.datasets_utils import Dataset as DatasetDataClass
    from benchmark_utils.datasets_utils import Fold, parse_subjects


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
        labels = np.load(data_path / "schaefer_400_parcellation.npy")
        subjects = parse_subjects(data_path)[
            : self.n_subjects
        ]  # Limit to 100 subjects

        dict_alignment = {
            sub: [
                np.load(data_path / f"{sub}_movie{i}.npy", mmap_mode="r")
                for i in range(1, self.n_movies + 1)
            ]
            for sub in np.array(subjects)
        }
        timepoints_masks = [
            np.load(data_path / f"movie{i}_mask.npy")
            for i in range(1, self.n_movies + 1)
        ]
        dict_decoding = {
            sub: np.load(data_path / f"{sub}_task.npy", mmap_mode="r")
            for sub in np.array(subjects)
        }
        dict_y = {
            sub: (
                pd.read_csv(data_path / f"{sub}_labels.csv")["condition"]
                .values.astype(str)
                .ravel()
            )
            for sub in np.array(subjects)
        }
        folds = [
            Fold(
                index=0,
                dict_alignment=dict_alignment,
                dict_decoding=dict_decoding,
                dict_y=dict_y,
                timepoints_masks=timepoints_masks,
            )
        ]

        self.dataset = DatasetDataClass(
            name=self.name,
            subjects=subjects,
            n_subjects=len(subjects),
            labels=labels,
            folds=folds,
            task_name="hcp" + f"_{self.n_subjects}_{self.n_movies}",
            target=self.target,
        )

        return {"dataset": self.dataset}
