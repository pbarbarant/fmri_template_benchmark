from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold

    from benchmark_utils.conf import NSD_DIR
    from benchmark_utils.datasets_utils import Dataset as DatasetDataClass
    from benchmark_utils.datasets_utils import Fold, parse_subjects


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "NSD"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    data_path = NSD_DIR

    parameters = {
        "n_parcels": [200, 400, 600, 800],
        "target": ["template_in_sample", "template_out_of_sample"]
        + parse_subjects(data_path),
    }

    def get_data(self, data_path: Path = data_path):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`
        labels = np.load(data_path / f"schaefer_{self.n_parcels}.npy")
        subjects = parse_subjects(data_path)

        y = (
            pd.read_csv(data_path / f"{subjects[0]}_labels.csv", header=None)
            .values.astype(str)
            .ravel()
        )

        subjects_data = [np.load(data_path / f"{s}.npy") for s in subjects]

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        splits = kf.split(np.arange(len(y)))

        folds = []
        for fold_idx, (decoding_idx, alignment_idx) in enumerate(splits):
            dict_alignment = {
                s: data[alignment_idx]
                for s, data in zip(subjects, subjects_data)
            }
            dict_decoding = {
                s: data[decoding_idx]
                for s, data in zip(subjects, subjects_data)
            }
            dict_y = {s: y[decoding_idx] for s in subjects}
            folds.append(
                Fold(
                    index=fold_idx,
                    dict_alignment=dict_alignment,
                    dict_decoding=dict_decoding,
                    dict_y=dict_y,
                )
            )

        self.dataset = DatasetDataClass(
            name=self.name + f"_{self.n_parcels}",
            subjects=subjects,
            n_subjects=len(subjects),
            labels=labels,
            folds=folds,
            task_name="NSD",
            target=self.target,
        )

        return {"dataset": self.dataset}
