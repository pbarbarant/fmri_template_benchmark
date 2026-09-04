from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold

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

    parameters = {"target": ["template_out_of_sample"]}

    def get_data(self, data_path: Path = data_path):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`
        labels = np.load(data_path / "schaefer_400_parcellation.npy")
        subjects = parse_subjects(data_path)[:100]  # Limit to 100 subjects
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        folds_indices = list(kf.split(subjects))

        folds = []
        for fold_idx, (decoding_idx, _) in enumerate(folds_indices):
            dict_alignment = {
                sub: np.load(data_path / f"{sub}_movie.npy", mmap_mode="r")[
                    10:,
                ]  # Skip first 10 TRs to avoid movie onset effects
                for sub in np.array(subjects)
            }
            dict_decoding = {
                sub: np.load(data_path / f"{sub}_task.npy", mmap_mode="r")
                for sub in np.array(subjects)[decoding_idx]
            }
            dict_y = {
                sub: (
                    pd.read_csv(data_path / f"{sub}_labels.csv")["condition"]
                    .values.astype(str)
                    .ravel()
                )
                for sub in np.array(subjects)[decoding_idx]
            }
            folds.append(
                Fold(
                    index=fold_idx,
                    dict_alignment=dict_alignment,
                    dict_decoding=dict_decoding,
                    dict_y=dict_y,
                    decoding_subjects=np.array(subjects)[decoding_idx].tolist(),
                )
            )

        self.dataset = DatasetDataClass(
            name=self.name,
            subjects=subjects,
            n_subjects=len(subjects),
            labels=labels,
            folds=folds,
            task_name="hcp",
            target=self.target,
        )

        return {"dataset": self.dataset}
