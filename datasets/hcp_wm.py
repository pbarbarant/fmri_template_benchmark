from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchmark_utils.datasets_utils import (
        check_dataset,
        fetch_one_hcp_fold,
        log_dataset_info,
        fit_masker_to_data,
        fetch_clustering_img,
    )


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "HCP_Wm"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    def __init__(self):
        self.subjects = [
            "552544",
            "553344",
            "555348",
        ]
        self.n_parcels = 400

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # The dictionary defines the keyword arguments for `Objective.set_data`

        folds = [fetch_one_hcp_fold("fold-00", self.subjects, task="WM")]
        masker = fit_masker_to_data(folds[0].dict_alignment[self.subjects[0]].img)
        clustering_img = fetch_clustering_img(masker, n_rois=self.n_parcels)

        check_dataset(folds, masker, clustering_img)
        log_dataset_info(self.name, folds, clustering_img)

        return dict(
            dataset_name=self.name,
            folds=folds,
            masker=masker,
            clustering_img=clustering_img,
        )
