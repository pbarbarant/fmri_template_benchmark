from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from pathlib import Path

    from nilearn.experimental import surface

    from benchmark_utils.config import DATA_PATH_IBC_MATHLANG, MEMORY
    from benchmark_utils.utils import load_dataset_surf, load_mask


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "IBC_MathLanguage"
    mesh_name = "fsaverage5"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    def __init__(self):
        self.subjects = [
            "sub-01",
            "sub-04",
            "sub-05",
            "sub-06",
            "sub-07",
            "sub-09",
            "sub-11",
            "sub-12",
            "sub-13",
            "sub-14",
        ]

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        data_path = Path(DATA_PATH_IBC_MATHLANG)

        dict_alignment = dict()
        dict_decoding = dict()
        for subject in self.subjects:
            print(f"Loading data for subject {subject}")
            (
                data_alignment,
                data_decoding,
            ) = load_dataset_surf(subject, data_path, self.mesh_name)
            dict_alignment[subject] = data_alignment
            dict_decoding[subject] = data_decoding

        # Get the first image to create the masker
        masker_img = next(iter(dict_alignment.values())).img
        masker = surface.SurfaceMasker().fit(masker_img)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            dataset_name=self.name,
            dict_alignment=dict_alignment,
            dict_decoding=dict_decoding,
            masker=masker,
            mesh_name=self.mesh_name,
        )
