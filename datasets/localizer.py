from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn import datasets
    from nilearn.image import load_img
    from nilearn.experimental import surface
    from nilearn.experimental.surface._datasets import load_fsaverage
    from nilearn.experimental.surface._surface_image import SurfaceImage

    from benchmark_utils.datasets_utils import project_on_surf


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Localizer"
    mesh_name = "fsaverage3"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["nilearn", "numpy"]

    def __init__(
        self,
    ):
        self.subjects = [f"{i}" for i in range(1, 10)]

    def load_localizer_surf(self, subject, mesh_name):
        print(f"Loading subject {subject}")

        # List of contrasts to fetch
        contrasts = [
            "calculation vs sentences",
            "left vs right button press",
            "checkerboard",
            "sentence reading",
            "sentence listening",
        ]

        # Number of contrasts used for alignment
        n_training_contrasts = 3

        subject_data = datasets.fetch_localizer_contrasts(
            contrasts=contrasts,
            n_subjects=[int(subject)],
            get_anats=True,
        )

        # Load the data in volumetric form
        alignment_contrasts_vol = load_img(
            subject_data.cmaps[0:n_training_contrasts]
        )
        decoding_contrasts_vol = load_img(
            subject_data.cmaps[n_training_contrasts:]
        )

        # Create a SurfaceImage object for the alignment and decoding contrasts
        alignment_contrasts_surf = project_on_surf(
            alignment_contrasts_vol, mesh_name
        )
        decoding_contrasts_surf = project_on_surf(
            decoding_contrasts_vol, mesh_name
        )

        # Load labels
        labels = np.array(contrasts[n_training_contrasts:])

        return alignment_contrasts_surf, decoding_contrasts_surf, labels

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Create a masker to extract the data from the brain volume.
        dict_alignment = dict()
        dict_decoding = dict()
        dict_labels = dict()

        for subject in self.subjects:
            (
                data_alignment,
                data_decoding,
                labels,
            ) = self.load_localizer_surf(subject, self.mesh_name)
            dict_alignment[subject] = data_alignment
            dict_decoding[subject] = data_decoding
            dict_labels[subject] = labels

        masker = surface.SurfaceMasker().fit(
            next(iter(dict_alignment.values()))
        )

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            dataset_name=self.name,
            dict_alignment=dict_alignment,
            dict_decoding=dict_decoding,
            dict_labels=dict_labels,
            masker=masker,
            mesh_name=self.mesh_name,
        )
