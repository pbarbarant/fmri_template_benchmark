from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn import datasets
    from nilearn.experimental import surface
    from nilearn.image import load_img

    from benchmark_utils.utils import project_on_surf, LabeledImage


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Localizer"
    mesh_name = "fsaverage5"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["nilearn", "numpy"]

    def __init__(
        self,
    ):
        self.subjects = [f"{i}" for i in range(1, 6)]

    def load_localizer_surf(self, subject, mesh_name):
        print(f"Loading subject {subject}")

        # List of contrasts to fetch
        contrasts = [
            "horizontal checkerboard",
            "vertical checkerboard",
            "horizontal vs vertical checkerboard",
            "vertical vs horizontal checkerboard",
            "sentence reading",
            "sentence listening and reading",
            "sentence reading vs checkerboard",
            "calculation (auditory cue)",
            "calculation (visual cue)",
            "calculation (auditory and visual cue)",
            "calculation (auditory cue) vs sentence listening",
            "calculation (visual cue) vs sentence reading",
            "calculation vs sentences",
            "calculation (auditory cue) and sentence listening",
            "calculation (visual cue) and sentence reading",
            "calculation (visual cue) and sentence reading",
            "calculation (visual cue) and sentence reading vs checkerboard",
            "calculation and sentence listening/reading vs button press",
            "left button press (auditory cue)",
            "left button press (visual cue)",
            "left button press",
            "right button press (auditory cue)",
            "right button press (visual cue)",
            "right button press",
            "right vs left button press",
            "button press (auditory cue) vs sentence listening",
            "button press (visual cue) vs sentence reading",
            "button press vs calculation and sentence listening/reading",
            # Decoding contrasts
            "checkerboard",
            "left vs right button press",
            "sentence listening",
        ]

        # Number of contrasts used for alignment
        n_decoding_contrasts = 3

        subject_data = datasets.fetch_localizer_contrasts(
            contrasts=contrasts,
            n_subjects=[int(subject)],
            get_anats=True,
        )

        # Load the data in volumetric form
        alignment_contrasts_vol = load_img(
            subject_data.cmaps[:-n_decoding_contrasts]
        )
        decoding_contrasts_vol = load_img(
            subject_data.cmaps[-n_decoding_contrasts:]
        )

        # Create a SurfaceImage object for the alignment and decoding contrasts
        alignment_contrasts_surf = project_on_surf(
            alignment_contrasts_vol, mesh_name
        )
        decoding_contrasts_surf = project_on_surf(
            decoding_contrasts_vol, mesh_name
        )

        # Load labels
        labels = np.array(contrasts[-n_decoding_contrasts:])

        alignment_labeled = LabeledImage(
            img=alignment_contrasts_surf,
            labels=None,
        )

        decoding_labeled = LabeledImage(
            img=decoding_contrasts_surf,
            labels=labels,
        )

        return alignment_labeled, decoding_labeled

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Create a masker to extract the data from the brain volume.
        dict_alignment = dict()
        dict_decoding = dict()

        for subject in self.subjects:
            (
                data_alignment,
                data_decoding,
            ) = self.load_localizer_surf(subject, self.mesh_name)
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
