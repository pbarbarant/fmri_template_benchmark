from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn._utils.data_gen import generate_random_img
    from benchmark_utils.utils import LabeledImage


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    def __init__(
        self,
    ):
        self.subjects = ["sub-01", "sub-02", "sub-03"]
        self.n_samples_alignement = 200
        self.n_samples_decoding = 150

    def _sample_labels(self, n_samples):
        return (
            np.arange(3)
            .reshape(1, -1)
            .repeat(n_samples // 3, axis=0)
            .flatten()
        )

    def _sample_labeled_image(self, n_samples):
        img, _ = generate_random_img((8, 7, 6, n_samples))
        y = self._sample_labels(n_samples)
        return LabeledImage(
            img=img,
            y=y,
        )

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.
        dict_alignment = dict()
        dict_decoding = dict()

        for subject in self.subjects:
            # Generate random surface images for each subject.
            dict_alignment[subject] = self._sample_labeled_image(
                self.n_samples_alignement
            )
            dict_decoding[subject] = self._sample_labeled_image(
                self.n_samples_decoding
            )

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            dataset_name=self.name,
            dict_alignment=dict_alignment,
            dict_decoding=dict_decoding,
        )
