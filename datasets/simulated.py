from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn import maskers, datasets


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["nilearn", "pandas"]

    def __init__(
        self,
    ):
        self.subjects = ["sub-01", "sub-02", "sub-03"]
        self.n_samples_alignement = 150
        self.n_samples_decoding = 150
        self.n_features = 69765

    def generate_mock_data_subject(self, n_samples):
        data_decoding = np.random.randn(n_samples, self.n_features)
        return data_decoding

    def generate_fake_labels(self, n_samples):
        return np.random.randint(10, size=n_samples)

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Create a masker to extract the data from the brain volume.
        mask_img = datasets.load_mni152_brain_mask(resolution=3)
        mask = maskers.NiftiMasker(mask_img=mask_img).fit()

        dict_alignment = dict()
        dict_decoding = dict()
        dict_labels = dict()
        for subject in self.subjects:
            # Generate pseudorandom data using `numpy` for each subject.
            data_alignment = self.generate_mock_data_subject(
                n_samples=self.n_samples_alignement
            )
            data_decoding = self.generate_mock_data_subject(
                n_samples=self.n_samples_decoding
            )
            # Convert the data to a brain volume using the masker.
            data_alignment = mask.inverse_transform(data_alignment)
            data_decoding = mask.inverse_transform(data_decoding)
            # Generate pseudorandom labels using `numpy` for each subject.
            labels = self.generate_fake_labels(n_samples=self.n_samples_decoding)
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
