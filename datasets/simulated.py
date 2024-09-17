from benchopt import BaseDataset, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn.experimental import surface
    from nilearn.experimental.surface._datasets import load_fsaverage
    from nilearn.experimental.surface._surface_image import SurfaceImage
    from benchmark_utils.utils import LabeledImage


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):
    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"
    mesh_name = "fsaverage5"

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["nilearn", "numpy"]

    def __init__(
        self,
    ):
        self.subjects = ["sub-01", "sub-02", "sub-03"]
        self.n_samples_alignement = 200
        self.n_samples_decoding = 150

    def _sample_data_hemi(self, n_samples, n_vertices):
        return np.random.randn(n_samples, n_vertices)

    def _sample_data_subject(self, n_samples, n_vertices):
        left_data = self._sample_data_hemi(n_samples, n_vertices)
        right_data = self._sample_data_hemi(n_samples, n_vertices)
        return left_data, right_data

    def _sample_surface_image(self, mesh, n_samples, n_vertices):
        left_data, right_data = self._sample_data_subject(
            n_samples, n_vertices
        )
        return SurfaceImage(
            mesh=mesh,
            data={
                "left": left_data,
                "right": right_data,
            },
        )

    def _sample_labels(self, n_samples):
        return (
            np.arange(3)
            .reshape(1, -1)
            .repeat(n_samples // 3, axis=0)
            .flatten()
        )

    def _sample_labeled_image(self, mesh, n_samples, n_vertices):
        img = self._sample_surface_image(mesh, n_samples, n_vertices)
        labels = self._sample_labels(n_samples)
        return LabeledImage(
            img=img,
            labels=labels,
        )

    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Create a masker to extract the data from the brain volume.
        mesh = load_fsaverage(self.mesh_name)["pial"]
        n_vertices = mesh.n_vertices // 2

        dict_alignment = dict()
        dict_decoding = dict()

        for subject in self.subjects:
            # Generate random surface images for each subject.
            dict_alignment[subject] = self._sample_labeled_image(
                mesh, self.n_samples_alignement, n_vertices
            )
            dict_decoding[subject] = self._sample_labeled_image(
                mesh, self.n_samples_decoding, n_vertices
            )

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
