from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from fugw.mappings import FUGWSparseBarycenter
    from fugw.scripts import coarse_to_fine, lmds
    import numpy as np
    import torch
    from nilearn import masking


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "FUGW"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "alpha": [0.5],
        "rho": [1e2],
        "eps": [1e-2],
        "nits_barycenter": [10],
        "radius": [7],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["pip:fmralign", "joblib"]

    stopping_criterion = SingleRunCriterion()

    def set_objective(
        self,
        dict_alignment,
        dict_decoding,
        dict_labels,
        mask,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.dict_alignment = dict_alignment
        self.dict_decoding = dict_decoding
        self.dict_labels = dict_labels
        self.mask = mask
        self.folds_dict = dict()
        self.anisotropy = tuple(
            np.abs(self.mask.mask_img_.affine.diagonal()[:3])
        )
        # Get main connected component of segmentation
        self.segmentation = (
            masking.compute_background_mask(
                self.mask.mask_img_, connected=True
            ).get_fdata()
            > 0
        )
        self.n_samples = 1000 if self.anisotropy[0] < 3 else 3000
        print("Segmentation shape:", self.segmentation.shape)
        print("Anisotropy shape:", self.anisotropy)
        print("Number of samples:", self.n_samples)

    def sample_geometry(self, segmentation, geometry_embedding, n_samples):
        """Sample the geometry of the mask"""
        return coarse_to_fine.sample_volume_uniformly(
            segmentation,
            embeddings=geometry_embedding,
            n_samples=n_samples,
        )

    def prepare_geometry_embedding(
        self, segmentation, n_landmarks, anisotropy, verbose
    ):
        """Compute the normalized geometry embedding"""
        geometry_embedding = lmds.compute_lmds_volume(
            segmentation,
            k=12,
            n_landmarks=n_landmarks,
            anisotropy=anisotropy,
            verbose=verbose,
        ).nan_to_num()

        (
            geometry_embedding_normalized,
            max_distance,
        ) = coarse_to_fine.random_normalizing(geometry_embedding)

        return (
            geometry_embedding,
            geometry_embedding_normalized,
            max_distance,
        )

    def project(self, features, plan):
        """Project features using the given transport plan

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Source features

        Returns
        -------
        ndarray
            Projected features
        """
        source_features_tensor = torch.tensor(features, dtype=torch.float32)
        transformed_data = (
            (
                torch.sparse.mm(
                    plan.to("cpu").transpose(0, 1),
                    source_features_tensor.T,
                ).to_dense()
                / (
                    torch.sparse.sum(plan.to("cpu"), dim=0)
                    .to_dense()
                    .reshape(-1, 1)
                    # Add very small value to handle null rows
                    + 1e-16
                )
            )
            .T.detach()
            .cpu()
        )
        return transformed_data.numpy()

    def normalize(self, features):
        """Normalize the features between -1 and 1

        Parameters
        ----------
        features : ndarray of shape (n_samples, n_features)
            Features to normalize

        Returns
        -------
        ndarray
            Normalized features
        """
        return (
            2
            * (features - features.min(axis=0))
            / (features.max(axis=0) - features.min(axis=0))
            - 1
        )

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # List of source subjects
        subject_list = list(self.dict_alignment.keys())

        nits_bcd = 5
        nits_uot = 100
        features_list = [
            self.normalize(self.mask.transform(self.dict_alignment[subject]))
            for subject in subject_list
        ]
        n_voxels = features_list[0].shape[1]

        # Weights are uniform
        weights_list = [np.ones(n_voxels) / n_voxels for _ in features_list]

        _, geometry_embedding_normalized, max_distance = (
            self.prepare_geometry_embedding(
                self.segmentation,
                n_landmarks=100,
                anisotropy=self.anisotropy,
                verbose=True,
            )
        )

        mesh_sample = self.sample_geometry(
            self.segmentation,
            geometry_embedding_normalized,
            self.n_samples,
        )

        # Compute the Barycenter
        sparse_barycenter = FUGWSparseBarycenter(
            alpha_coarse=self.alpha,
            alpha_fine=self.alpha,
            rho_coarse=self.rho,
            rho_fine=self.rho,
            eps_coarse=self.eps,
            eps_fine=self.eps,
            selection_radius=self.radius / max_distance,
        )
        (
            _,
            _,
            plans,
            _,
        ) = sparse_barycenter.fit(
            weights_list,
            features_list,
            geometry_embedding_normalized,
            mesh_sample=mesh_sample,
            nits_barycenter=self.nits_barycenter,
            init_barycenter_features=np.mean(features_list, axis=0),
            solver="mm",
            coarse_mapping_solver_params={
                "nits_bcd": nits_bcd,
                "nits_uot": nits_uot,
            },
            fine_mapping_solver_params={
                "nits_bcd": nits_bcd,
                "nits_uot": nits_uot,
            },
            device="auto",
            verbose=True,
        )

        # Generate a dict of plan for each subject
        self.plans = dict()
        for subject, plan in zip(subject_list, plans):
            self.plans[subject] = plan

        self.X = np.concatenate(
            [
                self.project(
                    self.mask.transform(self.dict_decoding[subject]),
                    self.plans[subject],
                )
                for subject in subject_list
            ],
            axis=0,
        )

        self.y = np.concatenate(
            np.array(list(self.dict_labels.values())), axis=0
        )

        print("X shape:", self.X.shape)
        print("y shape:", self.y.shape)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(
            aligned_dataset=(self.X, self.y, self.name),
        )
