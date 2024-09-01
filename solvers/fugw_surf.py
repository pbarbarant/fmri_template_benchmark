from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from fugw.mappings import FUGWBarycenter
    from fugw.datasets import fetch_surf_geometry
    import numpy as np
    import torch
    from nilearn import surface, plotting
    from nilearn.datasets import fetch_surf_fsaverage


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "FUGW_Surf"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "alpha": [0.5],
        "rho": [float("inf")],
        "eps": [1e-4],
        "resolution": ["fsaverage4"],
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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.nits_barycenter = 10
        self.nits_bcd = 5
        self.nits_uot = 100
        print("Device:", self.device)

    def _project(self, features, plan):
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
        source_features_tensor = torch.tensor(
            features, dtype=torch.float32, device=self.device
        )
        transformed_data = (
            (
                (plan.T @ source_features_tensor.T)
                / (
                    plan.sum(dim=0).reshape(-1, 1)
                    # Add very small value to handle null rows
                    + 1e-16
                )
            )
            .T.detach()
            .cpu()
        )
        return transformed_data.numpy()

    def _normalize(self, features):
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
        return features / np.linalg.norm(features, axis=1).reshape(-1, 1)

    def _compute_bary_plans(
        self,
        features_list,
        weights_list,
        geometry,
        device,
        hemi,
    ):
        """Compute the barycenter and the transport plans

        Parameters
        ----------
        features_list : list of ndarray of shape (n_samples, n_features)
            List of source features

        weights_list : list of ndarray of shape (n_features,)
            List of source weights

        geometry : ndarray of shape (n_features,)
            Geometry of the target space

        Returns
        -------
        ndarray
            Barycenter features

        list of ndarray of shape (n_features, n_samples)
            List of transport plans
        """
        euclidean_mean = np.mean(features_list, axis=0)
        fugw_barycenter = FUGWBarycenter(
            alpha=self.alpha,
            rho=self.rho,
            eps=self.eps,
        )
        _, barycenter_features, _, plans, _, _ = fugw_barycenter.fit(
            weights_list,
            features_list,
            [geometry],
            nits_barycenter=self.nits_barycenter,
            device=device,
            init_barycenter_features=euclidean_mean,
            solver="mm",
            solver_params={
                "nits_bcd": self.nits_bcd,
                "nits_uot": self.nits_uot,
            },
            verbose=True,
        )

        fsaverage = fetch_surf_fsaverage(mesh=self.resolution)
        plotting.plot_surf(
            fsaverage[f"infl_{hemi}"],
            barycenter_features[0, :].cpu().numpy(),
            bg_map=fsaverage[f"sulc_{hemi}"],
            hemi=hemi,
            cmap="coolwarm",
            title="Barycenter",
            output_file=f"barycenter_{hemi}.png",
        )
        plotting.plot_surf(
            fsaverage[f"infl_{hemi}"],
            euclidean_mean[0, :],
            bg_map=fsaverage[f"sulc_{hemi}"],
            hemi=hemi,
            cmap="coolwarm",
            title="Euclidean mean",
            output_file=f"euclidean_mean_{hemi}.png",
        )
        return plans

    def _barycenter_hemi(self, hemi, subject_list, device):
        fsaverage = fetch_surf_fsaverage(mesh=self.resolution)
        print(f"Computing features for {hemi} hemisphere")
        features_list = [
            self._normalize(
                np.nan_to_num(
                    surface.vol_to_surf(
                        self.dict_alignment[subject],
                        surf_mesh=fsaverage[f"infl_{hemi}"],
                    ).T
                )
            )
            for subject in subject_list
        ]

        print(f"Computing geometry for {hemi} hemisphere")
        geometry, d_max = fetch_surf_geometry(
            f"infl_{hemi}",
            method="euclidean",
            resolution=self.resolution,
        )
        geometry /= d_max
        n_voxels = features_list[0].shape[1]

        # Weights are uniform
        weights_list = [np.ones(n_voxels) / n_voxels for _ in features_list]
        plans = self._compute_bary_plans(
            features_list, weights_list, geometry, device=device, hemi=hemi
        )

        # Generate a dictionary of plans for each subject
        self.plans = dict(zip(subject_list, plans))

        # Project and concatenate the transformed data for each subject
        X_hemi = np.concatenate(
            [
                self._project(
                    np.nan_to_num(
                        surface.vol_to_surf(
                            self.dict_decoding[subject],
                            surf_mesh=fsaverage[f"infl_{hemi}"],
                        ).T
                    ),
                    plan,
                )
                for subject, plan in self.plans.items()
            ],
            axis=0,
        )

        return X_hemi

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # List of source subjects

        subject_list = list(self.dict_alignment.keys())
        self.X = np.concatenate(
            [
                self._barycenter_hemi(hemi, subject_list, device=self.device)
                for hemi in ["left", "right"]
            ],
            axis=1,
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
