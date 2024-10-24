from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    from benchopt.stopping_criterion import SingleRunCriterion
    from fugw.datasets import fetch_surf_geometry
    from fugw.mappings import FUGWBarycenter
    from benchmark_utils.utils import LabeledImage


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "FUGW"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
        "eps": [1e-6],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    stopping_criterion = SingleRunCriterion()

    def set_objective(
        self,
        dict_alignment,
        dict_decoding,
        masker,
        mesh_name,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.dict_alignment = dict_alignment
        self.dict_decoding = dict_decoding
        self.masker = masker
        self.mesh_name = mesh_name

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.nits_barycenter = 2
        self.nits_bcd = 5
        self.nits_uot = 100
        print("Device:", self.device)

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

    def _compute_plans_hemi(
        self,
        subject_list,
        device,
        hemi,
    ):
        """Compute the barycenter and the transport plans

        Parameters
        ----------

        Returns
        -------
        ndarray
            Barycenter features

        list of ndarray of shape (n_features, n_samples)
            List of transport plans
        """
        print(f"Computing features for {hemi} hemisphere")
        features_list = [
            self._normalize(
                np.nan_to_num(
                    self.dict_alignment[subject].img.data.parts[hemi]
                )
            )
            for subject in subject_list
        ]

        print(f"Computing geometry for {hemi} hemisphere")
        geometry, d_max = fetch_surf_geometry(
            f"pial_{hemi}",
            method="euclidean",
            resolution=self.mesh_name,
        )
        geometry /= d_max
        n_voxels = features_list[0].shape[1]

        # Weights are uniform
        weights_list = [np.ones(n_voxels) / n_voxels for _ in features_list]

        euclidean_mean = np.mean(features_list, axis=0)
        fugw_barycenter = FUGWBarycenter(
            alpha=self.alpha,
            rho=float("inf"),
            eps=self.eps,
        )
        _, barycenter_features_hemi, _, plans, _, _ = fugw_barycenter.fit(
            weights_list,
            features_list,
            [geometry],
            nits_barycenter=self.nits_barycenter,
            device=device,
            init_barycenter_geometry=geometry,
            init_barycenter_features=euclidean_mean,
            solver="mm",
            solver_params={
                "nits_bcd": self.nits_bcd,
                "nits_uot": self.nits_uot,
            },
            verbose=True,
        )

        # Generate a dictionary of plans for each subject
        plans_hemi = dict(zip(subject_list, plans))
        return plans_hemi, barycenter_features_hemi

    def _compute_plans(
        self,
        subject_list,
        masker,
        device,
    ):
        plans_left, barycenter_features_left = self._compute_plans_hemi(
            subject_list,
            device,
            "left",
        )
        plans_right, barycenter_features_right = self._compute_plans_hemi(
            subject_list,
            device,
            "right",
        )

        plans = dict()
        for subject in subject_list:
            plans[subject] = {
                "left": plans_left[subject],
                "right": plans_right[subject],
            }

        # Send to cpu and numpy
        barycenter_features_left = (
            barycenter_features_left.detach().cpu().numpy()
        )
        barycenter_features_right = (
            barycenter_features_right.detach().cpu().numpy()
        )
        barycenter_features = masker.inverse_transform(
            np.concatenate(
                [barycenter_features_left, barycenter_features_right], axis=1
            )
        )
        barycenter = LabeledImage(
            img=barycenter_features,
            labels=self.dict_alignment[subject_list[0]].labels,
        )

        return plans, barycenter

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

    def _project_img(self, subject_data, masker, plan):
        labels = subject_data.labels
        img = subject_data.img
        features = []
        for hemi in ["left", "right"]:
            features_hemi = img.data.parts[hemi]
            features.append(self._project(features_hemi, plan[hemi]))
        projected_data = np.concatenate(features, axis=1)
        return LabeledImage(
            img=masker.inverse_transform(projected_data),
            labels=labels,
        )

    def _compute_template(self, dict_subjects, masker, plans):
        subject_list = list(dict_subjects.keys())
        first_subject_data = masker.transform(
            dict_subjects[subject_list[0]].img
        )
        features = np.zeros_like(first_subject_data)
        for subject in subject_list:
            subject_data = dict_subjects[subject]
            subject_img = subject_data.img
            features_left = self._project(
                subject_img.data.parts["left"], plans[subject]["left"]
            )
            features_right = self._project(
                subject_img.data.parts["right"], plans[subject]["right"]
            )
            fused_features = np.concatenate(
                [features_left, features_right], axis=1
            )
            features += fused_features / len(subject_list)

        template_img = masker.inverse_transform(features)
        template_labels = dict_subjects[subject_list[0]].labels
        return LabeledImage(img=template_img, labels=template_labels)

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # List of source subjects
        subject_list = list(self.dict_alignment.keys())

        # Compute the transport plans and the barycenter
        plans, barycenter = self._compute_plans(
            subject_list=subject_list,
            masker=self.masker,
            device=self.device,
        )
        self.barycenter = barycenter  # TODO: do smth with it

        # Compute the decoding template using the transport plans
        self.decoding_template = self._compute_template(
            dict_subjects=self.dict_decoding,
            masker=self.masker,
            plans=plans,
        )

        # Align the data
        self.dict_aligned = {
            subject: self._project_img(
                subject_data=self.dict_decoding[subject],
                masker=self.masker,
                plan=plans[subject],
            )
            for subject in subject_list
        }

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        solver_name = self.name + f"_alpha_{self.alpha}_eps_{self.eps}"
        return dict(
            aligned_dataset=(
                self.decoding_template,
                self.dict_aligned,
                solver_name,
            ),
        )
