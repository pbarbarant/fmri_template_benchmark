from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    # import warnings
    from pathlib import Path

    import numpy as np
    import joblib

    from sklearn import neighbors
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyClassifier
    from sklearn.utils import shuffle

    from nilearn import datasets, surface, decoding, plotting
    from nilearn._utils import param_validation


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):
    # Name to select the objective in the CLI and to display the results.
    name = "fMRI decoding"

    # URL of the main repo for this benchmark.
    url = "https://github.com/pbarbarant/fmri_alignment_benchmark"

    # List of parameters for the objective. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    # This means the OLS objective will have a parameter `self.whiten_y`.
    parameters = {
        "max_iter": [1e2],
        "mesh": ["fsaverage5"],
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    install_pip = "pip"
    requirements = [
        "pip:fmralign",
        "pip:fastsrm",
        "scikit-learn",
        "numpy",
        "joblib",
    ]

    # warnings.filterwarnings("ignore", category=FutureWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(
        self,
        dataset_name,
        dict_alignment,
        dict_decoding,
        dict_labels,
        masker,
    ):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.dataset_name = dataset_name
        self.dict_alignment = dict_alignment
        self.dict_decoding = dict_decoding
        self.dict_labels = dict_labels
        self.masker = masker

        print(f"Running on: {dataset_name}")

    def _project_on_surface(self, X, hemi="left", mesh="fsaverage5"):
        """Project data on fsaverage surface"""
        fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
        pial_mesh = fsaverage[f"pial_{hemi}"]
        img = self.mask.inverse_transform(X)
        X_hemi = surface.vol_to_surf(img, pial_mesh).T
        return X_hemi

    def _compute_adjacency_matrix(self, hemi="left", mesh="fsaverage5"):
        fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
        infl_mesh = fsaverage[f"infl_{hemi}"]
        coords, _ = surface.load_surf_mesh(infl_mesh)
        radius = 3.0
        nn = neighbors.NearestNeighbors(radius=radius)
        adjacency = nn.fit(coords).radius_neighbors_graph(coords).tolil()
        return adjacency

    def _compute_searchlight_scores(self, estimator, X, y, adjacency, cv=3):
        # Cross-validated search light
        scores = decoding.searchlight.search_light(
            X, y, estimator, adjacency, cv=cv, n_jobs=10
        )
        return scores

    def _plot_searchlight_scores(
        self,
        estimator,
        X,
        y,
        cv=None,
        hemi="left",
        mesh="fsaverage5",
        threshold=0.0,
        solver_name="",
        output_dir=Path(__file__).parent / "figures",
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh)
        inflated_mesh = fsaverage[f"infl_{hemi}"]

        X_hemi = self._project_on_surface(X, hemi, mesh)

        # Plot the searchlight scores
        scores = self._compute_searchlight_scores(
            estimator,
            X_hemi,
            y,
            adjacency=self._compute_adjacency_matrix(hemi, mesh),
            cv=cv,
        )

        # Save scores with joblib
        joblib.dump(scores, output_dir / f"{hemi}_searchlight_scores.pkl")

        # Plot the scores on the surface
        fig = plotting.plot_surf_stat_map(
            inflated_mesh,
            scores,
            hemi=hemi,
            title=f"Accuracy map, {hemi} hemisphere",
            threshold=threshold,
            bg_map=fsaverage[f"sulc_{hemi}"],
        )
        fig.savefig(output_dir / f"{hemi}_searchlight_scores.png")

    def _compute_decoding_scores(self, X, y, groups, estimator="svc"):
        # Patch the decoder to avoid using a NiftiMasker
        def monkeypatch_masker_checks():
            def adjust_screening_percentile(
                screening_percentile, *args, **kwargs
            ):
                return screening_percentile

            param_validation.adjust_screening_percentile = (
                adjust_screening_percentile
            )

        monkeypatch_masker_checks()

        # Create a decoder
        decoder = decoding.Decoder(
            estimator=estimator,
            mask=self.masker,
            standardize="zscore_sample",
            screening_percentile=5,
            scoring="accuracy",
            cv=LeaveOneGroupOut(),
        )
        decoder.fit(X, y, groups=groups)
        cv_scores = decoder.cv_scores_
        # Turn cv_scores into a matrix
        cv_scores = (
            np.array(list(cv_scores.values()))
            .reshape(len(np.unique(groups)), -1)
            .mean(axis=1)
        )
        return cv_scores.flatten()

    def evaluate_result(self, aligned_dataset):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        X, y, groups, solver_name = aligned_dataset
        cv_scores_svc = self._compute_decoding_scores(X, y, groups)
        cv_scores_dummy = self._compute_decoding_scores(
            X,
            y,
            groups,
            estimator="dummy_classifier",
        )
        avg_score = np.mean(cv_scores_svc)
        # for hemi in ["left", "right"]:
        #     self._plot_searchlight_scores(
        #         svc_estimator,
        #         X,
        #         y,
        #         cv,
        #         hemi=hemi,
        #         mesh=self.mesh,
        #         threshold=chance,
        #         solver_name=solver_name,
        #         output_dir=Path(__file__).parent
        #         / "figures"
        #         / self.dataset_name
        #         / solver_name,
        #     )

        print(f"Average decoding accuracy: {avg_score:.2f}")
        print(f"Chance level: {np.mean(cv_scores_dummy):.2f}")
        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=avg_score,
            cv_scores=cv_scores_svc,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        result = dict(
            aligned_dataset=(
                np.random.randn(100, 10),
                np.random.randint(2, size=100),
            )
        )
        return result

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            dict_alignment=self.dict_alignment,
            dict_decoding=self.dict_decoding,
            dict_labels=self.dict_labels,
            masker=self.masker,
        )
