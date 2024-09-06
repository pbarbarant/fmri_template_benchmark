from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    # import warnings
    from pathlib import Path

    import joblib
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    from benchmark_utils.solver_utils import plot_surf_img


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
        mesh_name,
    ):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.dataset_name = dataset_name
        self.dict_alignment = dict_alignment
        self.dict_decoding = dict_decoding
        self.dict_labels = dict_labels
        self.masker = masker
        self.mesh_name = mesh_name

        print(f"Running on: {dataset_name}")

    def plot_aligned_dataset(self, X, y, groups, solver_name, dataset_name):
        contrasts = np.unique(y)
        subjects = np.unique(groups)
        for subject in subjects:
            X_subject = X[groups == subject]
            y_subject = y[groups == subject]
            for contrast in contrasts:
                avg_contrast = np.mean(
                    X_subject[y_subject == contrast], axis=0
                )
                img = self.masker.inverse_transform(avg_contrast)
                fig = plot_surf_img(
                    img,
                    colorbar=True,
                    cmap="coolwarm",
                )
                fig.suptitle(
                    f"Subject {subject} - {solver_name} - contrast {contrast}"
                )
                output_dir = (
                    Path(__file__).parent
                    / "figures/aligned_datasets"
                    / dataset_name
                    / solver_name
                    / f"{subject}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(output_dir / f"{contrast}.pdf")
                plt.close(fig)

    def evaluate_result(self, aligned_dataset):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        X, y, solver_name = aligned_dataset

        # Create cross-validation object on each subject
        groups = np.concatenate(
            [
                np.repeat(i, self.dict_decoding[subject].data.shape[0])
                for i, subject in enumerate(self.dict_decoding.keys())
            ]
        )

        self.plot_aligned_dataset(X, y, groups, solver_name, self.dataset_name)

        pipeline_svc = make_pipeline(
            StandardScaler(), LinearSVC(max_iter=int(self.max_iter))
        )

        cv_scores_svc = cross_val_score(
            pipeline_svc,
            X,
            y,
            groups=groups,
            cv=LeaveOneGroupOut(),
            n_jobs=10,
        )
        cv_scores_dummy = cross_val_score(
            DummyClassifier(strategy="most_frequent"),
            X,
            y,
            groups=groups,
            cv=LeaveOneGroupOut(),
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
                "random",
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
            mesh_name=self.mesh_name,
        )
