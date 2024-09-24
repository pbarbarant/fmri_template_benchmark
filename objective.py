from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import (
        LeaveOneGroupOut,
        cross_val_score,
        cross_validate,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    from benchmark_utils.utils import (
        generate_aligned_dataset_gii,
        generate_template_gii,
    )


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
    }

    # List of packages needed to run the benchmark.
    # They are installed with conda; to use pip, use 'pip:packagename'. To
    # install from a specific conda channel, use 'channelname:packagename'.
    # Packages that are not necessary to the whole benchmark but only to some
    # solvers or datasets should be declared in Dataset or Solver (see
    # simulated.py and python-gd.py).
    # Example syntax: requirements = ['numpy', 'pip:jax', 'pytorch:pytorch']
    install_pip = "pip"
    requirements = []

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.5"

    def set_data(
        self,
        dataset_name,
        dict_alignment,
        dict_decoding,
        masker,
        mesh_name,
    ):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.dataset_name = dataset_name
        self.dict_alignment = dict_alignment
        self.dict_decoding = dict_decoding
        self.masker = masker
        self.mesh_name = mesh_name

        print(f"Running on: {dataset_name}")

    def _compute_groups(self, subject_dict):
        groups = np.concatenate(
            [
                np.repeat(i, subject_dict[subject].img.data.shape[0])
                for i, subject in enumerate(subject_dict.keys())
            ]
        )
        return groups

    def _compute_X_y(self, subject_dict):
        subject_list = list(subject_dict.keys())
        X = np.concatenate(
            [
                self.masker.transform(subject_dict[subject].img)
                for subject in subject_list
            ]
        )
        y = np.concatenate(
            [subject_dict[subject].labels for subject in subject_list]
        )
        return X, y

    def evaluate_result(self, aligned_dataset):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        template, dict_aligned, solver_name = aligned_dataset

        # Create cross-validation object on each subject
        groups = self._compute_groups(self.dict_decoding)

        pipeline_svc = make_pipeline(
            StandardScaler(),
            LinearSVC(max_iter=int(self.max_iter), penalty="l1"),
        )

        X, y = self._compute_X_y(dict_aligned)

        cv_results_svc = cross_validate(
            pipeline_svc,
            X,
            y,
            groups=groups,
            cv=LeaveOneGroupOut(),
            n_jobs=10,
            return_estimator=True,  # This option will return the estimators
        )
        cv_scores_svc = cv_results_svc["test_score"]
        fitted_estimators = cv_results_svc["estimator"]
        dict_estimators = {
            subject: estimator
            for subject, estimator in zip(
                list(dict_aligned.keys()), fitted_estimators
            )
        }

        # Plot the template
        generate_template_gii(
            template=template,
            masker=self.masker,
            solver_name=solver_name,
            dataset_name=self.dataset_name,
        )

        # Plot the aligned features
        generate_aligned_dataset_gii(
            dict_aligned=dict_aligned,
            dict_estimators=dict_estimators,
            solver_name=solver_name,
            dataset_name=self.dataset_name,
            subjects_list=list(self.dict_decoding.keys()),
            masker=self.masker,
        )

        cv_scores_dummy = cross_val_score(
            DummyClassifier(strategy="most_frequent"),
            X,
            y,
            groups=groups,
            cv=LeaveOneGroupOut(),
        )

        avg_score = np.mean(cv_scores_svc)

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
            masker=self.masker,
            mesh_name=self.mesh_name,
        )
