from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    # import warnings
    import numpy as np
    from sklearn.svm import LinearSVC

    # from sklearn.exceptions import ConvergenceWarning


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
        dict_alignment,
        dict_decoding,
        dict_labels,
        mask,
    ):
        # The keyword arguments of this function are the keys of the dictionary
        # returned by `Dataset.get_data`. This defines the benchmark's
        # API to pass data. This is customizable for each benchmark.
        self.dict_alignment = dict_alignment
        self.dict_decoding = dict_decoding
        self.dict_labels = dict_labels
        self.mask = mask

    def compute_score(self, X_train, y_train, X_test, y_test):
        clf = LinearSVC(max_iter=int(self.max_iter))
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)

    def evaluate_result(self, folds_dict):
        # The keyword arguments of this function are the keys of the
        # dictionary returned by `Solver.get_result`. This defines the
        # benchmark's API to pass solvers' result. This is customizable for
        # each benchmark.

        # Fit a linear SVM on the training data and evaluate the score on the
        # test data.
        score_dict = dict()
        for subject in folds_dict.keys():
            X_train = folds_dict[subject]["X_train"]
            y_train = folds_dict[subject]["y_train"]
            X_test = folds_dict[subject]["X_test"]
            y_test = folds_dict[subject]["y_test"]
            score_dict[subject] = self.compute_score(X_train, y_train, X_test, y_test)

        avg_score = np.mean(list(score_dict.values()))
        print(f"Average decoding accuracy: {avg_score:.2f}")
        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=avg_score,
            scores=list(score_dict.values()),
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        result = dict(
            X_train=np.random.randn(10, 10),
            y_train=np.random.randint(2, size=10),
            X_test=np.random.randn(10, 10),
            y_test=np.random.randint(2, size=10),
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
            mask=self.mask,
        )
