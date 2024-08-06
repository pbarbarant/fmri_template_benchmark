from numpy import source
import pandas as pd
from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from fmralign.pairwise_alignment import PairwiseAlignment
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    from joblib import Memory


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Anatomical"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "n_pieces": [300],
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

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # List of source subjects
        subject_list = list(self.dict_alignment.keys())

        for left_out_subject in subject_list:
            # Train data
            X_train = np.vstack(
                [
                    self.mask.transform(self.dict_decoding[subject])
                    for subject in subject_list
                    if subject != left_out_subject
                ]
            )
            self.y_train = np.hstack(
                [
                    self.dict_labels[subject]
                    for subject in subject_list
                    if subject != left_out_subject
                ]
            ).ravel()

            # Test data
            X_test = self.mask.transform(self.dict_decoding[left_out_subject])
            self.y_test = self.dict_labels[left_out_subject].ravel()

            # Standard scaling
            se = StandardScaler()
            self.X_train = se.fit_transform(X_train)
            self.X_test = se.transform(X_test)

            self.folds_dict[left_out_subject] = dict(
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
            )

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(
            folds_dict=self.folds_dict,
        )
