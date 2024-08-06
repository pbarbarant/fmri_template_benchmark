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
        target,
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
        self.target = target
        self.mask = mask

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        X_train = []
        y_train = []
        X_test = []

        # List of source subjects
        source_subjects = list(self.dict_alignment.keys())
        source_subjects.remove(self.target)
        target_data_alignment = self.dict_alignment[self.target]
        target_data_decoding = self.dict_decoding[self.target]

        # Launch the alignments
        for source_subject in source_subjects:
            source_data_alignment = self.dict_alignment[source_subject]
            source_data_decoding = self.dict_decoding[source_subject]
            
            alignment_estimator = PairwiseAlignment(
                alignment_method="identity",
                n_pieces=self.n_pieces,
                mask=self.mask,
                memory=Memory(),
                memory_level=1,
            ).fit(source_data_alignment, target_data_alignment)

            aligned_data = alignment_estimator.transform(source_data_decoding)
            X_train.append(self.mask.transform(aligned_data))
            source_labels = self.dict_labels[source_subject]
            y_train.append(source_labels)

        # Train data
        X_train = np.vstack(X_train)
        self.y_train = np.hstack(y_train).ravel()

        # Test data
        X_test = self.mask.transform(target_data_decoding)
        self.y_test = self.dict_labels[self.target].ravel()

        # Standard scaling
        se = StandardScaler()
        self.X_train = se.fit_transform(X_train)
        self.X_test = se.transform(X_test)

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
        )
