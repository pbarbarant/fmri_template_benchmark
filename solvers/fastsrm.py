from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from fastsrm.identifiable_srm import IdentifiableFastSRM
    import os
    import numpy as np
    from benchmark_utils.config import MEMORY


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "FastSRM"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "n_components": [20],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["pip:fastsrm", "joblib"]

    stopping_criterion = SingleRunCriterion()

    def set_objective(
        self,
        dict_alignment,
        dict_decoding,
        dict_labels,
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
        self.dict_labels = dict_labels
        self.masker = masker
        self.mesh_name = mesh_name

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # srm_path = os.path.join(MEMORY, "fastsrm")
        # if not os.path.exists(srm_path):
        #     os.makedirs(srm_path)

        srm = IdentifiableFastSRM(
            n_components=self.n_components,
            aggregate="mean",
            temp_dir=None,
            tol=1e-10,
            n_iter=100,
            n_jobs=5,
        )

        alignment_array = [
            self.masker.transform(contrasts).T
            for _, contrasts in self.dict_alignment.items()
        ]
        alignment_estimator = srm.fit(alignment_array)

        self.X = np.concatenate(
            [
                alignment_estimator.transform(
                    [self.masker.transform(self.dict_decoding[subject]).T]
                ).T
                for subject in self.dict_decoding.keys()
            ]
        )
        self.y = np.concatenate(
            np.array(list(self.dict_labels.values())), axis=0
        )

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(
            aligned_dataset=(self.X, self.y, self.name),
        )
