from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.stopping_criterion import SingleRunCriterion
    from benchmark_utils.procrustes_utils import (
        compute_parcellation,
        compute_alignments,
        compute_template,
        plot_parcellation,
        project,
    )


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Procrustes"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "nits": [10],
        "n_parcels": [200],
        "clustering": ["ward"],
        "scaling": [False],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = ["numpy", "nilearn", "joblib"]

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

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # Get the list of subjects
        subject_list = list(self.dict_alignment.keys())
        mesh = next(iter(self.dict_alignment.values())).img.mesh

        # Compute the parcellation
        parcellation_data = np.concatenate(
            [
                self.masker.transform(self.dict_alignment[subject].img)
                for subject in subject_list
            ],
            axis=0,
        )
        parcellation_labels = compute_parcellation(
            parcellation_data,
            clustering=self.clustering,
            n_parcels=self.n_parcels,
            mesh=mesh,
        )

        # Plot the parcellation
        plot_parcellation(
            mesh=mesh,
            labels=parcellation_labels,
            clustering=self.clustering,
            n_parcels=self.n_parcels,
        )

        # Compute the Procrustes alignment
        R_list, sc_list = compute_alignments(
            self.dict_alignment,
            parcellation_labels=parcellation_labels,
            masker=self.masker,
            scaling=self.scaling,
            n_iter=self.nits,
            n_jobs=10,
        )

        # Compute the barycenter
        self.template = compute_template(
            self.dict_decoding,
            parcellation_labels=parcellation_labels,
            masker=self.masker,
            R_list=R_list,
            sc_list=sc_list,
        )

        # Align the data
        self.dict_aligned = {
            subject: project(
                self.dict_decoding[subject].img,
                masker=self.masker,
                labels=parcellation_labels,
                R_list=R_list,
                sc_list=sc_list,
                n_sub=i,
                classes=self.dict_decoding[subject].labels,
            )
            for i, subject in enumerate(subject_list)
        }

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        solver_name = (
            self.name
            + f"_niter_{self.nits}_{self.clustering}_{self.n_parcels}_sc_{self.scaling}"
        )
        return dict(
            aligned_dataset=(
                self.template,
                self.dict_aligned,
                solver_name,
            ),
        )
