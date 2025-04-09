from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    import torch
    from fmralign.sparse_template_alignment import SparseTemplateAlignment
    from fmralign.sparse_pairwise_alignment import SparsePairwiseAlignment

    from benchmark_utils.conf import N_JOBS
    from benchmark_utils.solver_utils import compute_alignment


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "SparseOT"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {"reg": [0.1]}

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    stopping_criterion = SingleRunCriterion()

    def set_objective(
        self,
        dataset,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.dataset = dataset

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if self.dataset.target == "template":
            algo = SparseTemplateAlignment(
                masker=self.dataset.masker,
                clustering=self.dataset.clustering_img,
                device=self.device,
                n_jobs=N_JOBS,
                verbose=1,
                reg=self.reg,
                solver="sinkhorn_stabilized",
            )
        else:
            algo = SparsePairwiseAlignment(
                masker=self.dataset.masker,
                clustering=self.dataset.clustering_img,
                device=self.device,
                n_jobs=N_JOBS,
                verbose=1,
                reg=self.reg,
                solver="sinkhorn_stabilized",
            )

        self.dataset = compute_alignment(
            algo=algo,
            dataset=self.dataset,
            solver_name=self.name + f"_{self.reg}",
        )

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(dataset=self.dataset)
