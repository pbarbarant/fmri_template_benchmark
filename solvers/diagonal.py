from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from benchmark_utils.utils import DecodingFold
    from benchmark_utils.solver_utils import _compute_template_one_fold
    from fmralign.template_alignment import TemplateAlignment


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Diagonal"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {}

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    stopping_criterion = SingleRunCriterion()

    def set_objective(
        self,
        folds,
        masker,
        clustering_img,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.folds = folds
        self.masker = masker
        self.clustering_img = clustering_img

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        algo = TemplateAlignment(
            alignment_method="diagonal",
            mask=self.masker,
            clustering=self.clustering_img,
        )
        
        decoding_folds = []
        for fold in self.folds:
            print(f"Running {self.name} solver on fold {fold.name}")
            template, dict_aligned = _compute_template_one_fold(
                algo,
                fold.dict_alignment,
                fold.dict_decoding,
                self.masker,
            )

            decoding_fold = DecodingFold(
                name=fold.name, template=template, dict_aligned=dict_aligned
            )
            decoding_folds.append(decoding_fold)

        self.decoding_folds = decoding_folds

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        return dict(
            decoding_folds=self.decoding_folds,
        )
