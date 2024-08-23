from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from fmralign.template_alignment import TemplateAlignment
    from sklearn.preprocessing import StandardScaler
    from nilearn import image
    import numpy as np


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Ridge"

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
        target_train = list(self.dict_alignment.values())

        # Build a merged alignment/decoding dataset for each subject
        template_train = [
            image.concat_imgs(
                [self.dict_alignment[subject], self.dict_decoding[subject]]
            )
            for subject in subject_list
        ]
        train_index = range(target_train[0].shape[-1])
        test_index = range(
            target_train[0].shape[-1], template_train[0].shape[-1]
        )

        template_estim = TemplateAlignment(
            n_pieces=self.n_pieces,
            alignment_method="ridge_cv",
            mask=self.mask,
            n_jobs=10,
        )
        template_estim.fit(template_train)
        predicted_imgs = template_estim.transform(
            target_train, train_index, test_index
        )

        self.X = np.concatenate(
            [self.mask.transform(img) for img in predicted_imgs],
            axis=0,
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
