from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from benchmark_utils.utils import LabeledImage
    import numpy as np
    from fmralign.template_alignment import TemplateAlignment
    from fmralign.alignment_methods import POTAlignment


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "OptimalTransport"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "n_parcels": [
            3,
        ],
        "clustering": ["ward"],
        "scaling": [True],
        "reg": [0.1],
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    install_pip = "pip"
    requirements = []

    stopping_criterion = SingleRunCriterion()

    def set_objective(
        self,
        dict_alignment,
        dict_decoding,
    ):
        # Define the information received by each solver from the objective.
        # The arguments of this function are the results of the
        # `Objective.get_objective`. This defines the benchmark's API for
        # passing the objective to the solver.
        # It is customizable for each benchmark.
        self.dict_alignment = dict_alignment
        self.dict_decoding = dict_decoding
        self.dict_aligned = dict()

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        # Get the list of subjects
        subject_list = list(self.dict_alignment.keys())

        # Get the list of images
        imgs = [self.dict_alignment[subject].img for subject in subject_list]

        # Align the images
        algo = TemplateAlignment(
            alignment_method="optimal_transport",
            n_pieces=self.n_parcels,
            clustering=self.clustering,
            # scaling=self.scaling,
        )
        algo.fit(imgs)

        # Retrieve the parcellation, masker
        self.labels, self.parcellation_img = algo.get_parcellation()
        self.masker = algo.masker

        # Align the images
        template_data = np.zeros_like(
            self.masker.transform(self.dict_decoding[subject_list[0]].img)
        )
        for i, subject in enumerate(subject_list):
            transformed_img = algo.transform(
                self.dict_decoding[subject].img, subject_index=i
            )
            self.dict_aligned[subject] = LabeledImage(
                img=transformed_img,
                y=self.dict_decoding[subject].y,
            )
            template_data += self.masker.transform(transformed_img) / len(
                subject_list
            )

        # Generate the template
        self.template = LabeledImage(
            img=self.masker.inverse_transform(template_data),
            y=self.dict_decoding[subject_list[0]].y,
        )

    def get_result(self):
        # Return the result from one optimization run.
        # The outputs of this function is a dictionary which defines the
        # keyword arguments for `Objective.evaluate_result`
        # This defines the benchmark's API for solvers' results.
        # it is customizable for each benchmark.
        solver_name = (
            self.name
            + f"_{self.clustering}_{self.n_parcels}_sc_{self.scaling}"
        )
        return dict(
            aligned_dataset=(
                self.parcellation_img,
                self.labels,
                self.masker,
                self.template,
                self.dict_aligned,
                solver_name,
            ),
        )
