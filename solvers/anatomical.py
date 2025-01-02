from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.stopping_criterion import SingleRunCriterion
    from benchmark_utils.utils import LabeledImage, DecodingFold
    from fmralign.template_alignment import TemplateAlignment


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Anatomical"

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

    def _compute_template_one_fold(
        self, dict_alignment, dict_decoding, masker, clustering_img
    ):
        # Get the list of subjects
        subject_list = list(dict_alignment.keys())

        # Get the list of images
        imgs = [dict_alignment[subject].img for subject in subject_list]

        # Align the images
        algo = TemplateAlignment(
            alignment_method="identity",
            mask=masker,
            clustering=clustering_img,
        )
        algo.fit(imgs)

        # Align the images
        template_data = np.zeros_like(
            masker.transform(dict_decoding[subject_list[0]].img)
        )
        dict_aligned = dict()
        for i, subject in enumerate(subject_list):
            transformed_img = algo.transform(
                dict_decoding[subject].img, subject_index=i
            )
            dict_aligned[subject] = LabeledImage(
                img=transformed_img,
                y=dict_decoding[subject].y,
            )
            template_data += masker.transform(transformed_img) / len(
                subject_list
            )

        # Generate the template
        template = LabeledImage(
            img=masker.inverse_transform(template_data),
            y=dict_decoding[subject_list[0]].y,
        )

        return template, dict_aligned

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html
        decoding_folds = []
        for fold in self.folds:
            print(f"Running {self.name} solver on fold {fold.name}")
            template, dict_aligned = self._compute_template_one_fold(
                fold.dict_alignment,
                fold.dict_decoding,
                self.masker,
                self.clustering_img,
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
