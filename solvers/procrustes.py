from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from joblib import Parallel, delayed
    from functools import partial
    from benchopt.stopping_criterion import SingleRunCriterion
    from benchmark_utils.procrustes_utils import (
        rescaled_euclidean_mean,
        align_images_to_template,
        compute_parcellation,
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
        "n_parcels": [100,],
        "clustering": ["ward"],
        "scaling": [True],
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

    def template_procrustes(self, imgs, n_iter=2, scaling=False, primal=None):
        """
        Compute the template of a set of images using Procrustes analysis


        Parameters
        ----------
        imgs: (n_subjects, n_features, n_vertices) nd array
            set of images
        n_iter: int, optional
            number of iterations
        scaling: bool, optional
            If scaling is true, computes a floating scaling parameter
            sc such that:
            ||sc * RX - Y||^2 is minimized and
            - R is an orthogonal matrix
            - sc is a scalar
            If scaling is false sc is set to 1
        primal: bool or None, optional,
            Whether the SVD is done on the YX^T (primal) or Y^TX (dual)
            if None primal is used iff n_features <= n_timeframes

        Returns
        ----------
        X: (n_features, n_vertices) nd array
            template
        R_list: list of (n_features, n_features) nd array
            list of transformation matrices
        sc_list: list of int
            list of scaling parameters
        """
        aligned_imgs = imgs
        for _ in range(n_iter):
            template = rescaled_euclidean_mean(aligned_imgs, scaling)
            aligned_imgs, R_list, sc_list = align_images_to_template(
                imgs,
                template,
            )
        return template, R_list, sc_list

    def compute_alignments(
        self,
        dict_subjects,
        parcellation_labels,
        masker,
        scaling=False,
        n_iter=2,
        n_jobs=10,
    ):
        """
        Compute the template and the transformation matrices
        in parceled fashion for a set of subjects

        Parameters
        ----------
        Dict_subjects: Dict[str, LabeledImage]
            Dictionary containing the data of the subjects
        masker: NiftiMasker
            Masker used to transform the data
        parcellation_labels: ndarray of shape (n_vertices,)
            Array containing the parcel labels of each vertex
        scaling: bool, optional
            Compute a scaling parameter, by default False
        n_iter: int, optional
            Number of iterations, by default 10
        n_jobs: int, optional
            Number of jobs to run in parallel, by default 10

        Returns
        -------
        R_list_parcelled: list of list of rotation matrices
            for each parcel and each subject
        sc_list_parcelled: list of list of scaling parameters
            for each parcel and each subject
        """
        subject_list = list(dict_subjects.keys())
        imgs = np.stack(
            [
                masker.transform(dict_subjects[subject].img)
                for subject in subject_list
            ],
        )
        unique_labels = np.unique(parcellation_labels)
        # Compute the template and the transformation matrices for each label
        template_procrustes_partial = partial(
            self.template_procrustes,
            n_iter=n_iter,
            scaling=scaling,
            primal=None,
        )
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(template_procrustes_partial)(
                imgs[..., parcellation_labels == label]
            )
            for label in unique_labels
        )
        R_list_parcelled = [output[1] for output in outputs]
        sc_list_parcelled = [output[2] for output in outputs]
        return R_list_parcelled, sc_list_parcelled

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
        R_list, sc_list = self.compute_alignments(
            dict_subjects=self.dict_alignment,
            parcellation_labels=parcellation_labels,
            masker=self.masker,
            scaling=self.scaling,
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
            + f"_{self.clustering}_{self.n_parcels}_sc_{self.scaling}"
        )
        return dict(
            aligned_dataset=(
                self.template,
                self.dict_aligned,
                solver_name,
            ),
        )
