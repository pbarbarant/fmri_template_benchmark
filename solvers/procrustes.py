from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from benchopt.stopping_criterion import SingleRunCriterion
    from scipy import linalg
    from sklearn.cluster import KMeans, FeatureAgglomeration
    from joblib import Parallel, delayed

    import numpy as np


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Procrustes"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {}

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

    def _scaled_procrustes(self, X, Y, scaling=False, primal=None):
        """
        Compute a mixing matrix R and a scaling sc such that Frobenius norm
        ||sc RX - Y||^2 is minimized and R is an orthogonal matrix

        Parameters
        ----------
        X: (n_samples, n_features) nd array
            source data
        Y: (n_samples, n_features) nd array
            target data
        scaling: bool
            If scaling is true, computes a floating scaling parameter sc such that:
            ||sc * RX - Y||^2 is minimized and
            - R is an orthogonal matrix
            - sc is a scalar
            If scaling is false sc is set to 1
        primal: bool or None, optional,
            Whether the SVD is done on the YX^T (primal) or Y^TX (dual)
            if None primal is used iff n_features <= n_timeframes

        Returns
        ----------
        R: (n_features, n_features) nd array
            transformation matrix
        sc: int
            scaling parameter
        """
        X = X.astype(np.float64, copy=False)
        Y = Y.astype(np.float64, copy=False)
        if np.linalg.norm(X) == 0 or np.linalg.norm(Y) == 0:
            return np.eye(X.shape[1]), 1
        if primal is None:
            primal = X.shape[0] >= X.shape[1]
        if primal:
            A = Y.T.dot(X)
            if A.shape[0] == A.shape[1]:
                A += +1.0e-18 * np.eye(A.shape[0])
            U, s, V = linalg.svd(A, full_matrices=0)
            R = U.dot(V)
        else:  # "dual" mode
            Uy, sy, Vy = linalg.svd(Y, full_matrices=0)
            Ux, sx, Vx = linalg.svd(X, full_matrices=0)
            A = np.diag(sy).dot(Uy.T).dot(Ux).dot(np.diag(sx))
            U, s, V = linalg.svd(A)
            R = Vy.T.dot(U).dot(V).dot(Vx)

        if scaling:
            sc = s.sum() / (np.linalg.norm(X) ** 2)
        else:
            sc = 1
        return R.T, sc

    def _template_procrustes(
        self, imgs, n_iter=10, scaling=False, primal=None
    ):
        """
        Compute a template X and a list of transformation matrices R.

        Parameters
        ----------
        imgs: list of (n_samples, n_features) nd array
            list of data to align
        scaling: bool
            If scaling is true, computes a floating scaling parameter sc such that:
            ||sc * RX - Y||^2 is minimized and
            - R is an orthogonal matrix
            - sc is a scalar
            If scaling is false sc is set to 1
        primal: bool or None, optional,
            Whether the SVD is done on the YX^T (primal) or Y^TX (dual)
            if None primal is used iff n_features <= n_timeframes

        Returns
        ----------
        X: (n_samples, n_features) nd array
            template data
        R: list of (n_features, n_features) nd array
            list of transformation matrix
        sc: int
            scaling parameter
        """
        # Initialize the template as the mean of the images
        X = np.mean(imgs)
        R_list = []
        for i in range(n_iter):
            for Y in imgs:
                R, sc = self._scaled_procrustes(
                    X, Y, scaling=scaling, primal=primal
                )
                X = X.dot(R.T) * sc
                if i == n_iter - 1:
                    R_list.append(R)
        return X, R_list, sc

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        self.X = self.masker.inverse_transform(
            np.concatenate(
                [
                    self.masker.transform(self.dict_decoding[subject])
                    for subject in self.dict_decoding.keys()
                ]
            )
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
