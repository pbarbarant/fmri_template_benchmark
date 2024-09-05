from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.stopping_criterion import SingleRunCriterion
    from joblib import Parallel, delayed
    from nilearn.experimental.surface._surface_image import SurfaceImage
    from nilearn.surface import load_surf_mesh
    from scipy import linalg
    from sklearn.cluster import AgglomerativeClustering, KMeans

    from benchmark_utils.solver_utils import (mesh_connectivity_matrix,
                                              plot_surf_img)


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Procrustes"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "n_iter": [10],
        "n_parcels": [300],
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
        Compute the template of a set of images using Procrustes analysis


        Parameters
        ----------
        imgs: (n_subjects, n_features, n_vertices) nd array
            set of images
        n_iter: int, optional
            number of iterations
        scaling: bool, optional
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
        X: (n_features, n_vertices) nd array
            template
        R_list: list of (n_features, n_features) nd array
            list of transformation matrices
        sc_list: list of int
            list of scaling parameters
        """
        # Initialize the template as the mean of the images
        n_sub, _, n_vertices = imgs.shape
        X = np.mean(imgs, axis=0)
        R_list = [np.eye(n_vertices) for _ in range(n_sub)]
        sc_list = [1 for _ in range(n_sub)]
        for _ in range(n_iter):
            for i, Y in enumerate(imgs):
                R, sc = self._scaled_procrustes(
                    X, Y, scaling=scaling, primal=primal
                )
                X = X.dot(R.T) * sc
                R_list[i] = R
                sc_list[i] = sc
        return X, R_list, sc_list

    def _compute_alignments(self, subject_list, n_jobs=10):
        """
        Compute the template and the transformation matrices in parceled fashion
        for a set of subjects

        Parameters
        ----------
        subject_list: list of str
            List of subjects

        Returns
        -------
        labels: (n_vertices,) nd array
            parcellation
        template: (n_features, n_vertices) nd array
            template
        R_list_parcelled: list of list of rotation matrices
            for each parcel and each subject
        sc_list_parcelled: list of list of scaling parameters
            for each parcel and each subject
        """
        imgs = np.stack(
            [
                self.masker.transform(self.dict_alignment[subject])
                for subject in subject_list
            ]
        )

        labels = self._compute_parcellation(
            imgs[0],
            clustering="ward",
            n_parcels=self.n_parcels,
            mesh=next(iter(self.dict_alignment.values())).mesh,
        )
        unique_labels = np.unique(labels)
        # Compute the template and the transformation matrices for each label
        outputs = Parallel(n_jobs=n_jobs)(
            delayed(self._template_procrustes)(imgs[..., labels == label])
            for label in unique_labels
        )
        template = [output[0] for output in outputs]
        R_list_parcelled = [output[1] for output in outputs]
        sc_list_parcelled = [output[2] for output in outputs]

        return labels, template, R_list_parcelled, sc_list_parcelled

    def _compute_parcellation(
        self, data, mesh=None, clustering="kmeans", n_parcels=10
    ):
        """Compute a parcellation of the data using clustering

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_vertices)
            Data used to compute the parcellation
        clustering : str, optional
            Clustering strategy, by default "kmeans"
        n_parcels : int, optional
            Number of parcels, by default 10

        Returns
        -------
        labels
            ndarray of shape (n_vertices,) containing the parcel labels of each
            vertex
        """
        if n_parcels % 2 != 0:
            raise ValueError("The number of parcels must be even.")
        n_parcels_hemi = n_parcels // 2

        # Reshape the data to 2D (n_vertices, n_samples)
        n_vertices = data.shape[1]
        data_2d = data.reshape(n_vertices, -1)

        # Choose the clustering method
        if clustering.lower() == "kmeans":
            clusterer = KMeans(n_clusters=n_parcels_hemi, random_state=42)
            # Fit the clustering
            labels = clusterer.fit_predict(data_2d).reshape(data.shape[1])

            return labels

        elif clustering.lower() == "ward":
            # Compute the connectivity matrix for each hemisphere
            coordinates_left, faces_left = load_surf_mesh(mesh.parts["left"])
            coordinates_right, faces_right = load_surf_mesh(
                mesh.parts["right"]
            )
            connectivity_left = mesh_connectivity_matrix(
                coordinates_left, faces_left
            )
            connectivity_right = mesh_connectivity_matrix(
                coordinates_right, faces_right
            )
            clusterer_left = AgglomerativeClustering(
                n_clusters=n_parcels_hemi,
                connectivity=connectivity_left,
                linkage="ward",
            )
            clusterer_right = AgglomerativeClustering(
                n_clusters=n_parcels_hemi,
                connectivity=connectivity_right,
                linkage="ward",
            )
            labels_left = clusterer_left.fit_predict(
                np.ones(n_vertices // 2).reshape(-1, 1)
            )
            labels_right = clusterer_right.fit_predict(
                np.ones(n_vertices // 2).reshape(-1, 1)
            )

            return np.concatenate([labels_left, labels_right])
        else:
            raise ValueError(
                "Unsupported clustering method. Choose 'kmeans' or 'ward'."
            )

    def _project(self, X, labels, R_list, sc_list, n_sub):
        """Project the data of a subject onto the template

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_vertices)
            Data to project
        labels : ndarray of shape (n_vertices,)
            Array containing the parcel labels of each vertex
        R_list : List
            List of list of rotation matrices for each parcel and each subject
        sc_list : List
            List of list of scaling parameters for each parcel and each subject
        n_sub : int
            Subject index

        Returns
        -------
        X_transform : ndarray of shape (n_samples, n_vertices)
            Transformed data
        """
        # Decompose X into parcels
        X_transform = np.zeros_like(X)
        unique_labels = np.unique(labels)

        for i in range(len(unique_labels)):
            label = unique_labels[i]
            X_reduced = X[:, labels == label]
            X_transform[:, labels == label] = (
                X_reduced.dot(R_list[i][n_sub].T) * sc_list[i][n_sub]
            )
        return X_transform

    def run(self, n_iter):
        # This is the function that is called to evaluate the solver.
        # It runs the algorithm for a given a number of iterations `n_iter`.
        # You can also use a `tolerance` or a `callback`, as described in
        # https://benchopt.github.io/performance_curves.html

        labels, template, R_list, sc_list = self._compute_alignments(
            list(self.dict_alignment.keys()), n_jobs=10
        )

        img_labels = SurfaceImage(
            mesh=next(iter(self.dict_alignment.values())).mesh,
            data={
                "left": labels[: len(labels) // 2],
                "right": labels[len(labels) // 2 :],
            },
        )

        fig = plot_surf_img(img_labels, cmap="tab20")
        fig.suptitle(f"Parcellation of the brain in {self.n_parcels} parcels")
        fig.savefig(f"figures/parcellation_{self.n_parcels}.pdf")

        self.X = np.concatenate(
            [
                self._project(
                    self.masker.transform(self.dict_decoding[subject]),
                    labels,
                    R_list,
                    sc_list,
                    i,
                )
                for i, subject in enumerate(self.dict_decoding.keys())
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
