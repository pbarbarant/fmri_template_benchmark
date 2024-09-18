from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.experimental.surface._surface_image import SurfaceImage
from nilearn.surface import load_surf_mesh
from scipy import linalg
from sklearn.cluster import AgglomerativeClustering, KMeans

from benchmark_utils.utils import (
    mesh_connectivity_matrix,
    plot_surf_img,
    LabeledImage,
)


def compute_parcellation(data, mesh=None, clustering="kmeans", n_parcels=10):
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
    # Reshape the data to 2D (n_vertices, n_samples)
    n_vertices = data.shape[1]
    data_2d = data.reshape(n_vertices, -1)

    # Choose the clustering method
    if clustering.lower() == "kmeans":
        clusterer = KMeans(n_clusters=n_parcels, random_state=42)
        # Fit the clustering
        labels = clusterer.fit_predict(data_2d).reshape(data.shape[1])

        return labels

    elif clustering.lower() == "ward":
        # Compute the connectivity matrix for each hemisphere
        coordinates_left, faces_left = load_surf_mesh(mesh.parts["left"])
        coordinates_right, faces_right = load_surf_mesh(mesh.parts["right"])
        connectivity_left = mesh_connectivity_matrix(
            coordinates_left, faces_left
        )
        connectivity_right = mesh_connectivity_matrix(
            coordinates_right, faces_right
        )
        clusterer_left = AgglomerativeClustering(
            n_clusters=n_parcels,
            connectivity=connectivity_left,
            linkage="ward",
        )
        clusterer_right = AgglomerativeClustering(
            n_clusters=n_parcels,
            connectivity=connectivity_right,
            linkage="ward",
        )
        X_left = np.mean(data_2d[: n_vertices // 2, :], axis=1).reshape(-1, 1)
        X_right = np.mean(data_2d[n_vertices // 2 :, :], axis=1).reshape(-1, 1)
        labels_left = clusterer_left.fit_predict(X_left)
        # Offset the labels of the right hemisphere
        labels_right = clusterer_right.fit_predict(X_right) + n_parcels

        return np.concatenate([labels_left, labels_right])
    elif clustering.lower() == "destrieux":
        if data_2d.shape[0] != 20484:
            raise ValueError(
                (
                    "The Destrieux parcellation is only available "
                    "for the fsaverage5 mesh."
                )
            )
        destrieux = datasets.fetch_atlas_surf_destrieux()
        labels_left = destrieux["map_left"]
        labels_right = destrieux["map_right"]
        return np.concatenate([labels_left, labels_right])
    else:
        raise ValueError(
            (
                "Unsupported clustering method. Choose 'kmeans',"
                "'ward' or 'destrieux'."
            )
        )


def scaled_procrustes(X, Y, scaling=False, primal=None):
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


def project(
    img,
    masker,
    labels,
    R_list,
    sc_list,
    n_sub,
    classes=None,
):
    """Project the data of a subject onto the template

    Parameters
    ----------
    img : SurfaceImage
        Data to project
    masker : NiftiMasker
        Masker used to transform the data
    labels : ndarray of shape (n_vertices,)
        Array containing the parcel labels of each vertex
    R_list : List
        List of list of rotation matrices for each parcel and each subject
    sc_list : List
        List of list of scaling parameters for each parcel and each subject
    n_sub : int
        Subject index
    classes : ndarray of shape (n_samples,)
        Array containing the class of each sample

    Returns
    -------
    projected_data : LabeledImage
        Projected data
    """
    X = masker.transform(img)
    # Decompose X into parcels
    X_transform = np.zeros_like(X)
    unique_labels = np.unique(labels)

    for i in range(len(unique_labels)):
        label = unique_labels[i]
        X_reduced = X[:, labels == label]
        X_transform[:, labels == label] = (
            X_reduced.dot(R_list[i][n_sub].T) * sc_list[i][n_sub]
        )

    projected_img = masker.inverse_transform(X_transform)
    projected_data = LabeledImage(
        img=projected_img,
        labels=classes,
    )

    return projected_data


def compute_template(
    dict_subjects,
    masker,
    parcellation_labels,
    R_list,
    sc_list,
):
    """Compute the template of a set of subjects from given
    parcellation labels, rotation matrices and scaling parameters

    Parameters
    ----------
    dict_subjects : Dict[str, LabeledImage]
        Dictionary containing the data of the subjects
    masker : NiftiMasker
        Masker used to transform the data
    parcellation_labels : ndarray of shape (n_vertices,)
        Array containing the parcel labels of each vertex
    R_list : List
        List of list of rotation matrices for each parcel
        and each subject
    sc_list : List
        List of list of scaling parameters for each parcel
        and each subject

    Returns
    -------
    template : LabeledImage
        Functional template
    """
    subject_list = list(dict_subjects.keys())
    template_data = np.zeros_like(
        masker.transform(dict_subjects[subject_list[0]].img)
    )
    for i, subject in enumerate(subject_list):
        projected_img = project(
            dict_subjects[subject].img,
            masker=masker,
            labels=parcellation_labels,
            R_list=R_list,
            sc_list=sc_list,
            n_sub=i,
            classes=None,
        ).img
        projected_data = masker.transform(projected_img)
        template_data += projected_data / len(subject_list)

    template_img = masker.inverse_transform(template_data)
    template_labels = dict_subjects[subject_list[0]].labels
    template = LabeledImage(img=template_img, labels=template_labels)
    return template


def plot_parcellation(mesh, labels, n_parcels, clustering):
    """Plot the parcellation of the brain

    Parameters
    ----------
    mesh : str
        Path to the mesh
    labels : ndarray of shape (n_vertices,)
        Array containing the parcel labels of each vertex
    n_parcels : int
        Number of parcels
    clustering : str
        Clustering strategy
    """
    img_labels = SurfaceImage(
        mesh=mesh,
        data={
            "left": labels[: len(labels) // 2],
            "right": labels[len(labels) // 2 :],
        },
    )

    fig = plot_surf_img(img_labels, cmap="tab20")
    fig.suptitle(f"Parcellation of the brain in {n_parcels} parcels")
    output_dir = (
        Path(__file__).parent.parent
        / "figures"
        / "parcellations"
        / f"{clustering}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{n_parcels}.pdf")
    plt.close(fig)
