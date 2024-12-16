from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from nilearn import maskers, masking, surface
from nibabel.nifti1 import Nifti1Image
from nilearn.experimental.surface._datasets import load_fsaverage
from nilearn.experimental.surface._surface_image import SurfaceImage
from scipy.sparse import coo_matrix


@dataclass
class LabeledImage:
    img: Nifti1Image
    y: np.ndarray


def mesh_connectivity_matrix(coordinates, triangles):
    """
    Compute sparse matrix representing edges of a given mesh.

    Parameters
    ----------
    coordinates: ndarray of size (n, k)
    triangles: ndarray of size (e, 3)

    Returns
    -------
    connectivity: sparse coo matrix of size (n, n)
    """
    n_vertices = coordinates.shape[0]
    edges = np.hstack(
        (
            np.vstack((triangles[:, 0], triangles[:, 1])),
            np.vstack((triangles[:, 0], triangles[:, 2])),
            np.vstack((triangles[:, 1], triangles[:, 0])),
            np.vstack((triangles[:, 1], triangles[:, 2])),
            np.vstack((triangles[:, 2], triangles[:, 0])),
            np.vstack((triangles[:, 2], triangles[:, 1])),
        )
    )
    weights = np.ones(edges.shape[1])

    # Divide data by 2 since all edges i -> j are counted twice
    # because they all belong to exactly two triangles on the mesh
    connectivity = (
        coo_matrix((weights, edges), (n_vertices, n_vertices)).tocsr() / 2
    )

    # Force symmetry
    connectivity = (connectivity + connectivity.T) / 2

    return connectivity


def load_dataset_vol(subject, data_path, mask):
    data_alignment = mask.inverse_transform(
        joblib.load(data_path / "alignment" / f"{subject}.pkl")
    )
    data_decoding = mask.inverse_transform(
        joblib.load(data_path / "decoding" / f"{subject}.pkl")
    )
    labels_decoding = pd.read_csv(
        data_path / "decoding" / f"{subject}_labels.csv",
        header=None,
    ).values.ravel()

    return data_alignment, data_decoding, labels_decoding


def load_mask(data_path, memory):
    masker_path = data_path / "masks" / "mask.nii.gz"
    connected_mask = masking.compute_background_mask(
        masker_path, connected=True
    )
    mask = maskers.NiftiMasker(connected_mask, memory=memory).fit()
    return mask


def project_on_surf(data, mesh_name="fsaverage5"):
    mesh = load_fsaverage(mesh_name)["pial"]
    left_data = surface.vol_to_surf(data, mesh.parts["left"]).T
    right_data = surface.vol_to_surf(data, mesh.parts["right"]).T
    left_data_sanitized = np.nan_to_num(left_data)
    right_data_sanitized = np.nan_to_num(right_data)
    return SurfaceImage(
        mesh=mesh,
        data={
            "left": left_data_sanitized,
            "right": right_data_sanitized,
        },
    )


def load_dataset_surf(subject, data_path, mesh_name):
    data_alignment_left = joblib.load(
        data_path / "alignment" / f"{subject}_left.pkl"
    ).T
    data_alignment_right = joblib.load(
        data_path / "alignment" / f"{subject}_right.pkl"
    ).T
    data_alignment_img = SurfaceImage(
        mesh=load_fsaverage(mesh_name)["pial"],
        data={
            "left": data_alignment_left,
            "right": data_alignment_right,
        },
    )

    data_decoding_left = joblib.load(
        data_path / "decoding" / f"{subject}_left.pkl"
    ).T
    data_decoding_right = joblib.load(
        data_path / "decoding" / f"{subject}_right.pkl"
    ).T
    data_decoding_img = SurfaceImage(
        mesh=load_fsaverage(mesh_name)["pial"],
        data={
            "left": data_decoding_left,
            "right": data_decoding_right,
        },
    )

    labels_decoding = pd.read_csv(
        data_path / "labels" / f"{subject}.csv",
        header=None,
    ).values.ravel()

    alignment_labeled = LabeledImage(
        img=data_alignment_img,
        labels=None,
    )

    decoding_labeled = LabeledImage(
        img=data_decoding_img,
        labels=labels_decoding,
    )
    return alignment_labeled, decoding_labeled
