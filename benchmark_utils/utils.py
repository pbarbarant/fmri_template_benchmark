from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from nilearn import maskers, masking, surface
from nibabel.nifti1 import Nifti1Image
from scipy.sparse import coo_matrix


@dataclass
class LabeledImage:
    img: Nifti1Image
    y: np.ndarray


@dataclass
class Fold:
    name: str
    dict_alignment: dict
    dict_decoding: dict


@dataclass
class DecodingFold:
    name: str
    template: LabeledImage
    dict_aligned: dict


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
