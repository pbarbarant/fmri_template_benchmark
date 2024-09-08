import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from scipy.sparse import coo_matrix


def plot_surf_img(
    img,
    **kwargs,
):
    mesh = img.mesh
    parts = list(img.data.parts.keys())
    fig, axes = plt.subplots(
        1,
        len(parts),
        subplot_kw={"projection": "3d"},
        figsize=(4 * len(parts), 4),
    )
    for ax, mesh_part in zip(axes, parts):
        plotting.plot_surf(
            mesh.parts[mesh_part],
            img.data.parts[mesh_part],
            hemi=mesh_part,
            axes=ax,
            title=mesh_part,
            **kwargs,
        )
    assert isinstance(fig, plt.Figure)
    return fig


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
