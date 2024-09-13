import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from scipy.sparse import coo_matrix
import joblib
import pandas as pd
from pathlib import Path

from nilearn import maskers, masking, surface, datasets
from nilearn.experimental.surface._datasets import load_fsaverage
from nilearn.experimental.surface._surface_image import SurfaceImage


def plot_surf_img(
    img,
    bg_map=None,
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
            bg_map=bg_map[mesh_part],
            hemi=mesh_part,
            axes=ax,
            title=mesh_part,
            bg_on_data=True,
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


def load_dataset(subject, data_path, mask):
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


def project_on_surf(data, mesh_name="fsaverage3"):
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


def load_dataset_surf(subject, data_path, mask, mesh_name):
    data_alignment, data_decoding, labels_decoding = load_dataset(
        subject, data_path, mask
    )
    data_alignment_surf = project_on_surf(data_alignment, mesh_name)
    data_decoding_surf = project_on_surf(data_decoding, mesh_name)
    return data_alignment_surf, data_decoding_surf, labels_decoding


def plot_aligned_dataset(
    X,
    y,
    groups,
    labels,
    fitted_estimators,
    solver_name,
    dataset_name,
    subjects_list,
    masker,
):
    fs5 = datasets.fetch_surf_fsaverage("fsaverage5")
    bg_map = {"left": fs5["sulc_left"], "right": fs5["sulc_right"]}
    subjects = np.unique(groups)
    for i, subject in enumerate(subjects):
        X_subject = X[groups == subject]
        y_subject = y[groups == subject]
        estimator = fitted_estimators[i]
        # Get the coefficients of the SVC
        coefs = estimator[-1].coef_
        for contrast_index, contrast in enumerate(labels):
            print(
                f"Plotting subject {subjects_list[subject]} - contrast {contrast}"
            )
            output_dir = (
                Path(__file__).parent.parent
                / "figures"
                / "aligned_datasets"
                / dataset_name
                / solver_name
                / f"{subjects_list[subject]}"
            )
            # Plot the average contrast
            avg_contrast = np.mean(X_subject[y_subject == contrast], axis=0)
            img = masker.inverse_transform(avg_contrast)
            fig = plot_surf_img(
                img,
                bg_map=bg_map,
                colorbar=True,
                cmap="coolwarm",
            )
            fig.suptitle(
                f"Subject {subjects_list[subject]} - {solver_name} - contrast {contrast}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"{contrast}.pdf")
            plt.close(fig)

            # Plot the weights
            # Get the contrast index
            img_coefs = masker.inverse_transform(coefs[contrast_index])
            fig = plot_surf_img(
                img_coefs,
                bg_map=bg_map,
                colorbar=True,
                cmap="hot",
                # Keep only the significant weights
                threshold=1e-3,
            )
            fig.suptitle(
                f"Subject {subjects_list[subject]} - {solver_name} - contrast {contrast}"
            )
            fig.savefig(output_dir / f"coefs_{contrast}.pdf")
            plt.close(fig)
