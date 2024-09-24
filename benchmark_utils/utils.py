from pathlib import Path
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import datasets, maskers, masking, plotting, surface
from nilearn.experimental.surface._datasets import load_fsaverage
from nilearn.experimental.surface._surface_image import SurfaceImage
from scipy.sparse import coo_matrix


@dataclass
class LabeledImage:
    labels: np.ndarray
    img: SurfaceImage


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
            bg_map=bg_map[mesh_part] if bg_map is not None else None,
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
    mesh = load_fsaverage(mesh_name)["inflated"]
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
        mesh=load_fsaverage(mesh_name)["inflated"],
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
        mesh=load_fsaverage(mesh_name)["inflated"],
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


def plot_template(
    template,
    masker,
    solver_name,
    dataset_name,
    mesh_name,
):
    fsaverage = datasets.fetch_surf_fsaverage(mesh_name)
    bg_map = {"left": fsaverage["sulc_left"], "right": fsaverage["sulc_right"]}
    labels = template.labels
    for contrast in np.unique(labels):
        print(f"Plotting template - contrast {contrast}")
        output_dir = (
            Path(__file__).parent.parent
            / "figures"
            / "aligned_datasets"
            / dataset_name
            / solver_name
            / "template"
        )
        template_data = masker.transform(template.img)
        avg_contrast = np.mean(template_data[labels == contrast], axis=0)
        img = masker.inverse_transform(avg_contrast)
        fig = plot_surf_img(
            img,
            bg_map=bg_map,
            colorbar=True,
            cmap="coolwarm",
        )
        fig.suptitle(f"Template - {solver_name} - contrast {contrast}")
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{contrast}.pdf")
        plt.close(fig)


def plot_aligned_dataset(
    dict_aligned,
    dict_estimators,
    solver_name,
    dataset_name,
    subjects_list,
    masker,
    mesh_name,
):
    fsaverage = datasets.fetch_surf_fsaverage(mesh_name)
    bg_map = {"left": fsaverage["sulc_left"], "right": fsaverage["sulc_right"]}
    subjects_list = list(dict_aligned.keys())
    for subject in subjects_list:
        # Get the labels
        labels = dict_aligned[subject].labels
        # Get the data
        X_subject = masker.transform(dict_aligned[subject].img)
        # Get the estimator
        estimator = dict_estimators[subject]
        # Get the coefficients and the associated classes for the SVC
        coefs = estimator[-1].coef_
        class_labels = estimator[-1].classes_
        for contrast_index, contrast in enumerate(class_labels):
            print(f"Plotting subject {subject} - contrast {contrast}")
            output_dir = (
                Path(__file__).parent.parent
                / "figures"
                / "aligned_datasets"
                / dataset_name
                / solver_name
                / f"{subject}"
            )
            # Plot the average contrast
            avg_contrast = np.mean(X_subject[labels == contrast], axis=0)
            img = masker.inverse_transform(avg_contrast)
            fig = plot_surf_img(
                img,
                bg_map=bg_map,
                colorbar=True,
                cmap="coolwarm",
            )
            fig.suptitle(
                f"Subject {subject} - {solver_name} - contrast {contrast}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / f"{contrast}.pdf")
            plt.close(fig)

            # Plot the weights
            # Get the contrast index
            if len(coefs) == 1:  # binary classification
                contrast_index = 0
                coefs *= -1
            img_coefs = masker.inverse_transform(coefs[contrast_index])
            fig = plot_surf_img(
                img_coefs,
                bg_map=bg_map,
                colorbar=True,
                cmap="hot",
                # Keep only the significant weights
                threshold=1e-6,
            )
            fig.suptitle(
                f"Subject {subject} - {solver_name} - contrast {contrast}"
            )
            fig.savefig(output_dir / f"coefs_{contrast}.pdf")
            plt.close(fig)


def generate_template_gii(
    template,
    masker,
    solver_name,
    dataset_name,
):
    labels = template.labels
    for contrast in np.unique(labels):
        print(f"Saving template - contrast {contrast}")
        output_dir = (
            Path(__file__).parent.parent
            / "figures"
            / "aligned_datasets"
            / dataset_name
            / solver_name
            / "template"
        )
        template_data = masker.transform(template.img)
        avg_contrast = np.mean(template_data[labels == contrast], axis=0)
        img = masker.inverse_transform(avg_contrast)

        img_to_gifti(
            img,
            img_name=f"{contrast}",
            output_dir=output_dir,
        )


def generate_aligned_dataset_gii(
    dict_aligned,
    dict_estimators,
    solver_name,
    dataset_name,
    subjects_list,
    masker,
):
    subjects_list = list(dict_aligned.keys())
    for subject in subjects_list:
        # Get the labels
        labels = dict_aligned[subject].labels
        # Get the data
        X_subject = masker.transform(dict_aligned[subject].img)
        # Get the estimator
        estimator = dict_estimators[subject]
        # Get the coefficients and the associated classes for the SVC
        coefs = estimator[-1].coef_
        class_labels = estimator[-1].classes_
        for contrast_index, contrast in enumerate(class_labels):
            print(f"Plotting subject {subject} - contrast {contrast}")
            output_dir = (
                Path(__file__).parent.parent
                / "figures"
                / "aligned_datasets"
                / dataset_name
                / solver_name
                / f"{subject}"
            )
            # Save the average contrast
            avg_contrast = np.mean(X_subject[labels == contrast], axis=0)
            img = masker.inverse_transform(avg_contrast)
            img_to_gifti(
                img,
                img_name=f"{contrast}",
                output_dir=output_dir,
            )

            # Save the weights
            # Get the contrast index
            if len(coefs) == 1:  # binary classification
                contrast_index = 0
                coefs *= -1
            img_coefs = masker.inverse_transform(coefs[contrast_index])
            img_to_gifti(
                img_coefs,
                img_name=f"coefs_{contrast}",
                output_dir=output_dir,
            )


def img_to_gifti(
    img,
    img_name,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for hemi in ["left", "right"]:
        gifti = nib.gifti.GiftiImage()
        data_array = nib.gifti.GiftiDataArray(
            img.data.parts[hemi].astype(np.float32)
        )
        gifti.add_gifti_data_array(data_array)
        nib.save(gifti, output_dir / f"{img_name}_{hemi}.gii")
