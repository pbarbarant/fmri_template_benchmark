from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, plotting


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
            / "outputs"
            / "figures"
            / "aligned_datasets"
            / dataset_name
            / solver_name
            / "template"
        )
        template_data = masker.transform(template.img)
        avg_contrast = np.mean(template_data[labels == contrast], axis=0)
        img = masker.inverse_transform(avg_contrast)
        fig = plotting.plot_surf_img(
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
                / "outputs"
                / "figures"
                / "aligned_datasets"
                / dataset_name
                / solver_name
                / f"{subject}"
            )
            # Plot the average contrast
            avg_contrast = np.mean(X_subject[labels == contrast], axis=0)
            img = masker.inverse_transform(avg_contrast)
            fig = plotting.plot_surf_img(
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
            fig = plotting.plot_surf_img(
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
            / "outputs"
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
                / "outputs"
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
