import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import h5py
import numpy as np
import pandas as pd
from fmralign._utils import _intersect_clustering_mask
from ibc_public import utils_data
from nibabel.nifti1 import Nifti1Image
from nilearn.datasets import (
    load_mni152_gm_mask,
    fetch_atlas_schaefer_2018,
    fetch_atlas_surf_destrieux,
    load_fsaverage,
)
from nilearn.image import (
    load_img,
    math_img,
    resample_to_img,
    concat_imgs,
)
from nilearn.maskers import MultiNiftiMasker, SurfaceMasker
from nilearn.masking import apply_mask_fmri, unmask
from nilearn.surface import PolyMesh, SurfaceImage
from tqdm import tqdm
import nibabel as nib
from benchmark_utils.conf import (
    IBC_PATH,
    IBC_SURF_PATH,
    MEMORY,
    N_JOBS,
    NEUROMOD_PATH,
)


@dataclass
class LabeledImage:
    img: Nifti1Image | SurfaceImage
    y: np.ndarray


@dataclass
class Dataset:
    name: str
    subjects: List[str]
    n_subjects: int
    labels: np.ndarray
    dict_alignment: Dict[str, np.ndarray]
    dict_decoding: Dict[str, np.ndarray]
    dict_y: Dict[str, np.ndarray]
    test_sub: str
    external_template: bool
    task_name: str
    is_faulty: bool = False
    output_dir: Optional[Path] = None
    time: Optional[float] = None
    dict_aligned: Optional[Dict[str, np.ndarray]] = None
    template: Optional[np.ndarray] = None
    solver: Optional[str] = None


def load_atlas(resolution=3, n_rois=100):
    atlas = fetch_atlas_schaefer_2018(n_rois=n_rois)
    atlas = resample_to_img(
        atlas.maps,
        load_mni152_gm_mask(resolution=resolution),
        interpolation="nearest",
    )
    return atlas


def get_mask_img(resolution=3, n_rois=100):
    atlas = load_atlas(resolution=resolution, n_rois=n_rois)
    gm_mask = load_mni152_gm_mask(resolution=resolution)
    mask_img = math_img("img1*img2 > 0", img1=atlas, img2=gm_mask)
    return mask_img


def fit_masker(resolution=3, n_rois=100, n_jobs=1):
    mask_img = get_mask_img(resolution=resolution, n_rois=n_rois)
    masker = MultiNiftiMasker(
        mask_img=mask_img,
        standardize=True,
        reports=False,
        n_jobs=n_jobs,
        verbose=11,
    ).fit()
    return masker


def sample_dataset(
    name: str,
    subjects: List[str],
    test_sub: str,
    external_template: bool = False,
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()

    for subject in tqdm(subjects, desc="Sampling dataset"):
        # Create a random alignment and decoding data for each subject
        dict_alignment[subject] = np.random.rand(100, 30)
        dict_decoding[subject] = np.random.rand(100, 30)
        dict_y[subject] = np.random.randint(0, 2, size=100)

    return Dataset(
        name=name,
        subjects=subjects,
        n_subjects=len(subjects),
        labels=np.ones(30, dtype=int),
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        test_sub=test_sub,
        external_template=external_template,
        task_name="simulated_task",
    )


def fetch_ibc_vol(
    test_sub: str,
    name: str = "IBC",
    subjects: List[str] = None,
    task: str = None,
    n_parcels: int = 400,
    external_template: bool = False,
) -> Dataset:
    df = utils_data.make_vol_db(
        derivatives=IBC_PATH,
        subject_list=subjects,
        task_list=[task],
        acquisition="all",
    )
    # Drop rows with with ffx acquisitions
    df = df[~df.path.str.contains("ffx")]
    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()

    masker = fit_masker(resolution=3, n_rois=n_parcels, n_jobs=N_JOBS)
    labels = apply_mask_fmri(
        load_atlas(resolution=3, n_rois=n_parcels), masker.mask_img_
    ).astype(int)
    missing_subjects = []
    for subject in tqdm(subjects, desc="Processing IBC data"):
        try:
            df_sub = df[(df.subject == subject)]
            # For each contrast, keep randomly one path
            alignment_df = df_sub.groupby(["contrast"]).apply(
                lambda x: x.sample(1, random_state=0)
            )
            # Put the rest in decoding_df
            decoding_df = df_sub[~df_sub.index.isin(alignment_df.index)]
            dict_alignment[subject] = np.vstack(
                masker.transform(alignment_df.path.to_list())
            )
            dict_decoding[subject] = np.vstack(
                masker.transform(decoding_df.path.to_list())
            )
            dict_y[subject] = decoding_df.contrast.to_numpy()
        except ValueError as e:
            print(f"Error processing subject {subject}: {e}")
            # Pop the subject from the dictionaries if it fails
            dict_alignment.pop(subject, None)
            dict_decoding.pop(subject, None)
            dict_y.pop(subject, None)
            # Add the subject to the missing subjects list
            missing_subjects.append(subject)
            continue

    valid_subjects = [sub for sub in subjects if sub not in missing_subjects]

    is_faulty = False
    if test_sub not in valid_subjects:
        print(f"Test subject {test_sub} data not found. Marking dataset as faulty.")
        is_faulty = True

    return Dataset(
        name=name,
        subjects=valid_subjects,
        n_subjects=len(valid_subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        test_sub=test_sub,
        external_template=external_template,
        task_name=task,
        is_faulty=is_faulty,
    )


def load_surface_img(
    paths: List[str], mesh: PolyMesh
) -> (
    SurfaceImage
):  # -> Any | SurfaceImage:# -> Any | SurfaceImage:# -> Any | SurfaceImage:
    # Remove lh.gii and rh.gii extension
    paths = [path[:-7] for path in paths]
    # Remove duplicates
    paths = list(dict.fromkeys(paths))
    surf_imgs = []
    for path in paths:
        surf_img = SurfaceImage(
            mesh=mesh,
            data={
                "left": path + "_lh.gii",
                "right": path + "_rh.gii",
            },
        )
        surf_imgs.append(surf_img)
    return concat_imgs(surf_imgs)


def fetch_ibc_surf(
    test_sub: str,
    name: str = "IBC",
    subjects: List[str] = None,
    task: str = None,
    external_template: bool = False,
) -> Dataset:
    df = utils_data.make_surf_db(
        derivatives=IBC_SURF_PATH,
        subject_list=subjects,
        task_list=[task],
        acquisition="all",
    )
    mesh = load_fsaverage("fsaverage5")["pial"]
    atlas = fetch_atlas_surf_destrieux()
    labels = np.hstack(
        [atlas["map_left"], atlas["map_right"] + atlas["map_left"].max()]
    ).astype(int)
    labels_img = SurfaceImage(
        mesh=mesh,
        data={
            "left": atlas["map_left"],
            "right": atlas["map_right"],
        },
    )
    masker = SurfaceMasker(
        mask_img=labels_img,
        standardize=True,
        reports=False,
        verbose=11,
    ).fit()

    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()
    missing_subjects = []
    for subject in tqdm(subjects, desc="Processing IBC data"):
        try:
            alignment_df = df[
                (df.subject == subject) & (df.path.str.contains("_dir-ap"))
            ]
            decoding_df = df[
                (df.subject == subject) & (df.path.str.contains("_dir-pa"))
            ]
            dict_alignment[subject] = masker.transform(
                load_surface_img(alignment_df.path.to_list(), mesh)
            )
            dict_decoding[subject] = masker.transform(
                load_surface_img(decoding_df.path.to_list(), mesh),
            )
            dict_y[subject] = decoding_df[
                decoding_df.side == "lh"
            ].contrast.to_numpy()
        except TypeError as e:
            print(f"Error processing subject {subject}: {e}")
            # Pop the subject from the dictionaries if it fails
            dict_alignment.pop(subject, None)
            dict_decoding.pop(subject, None)
            dict_y.pop(subject, None)
            # Add the subject to the missing subjects list
            missing_subjects.append(subject)
            continue

    valid_subjects = [sub for sub in subjects if sub not in missing_subjects]

    is_faulty = False
    if test_sub not in valid_subjects:
        print(f"Test subject {test_sub} data not found. Marking dataset as faulty.")
        is_faulty = True

    return Dataset(
        name=name,
        subjects=valid_subjects,
        n_subjects=len(valid_subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        test_sub=test_sub,
        external_template=external_template,
        task_name=task,
        is_faulty=is_faulty,
    )


def load_neuromod_labels(
    data_path: Path,
    subject: str,
):
    image_labels = np.load(
        f"{str(data_path)}/things.glmsingle/{subject}/descriptive/"
        f"{subject}_task-things_desc-perTrial_labels.npy",
        allow_pickle=True,
    )
    y = image_labels.copy()
    for i in range(image_labels.shape[0]):
        y[i] = str(image_labels[i])[:-4]

    return y


def load_neuromod_mask(data_path: Path, subject: str):
    path = (
        data_path
        / (
            f"things.glmsingle/{subject}/glmsingle/output/"
            f"{subject}_task-things_space-T1w_model-fitHrfGLMdenoiseRR"
            "_stat-trialBetas_desc-zscore_statseries.h5"
        )
    )
    h5file = h5py.File(path, "r")
    return nib.nifti1.Nifti1Image(
        np.array(h5file["mask_array"]), affine=np.array(h5file["mask_affine"])
    )


def load_neuromod_data(
    data_path: Path,
    subject: str,
):
    path = data_path / (
        "things.glmsingle/"
        f"{subject}/descriptive/"
        f"{subject}_task-things_space-T1w_stat-betas_desc-perTrial_"
        "statseries.npy"
    )
    # Memory-map the array to avoid loading the full file into memory
    return np.load(path, mmap_mode="r").astype(np.float32)


def fetch_neuromod(
    test_sub: str,
    name: str = "Neuromod",
    subjects: List[str] = None,
    task: str = "THINGS",
    n_parcels: int = 400,
    external_template: bool = False,
):
    data_path = Path(NEUROMOD_PATH)
    alignment_labels = ["cat", "dog"]
    decoding_labels = ["cat", "dog"]
    n_contrasts = 10

    masker = fit_masker(resolution=3, n_rois=n_parcels, n_jobs=N_JOBS)
    labels = apply_mask_fmri(
        load_atlas(resolution=3, n_rois=n_parcels), masker.mask_img_
    ).astype(int)

    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()
    for subject in tqdm(subjects, desc="Processing Neuromod data"):
        individual_mask = load_neuromod_mask(
            data_path=data_path,
            subject=subject,
        )
        data = load_neuromod_data(
            data_path=data_path,
            subject=subject,
        )
        img_labels = load_neuromod_labels(
            data_path=data_path,
            subject=subject,
        )
        alignment_indices = np.hstack(
            [
                np.where(img_labels == lbl)[0][:n_contrasts]
                for lbl in alignment_labels
            ]
        )
        decoding_indices = np.hstack(
            [
                np.where(img_labels == lbl)[0][:n_contrasts]
                for lbl in decoding_labels
            ]
        )
        dict_alignment[subject] = masker.transform(
            unmask(data[alignment_indices], individual_mask)
        )
        dict_decoding[subject] = masker.transform(
            unmask(data[decoding_indices], individual_mask)
        )
        dict_y[subject] = img_labels[decoding_indices].flatten()



    return Dataset(
        name=name,
        subjects=subjects,
        n_subjects=len(subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        test_sub=test_sub,
        external_template=external_template,
        task_name=task,
    )
