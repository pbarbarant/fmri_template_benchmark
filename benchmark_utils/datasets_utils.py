from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from nilearn.datasets import (
    fetch_atlas_schaefer_2018,
    load_mni152_gm_mask,
)
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.maskers import NiftiMasker
from nilearn.masking import apply_mask_fmri
from nilearn._utils.data_gen import generate_fake_fmri
from tqdm import tqdm
from benchmark_utils.conf import IBC_GM_MASK

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    name: str
    subjects: List[str]
    n_subjects: int
    labels: np.ndarray
    dict_alignment: Dict[str, np.ndarray]
    dict_decoding: Dict[str, np.ndarray]
    dict_y: Dict[str, np.ndarray]
    task_name: str
    target: str
    output_dir: Optional[Path] = None
    time: Optional[float] = None
    dict_aligned: Optional[Dict[str, np.ndarray]] = None
    template: Optional[np.ndarray] = None
    solver_name: Optional[str] = None
    masker: Optional[NiftiMasker] = None


def sample_dataset(
    name: str,
    subjects: List[str],
    target: str,
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()

    subjects_alignment_imgs = []
    subjects_decoding_imgs = []
    subjects_target = []
    for subject in subjects:
        alignment_img, mask = generate_fake_fmri()
        decoding_img, _, y = generate_fake_fmri(n_blocks=2)
        subjects_alignment_imgs.append(alignment_img)
        subjects_decoding_imgs.append(decoding_img)
        subjects_target.append(y)

    masker = NiftiMasker(mask).fit()

    for i, subject in enumerate(subjects):
        # Create a random alignment and decoding data for each subject
        dict_alignment[subject] = masker.transform(subjects_alignment_imgs[i])
        dict_decoding[subject] = masker.transform(subjects_decoding_imgs[i])
        dict_y[subject] = subjects_target[i]

    n_voxels = list(dict_alignment.values())[0].shape[1]
    labels = np.hstack(
        [np.ones(n_voxels // 2), 2 * np.ones(n_voxels - n_voxels // 2)]
    ).astype(int)
    return Dataset(
        name=name,
        subjects=list(dict_decoding.keys()),
        n_subjects=len(subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        task_name="simulated_task",
        masker=masker,
        target=target,
    )


ALIGNMENT_TASKS = [
    "ArchiStandard",
    "ArchiSocial",
    "ArchiEmotional",
    "ArchiSpatial",
    "HcpEmotion",
    "HcpGambling",
    "HcpMotor",
    "HcpLanguage",
    "HcpRelational",
    "HcpSocial",
    "HcpWm",
]


def intersect_masker_atlas(mask_path, n_parcels):
    if mask_path is str:
        mask_img = load_img(mask_path)
    else:
        mask_img = mask_path
    schaefer_atlas = fetch_atlas_schaefer_2018(n_rois=n_parcels).maps
    atlas_resampled = resample_to_img(
        schaefer_atlas,
        mask_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    # Intersect the mask with the parcellation
    intersect = math_img("(img1*img2)>0", img1=atlas_resampled, img2=mask_img)
    return intersect, atlas_resampled


def parse_subjects(data_path: Path) -> List[str]:
    niftis = sorted(data_path.glob("*.nii.gz"))
    subjects = [f.name[:-7] for f in niftis]
    return subjects


def fetch_dataset(
    name: str,
    subjects: List[str],
    target: str,
    data_path: Path,
    task: str,
    n_parcels: int = 400,
) -> Dataset:
    # Get the mask_img
    if "Neuromod" in name:
        mask_path = load_mni152_gm_mask(3)
    else:
        mask_path = IBC_GM_MASK
    mask_img, atlas_resampled = intersect_masker_atlas(mask_path, n_parcels)

    # Get the labels
    labels = apply_mask_fmri(atlas_resampled, mask_img).astype(int)

    # Get the masker
    masker = NiftiMasker(mask_img=mask_img).fit()

    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()
    for subject in tqdm(subjects, desc="Loading subjects data"):
        # Get the contrasts
        runs = (
            pd.read_csv(data_path / f"{subject}_runs.csv", header=None)
            .values.astype(str)
            .ravel()
        )
        y = (
            pd.read_csv(data_path / f"{subject}_labels.csv", header=None)
            .values.astype(str)
            .ravel()
        )
        X = masker.transform(data_path / f"{subject}.nii.gz")
        X_alignment, X_decoding, _, y_decoding = train_test_split(
            X, y, test_size=0.8, stratify=runs, random_state=0
        )

        dict_alignment[subject] = X_alignment
        dict_decoding[subject] = X_decoding
        dict_y[subject] = y_decoding

    return Dataset(
        name=name,
        subjects=list(dict_decoding.keys()),
        n_subjects=len(subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        task_name=task,
        target=target,
        masker=masker,
    )
