from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.datasets import (
    fetch_atlas_schaefer_2018,
    load_mni152_gm_mask,
)
from nilearn.image import load_img, math_img, resample_to_img
from nilearn.maskers import NiftiMasker
from nilearn.masking import apply_mask_fmri
from sklearn.model_selection import StratifiedKFold

from benchmark_utils.conf import GM_MASK


@dataclass
class Fold:
    index: int
    dict_alignment: Dict[str, np.ndarray]
    dict_decoding: Dict[str, np.ndarray]
    dict_y: Dict[str, np.ndarray]
    dict_aligned: Optional[Dict[str, np.ndarray]] = None
    time: Optional[float] = None


@dataclass
class Dataset:
    name: str
    subjects: List[str]
    n_subjects: int
    labels: np.ndarray
    folds: List[Fold]
    task_name: str
    target: str
    output_dir: Optional[Path] = None
    solver_name: Optional[str] = None
    masker: Optional[NiftiMasker] = None


def parse_subjects(data_path: Path) -> List[str]:
    niftis = sorted(data_path.glob("*.nii.gz"))
    subjects = [f.name[:-7] for f in niftis]
    return subjects


def _require_dataset_files(data_path: Path, subjects: List[str]) -> None:
    missing_files = []
    for subject in subjects:
        for suffix in [".nii.gz", "_runs.csv", "_labels.csv"]:
            file_path = data_path / f"{subject}{suffix}"
            if not file_path.exists():
                missing_files.append(str(file_path))
    if missing_files:
        missing_str = "\n".join(f"- {file_path}" for file_path in missing_files)
        raise FileNotFoundError(
            f"Missing dataset files under {data_path}:\n{missing_str}"
        )


def sample_dataset(
    name: str,
    subjects: List[str],
    target: str,
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()

    subjects_imgs = []
    subjects_target = []
    for subject in subjects:
        img, mask, y = generate_fake_fmri(length=100, n_blocks=2, block_size=10)
        subjects_imgs.append(img)
        subjects_target.append(y)

    runs = np.ones(img.shape[-1])
    masker = NiftiMasker(mask).fit()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds_indices = list(skf.split(runs, runs))

    folds = []
    for fold_idx, (decoding_idx, alignment_idx) in enumerate(folds_indices):
        dict_alignment = {
            sub: masker.transform(img)[alignment_idx]
            for sub, img in zip(subjects, subjects_imgs)
        }
        dict_decoding = {
            sub: masker.transform(img)[decoding_idx]
            for sub, img in zip(subjects, subjects_imgs)
        }
        dict_y = {
            sub: y[decoding_idx] for sub, y in zip(subjects, subjects_target)
        }
        folds.append(
            Fold(
                index=fold_idx,
                dict_alignment=dict_alignment,
                dict_decoding=dict_decoding,
                dict_y=dict_y,
            )
        )

    n_voxels = list(dict_alignment.values())[0].shape[1]
    labels = np.hstack(
        [np.ones(n_voxels // 2), 2 * np.ones(n_voxels - n_voxels // 2)]
    ).astype(int)
    return Dataset(
        name=name,
        subjects=list(dict_decoding.keys()),
        n_subjects=len(subjects),
        labels=labels,
        folds=folds,
        task_name="simulated_task",
        masker=masker,
        target=target,
    )


def intersect_masker_atlas(mask_img, n_parcels):
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


def fetch_dataset(
    name: str,
    subjects: List[str],
    target: str,
    data_path: Path,
    task: str,
    n_parcels: int = 400,
) -> Dataset:
    _require_dataset_files(data_path, subjects)

    # Get the mask_img
    if "Neuromod" in name:
        mask_img = load_mni152_gm_mask(3)
    else:
        mask_img = load_img(GM_MASK)
    mask_img, atlas_resampled = intersect_masker_atlas(mask_img, n_parcels)

    # Get the labels
    labels = apply_mask_fmri(atlas_resampled, mask_img).astype(int)

    # Get the masker
    masker = NiftiMasker(mask_img=mask_img).fit()

    # All runs/labels are structured similarly
    runs = (
        pd.read_csv(data_path / f"{subjects[0]}_runs.csv", header=None)
        .values.astype(str)
        .ravel()
    )
    y = (
        pd.read_csv(data_path / f"{subjects[0]}_labels.csv", header=None)
        .values.astype(str)
        .ravel()
    )

    subjects_data = [
        (masker.transform(data_path / f"{s}.nii.gz")) for s in subjects
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds_indices = list(skf.split(runs, runs))

    folds = []
    for fold_idx, (decoding_idx, alignment_idx) in enumerate(folds_indices):
        dict_alignment = {
            s: data[alignment_idx] for s, data in zip(subjects, subjects_data)
        }
        dict_decoding = {
            s: data[decoding_idx] for s, data in zip(subjects, subjects_data)
        }
        dict_y = {s: y[decoding_idx] for s in subjects}
        folds.append(
            Fold(
                index=fold_idx,
                dict_alignment=dict_alignment,
                dict_decoding=dict_decoding,
                dict_y=dict_y,
            )
        )

    return Dataset(
        name=name,
        subjects=list(dict_decoding.keys()),
        n_subjects=len(subjects),
        labels=labels,
        folds=folds,
        task_name=task,
        target=target,
        masker=masker,
    )
