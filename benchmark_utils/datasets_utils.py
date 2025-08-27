from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import h5py
import numpy as np
from ibc_public import utils_data
from nilearn.datasets import (
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
    N_JOBS,
    NEUROMOD_PATH,
    IBC_GM_MASK,
)


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
    is_faulty: bool
    output_dir: Optional[Path] = None
    time: Optional[float] = None
    dict_aligned: Optional[Dict[str, np.ndarray]] = None
    template: Optional[np.ndarray] = None
    solver: Optional[str] = None
    masker: Optional[MultiNiftiMasker] = None


def get_masker(mask_img, n_jobs=1):
    masker = MultiNiftiMasker(
        mask_img=mask_img,
        standardize=True,
        reports=False,
        n_jobs=n_jobs,
        verbose=1,
    )
    return masker


def sample_dataset(
    name: str,
    subjects: List[str],
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()

    for subject in subjects:
        # Create a random alignment and decoding data for each subject
        dict_alignment[subject] = np.random.rand(100, 30)
        dict_decoding[subject] = np.random.rand(100, 30)
        dict_y[subject] = np.array([subject] * 100)

    return Dataset(
        name=name,
        subjects=subjects,
        n_subjects=len(subjects),
        labels=np.ones(30, dtype=int),
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        task_name="simulated_task",
    )


def fetch_ibc_vol(
    name: str = "IBC",
    subjects: List[str] = None,
    task: str = None,
    n_parcels: int = 400,
) -> Dataset:
    n_subjects = len(subjects)
    is_faulty = False
    df = utils_data.make_vol_db(
        derivatives=IBC_PATH,
        subject_list=subjects,
        task_list=[
            "ArchiStandard",
            "ArchiSocial",
            "ArchiSpatial",
            "ArchiEmotional",
        ]
        + [task],
    )
    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()

    # Get the masker
    mask_img = load_img(IBC_GM_MASK)
    schaefer_atlas = fetch_atlas_schaefer_2018(n_rois=n_parcels).maps
    atlas_resampled = resample_to_img(
        schaefer_atlas, mask_img, interpolation="nearest"
    )
    # Intersect the mask with the parcellation
    intersect = math_img("(img1*img2)>0", img1=atlas_resampled, img2=mask_img)
    masker = get_masker(mask_img=intersect, n_jobs=N_JOBS)

    # Get the labels
    labels = apply_mask_fmri(atlas_resampled, intersect).astype(int)

    # Get the alignment contrasts
    alignment_contrasts_all = (
        df[df.task.str.contains("Archi")]
        .drop_duplicates(subset=["subject", "contrast"], keep="first")
        .sort_values(by=["subject", "contrast"])
        .path.tolist()
    )
    alignment_array_all = np.array(
        masker.fit_transform(alignment_contrasts_all)
    )
    n_alignment_contrasts = len(
        df[df.task.str.contains("Archi")].contrast.unique()
    )
    alignment_array_list = [
        alignment_array_all[
            n_alignment_contrasts * i : n_alignment_contrasts * (i + 1)
        ]
        for i in range(n_subjects)
    ]
    dict_alignment = dict(zip(subjects, alignment_array_list))

    for subject in tqdm(subjects, desc="Processing IBC data"):
        subject_decoding_df = df[
            (df["subject"] == subject) & (df.task.str.contains(task))
        ]
        decoding_contrasts = subject_decoding_df.path.tolist()
        decoding_labels = subject_decoding_df.contrast.tolist()
        if len(decoding_contrasts) != 0:
            dict_decoding[subject] = np.vstack(
                masker.transform(decoding_contrasts)
            )
            dict_y[subject] = np.array(decoding_labels)
        else:
            print(
                f"Error processing subject {subject}, contrasts are not present"
            )

    return Dataset(
        name=name,
        subjects=list(dict_decoding.keys()),
        n_subjects=n_subjects,
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        task_name=task,
        is_faulty=is_faulty,
        masker=masker,
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
        print(
            f"Test subject {test_sub} data not found. Marking dataset as faulty."
        )
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
    path = data_path / (
        f"things.glmsingle/{subject}/glmsingle/output/"
        f"{subject}_task-things_space-T1w_model-fitHrfGLMdenoiseRR"
        "_stat-trialBetas_desc-zscore_statseries.h5"
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

    masker = get_masker(resolution=3, n_rois=n_parcels, n_jobs=N_JOBS)
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
