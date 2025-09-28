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
from nilearn.image import load_img, math_img, resample_to_img, concat_imgs
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.masking import apply_mask_fmri, unmask
from nilearn.surface import PolyMesh, SurfaceImage
from nilearn._utils.data_gen import generate_fake_fmri
from tqdm import tqdm
import nibabel as nib
from benchmark_utils.conf import (
    IBC_PATH,
    IBC_SURF_PATH,
    N_JOBS,
    NEUROMOD_PATH,
    IBC_GM_MASK,
)
import pandas as pd


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
    alignment_modality: str
    output_dir: Optional[Path] = None
    time: Optional[float] = None
    dict_aligned: Optional[Dict[str, np.ndarray]] = None
    template: Optional[np.ndarray] = None
    solver: Optional[str] = None
    masker: Optional[NiftiMasker] = None
    connectivity: Optional[str] = None


def sample_dataset(
    name: str,
    subjects: List[str],
    alignment_modality: str,
    connectivity: Optional[str] = None,
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()

    subjects_alignment_imgs = []
    subjects_decoding_imgs = []
    subjects_target = []
    for subject in subjects:
        alignment_img, mask = generate_fake_fmri()
        decoding_img, _, target = generate_fake_fmri(n_blocks=2)
        subjects_alignment_imgs.append(alignment_img)
        subjects_decoding_imgs.append(decoding_img)
        subjects_target.append(target)

    masker = get_niftimasker(mask).fit(subjects_alignment_imgs)

    for i, subject in enumerate(subjects):
        # Create a random alignment and decoding data for each subject
        dict_alignment[subject] = masker.transform(subjects_alignment_imgs[i])
        dict_decoding[subject] = masker.transform(subjects_decoding_imgs[i])
        dict_y[subject] = subjects_target[i]

    return Dataset(
        name=name,
        subjects=list(dict_decoding.keys()),
        n_subjects=len(subjects),
        labels=np.ones(list(dict_alignment.values())[0].shape[1], dtype=int),
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        task_name="simulated_task",
        masker=masker,
        alignment_modality=alignment_modality,
        connectivity=connectivity,
    )


def get_niftimasker(mask_img, runs, detrend, t_r, smoothing_fwhm):
    masker = NiftiMasker(
        mask_img=mask_img,
        detrend=detrend,
        standardize=True,
        reports=False,
        verbose=1,
        smoothing_fwhm=smoothing_fwhm,
        t_r=t_r,
        runs=runs,
    )
    return masker


def get_surfacemasker(
    mask_img, runs=None, detrend=False, t_r=None, smoothing_fwhm=None
):
    masker = SurfaceMasker(
        mask_img=mask_img,
        detrend=detrend,
        standardize=True,
        reports=False,
        verbose=1,
        smoothing_fwhm=smoothing_fwhm,
        t_r=t_r,
        clean_args={"runs": runs},
    )
    return masker


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
    mask_img = load_img(mask_path)
    schaefer_atlas = fetch_atlas_schaefer_2018(n_rois=n_parcels).maps
    atlas_resampled = resample_to_img(
        schaefer_atlas, mask_img, interpolation="nearest"
    )
    # Intersect the mask with the parcellation
    intersect = math_img("(img1*img2)>0", img1=atlas_resampled, img2=mask_img)
    return intersect, atlas_resampled


def load_ibc_db_bold(task, path, surf=False):
    db = utils_data.data_parser(
        path,
        task_list=[task],
    )
    db = db[db.path.str.contains("/func/") & db.path.str.contains("nii.gz")]
    db.sort_values(by=["subject", "session", "path"])
    if surf:
        rows = []
        for _, row in db.iterrows():
            for side in ["lh", "rh"]:
                new_row = row.copy()
                new_row["mesh"] = "fsaverage5"
                new_row["side"] = side
                new_row["path"] = (
                    row["path"]
                    .replace("/func/", "/freesurfer/")
                    .replace("wrdcsub-", "rdcsub-")
                    .replace("_bold.nii.gz", f"_bold_fsaverage5_{side}.gii")
                )
                rows.append(new_row)
        db = pd.DataFrame(rows)
    return db


def load_ibc_db_contrasts(task, path, surf=False):
    if surf:
        space, extension = "fsaverage5", ".gii"
    else:
        space, extension = "MNI152", ".nii.gz"
    db = utils_data.make_db(
        path,
        space=space,
        extension=extension,
        task_list=[task],
        acquisition="all",
    )
    # Add acquisition
    db["acquisition"] = db["path"].str.extract(r"dir-(ap|pa)")
    # Keep only pa - ap acquisitions
    db = db[(db.acquisition.isin(["ap", "pa"]))]
    # Keep only one session
    db = db[
        db.groupby(["subject", "task"])["session"].transform("max")
        == db["session"]
    ]

    # Add alignment column
    rng = np.random.default_rng(0)
    db["alignment"] = False
    # For each subject/contrast, randomly pick one run of each acquisition
    if "side" in db.columns:
        db_lh = db[db["side"] == "lh"].copy()
        db_rh = db[db["side"] == "rh"].copy()

        for (_, group_lh), (_, group_rh) in zip(
            db_lh.groupby(["subject", "contrast"]),
            db_rh.groupby(["subject", "contrast"]),
        ):
            i = rng.integers(len(group_lh.index))
            lh_idx = group_lh.index[i]
            rh_idx = group_rh.index[i]
            db.loc[[lh_idx, rh_idx], "alignment"] = True
    else:
        for _, group in db.groupby(["subject", "contrast"]):
            chosen_idx = rng.choice(group.index)
            db.loc[chosen_idx, "alignment"] = True

    return db.sort_values(by=["subject", "task", "contrast", "path"])


def fetch_ibc_vol(
    name: str = "IBC_vol",
    task: str = None,
    alignment_modality: str = "contrast",
    n_parcels: int = 400,
    connectivity=None,
) -> Dataset:
    # Get the databases
    db_bold = load_ibc_db_bold(task, path=IBC_PATH)
    db_contrasts = load_ibc_db_contrasts(task, path=IBC_PATH)

    # Get the subjects
    subjects = db_contrasts.subject.unique().tolist()

    # Get the mask_img
    mask_img, atlas_resampled = intersect_masker_atlas(IBC_GM_MASK, n_parcels)

    # Get the labels
    labels = apply_mask_fmri(atlas_resampled, mask_img).astype(int)

    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()
    for subject in subjects:
        # Get the contrasts for decoding
        db_sub_decoding = db_contrasts[
            (db_contrasts.subject == subject) & ~db_contrasts.alignment
        ]
        dict_y[subject] = db_sub_decoding.contrast.values.astype(str)
        dict_decoding[subject] = apply_mask_fmri(
            db_sub_decoding.path.tolist(), mask_img
        )

        # Align with the bold
        if alignment_modality == "bold":
            db_sub_alignment = db_bold[(db_bold.subject == subject)]
            runs = np.hstack(
                [
                    i * np.ones(nib.load(path).shape[-1])
                    for i, path in enumerate(db_sub_alignment.path)
                ]
            )
            masker = get_niftimasker(
                mask_img, runs, detrend=True, t_r=2.0, smoothing_fwhm=5
            )
            dict_alignment[subject] = masker.fit_transform(
                concat_imgs(db_sub_alignment.path.tolist())
            )
        # Align with the contrasts
        else:
            db_sub_alignment = db_contrasts[
                (db_contrasts.subject == subject) & db_contrasts.alignment
            ]
            masker = get_niftimasker(
                mask_img,
                runs=None,
                detrend=False,
                t_r=None,
                smoothing_fwhm=None,
            )
            dict_alignment[subject] = masker.fit_transform(
                concat_imgs(db_sub_alignment.path.tolist())
            )

    return Dataset(
        name=name,
        subjects=list(dict_decoding.keys()),
        n_subjects=len(subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        task_name=task,
        masker=masker,
        connectivity=connectivity,
        alignment_modality=alignment_modality,
    )


def load_surface_img(
    db: List[str], mesh: PolyMesh
) -> (
    SurfaceImage
):  # -> Any | SurfaceImage:# -> Any | SurfaceImage:# -> Any | SurfaceImage:
    paths_lh = db[db.side == "lh"].path.tolist()
    paths_rh = db[db.side == "rh"].path.tolist()
    surf_imgs = []
    for path_lh, path_rh in zip(paths_lh, paths_rh):
        surf_img = SurfaceImage(
            mesh=mesh, data={"left": path_lh, "right": path_rh}
        )
        surf_imgs.append(surf_img)
    return concat_imgs(surf_imgs)


def load_labels_fs5():
    atlas = fetch_atlas_surf_destrieux()
    labels_lh = atlas["map_left"]
    labels_rh = atlas["map_right"]
    offset = labels_lh.max()
    return np.hstack([labels_lh, offset + labels_rh])


def fetch_ibc_surf(
    name: str = "IBC_surf",
    task: str = None,
    alignment_modality: str = "contrast",
    connectivity=None,
) -> Dataset:
    # Get the mesh
    mesh = load_fsaverage("fsaverage5")["pial"]
    # Get the databases
    db_bold = load_ibc_db_bold(task, IBC_SURF_PATH, surf=True)
    db_contrasts = load_ibc_db_contrasts(task, IBC_SURF_PATH, surf=True)

    # Get the subjects
    subjects = db_contrasts.subject.unique().tolist()

    # Get the labels
    labels = load_labels_fs5()

    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()
    for subject in subjects:
        # Get the contrasts for decoding
        db_sub_decoding = db_contrasts[
            (db_contrasts.subject == subject) & ~db_contrasts.alignment
        ]
        dict_y[subject] = db_sub_decoding[
            db_sub_decoding.side == "lh"
        ].contrast.values.astype(str)

        masker_contrasts = get_surfacemasker(mask_img=None)
        dict_decoding[subject] = masker_contrasts.fit_transform(
            load_surface_img(db_sub_decoding, mesh)
        )

        # Align with the bold
        if alignment_modality == "bold":
            db_sub_alignment = db_bold[(db_bold.subject == subject)]
            runs = np.hstack(
                [
                    i * np.ones(len(nib.load(path).darrays))
                    for i, path in enumerate(
                        db_sub_alignment[db_sub_alignment.side == "lh"].path
                    )
                ]
            )
            masker_bold = get_surfacemasker(
                mask_img=None,
                runs=runs,
                detrend=True,
                t_r=2.0,
                smoothing_fwhm=5,
            )
            dict_alignment[subject] = masker_bold.fit_transform(
                load_surface_img(db_sub_alignment, mesh)
            )

        # Align with the contrasts
        else:
            db_sub_alignment = db_contrasts[
                (db_contrasts.subject == subject) & db_contrasts.alignment
            ]
            dict_alignment[subject] = masker_contrasts.fit_transform(
                load_surface_img(db_sub_alignment, mesh)
            )

    return Dataset(
        name=name,
        subjects=list(dict_decoding.keys()),
        n_subjects=len(subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        task_name=task,
        masker=masker_contrasts,
        connectivity=connectivity,
        alignment_modality=alignment_modality,
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

    masker = get_niftimasker(resolution=3, n_rois=n_parcels, n_jobs=N_JOBS)
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
