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
from nilearn.maskers import MultiNiftiMasker, SurfaceMasker
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
    output_dir: Optional[Path] = None
    time: Optional[float] = None
    dict_aligned: Optional[Dict[str, np.ndarray]] = None
    template: Optional[np.ndarray] = None
    solver: Optional[str] = None
    masker: Optional[MultiNiftiMasker] = None
    connectivity: Optional[str] = None


def sample_dataset(
    name: str,
    subjects: List[str],
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

    masker = get_masker(mask).fit(subjects_alignment_imgs)

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
        connectivity=connectivity,
    )


def get_masker(mask_img, n_jobs=1):
    masker = MultiNiftiMasker(
        mask_img=mask_img,
        standardize=True,
        reports=False,
        n_jobs=n_jobs,
        verbose=1,
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
    masker = get_masker(mask_img=intersect, n_jobs=N_JOBS)
    return masker, intersect, atlas_resampled


def get_labels_from_events(img_path, events_path, slice_time_ref=0.5, tr=2.0):
    # Load the events
    events_db = pd.read_csv(events_path, sep="\t")
    img = nib.load(img_path)
    n_scans = img.shape[3]
    frametimes = np.linspace(
        slice_time_ref, (n_scans - 1 + slice_time_ref) * tr, n_scans
    )
    y = np.array(["others"] * len(frametimes), dtype=str)

    # Loop over events and label frametimes by trial_type
    for _, ev in events_db.iterrows():
        onset, duration, trial_type = (
            ev["onset"],
            ev["duration"],
            ev["trial_type"],
        )
        in_event = (frametimes >= onset) & (frametimes < onset + duration)
        y[in_event] = trial_type
    return y


def load_ibc_db_bold(task):
    db = utils_data.data_parser(
        IBC_PATH,
        task_list=ALIGNMENT_TASKS,
    )
    db = db[db.path.str.contains("/func/") & db.path.str.contains("nii.gz")]
    db["events"] = db["path"].apply(
        lambda p: Path(str(p).replace("/3mm/", "/derivatives/")).with_name(
            Path(p)
            .name.removeprefix("wrdc")
            .replace("_bold.nii.gz", "_events.tsv")
        )
    )
    db.sort_values(by=["subject", "session", "path"])
    return db


def load_ibc_db_contrasts(task):
    db = utils_data.data_parser(
        IBC_PATH,
        task_list=[task] + ALIGNMENT_TASKS,
    ).sort_values(by=["subject", "task", "contrast", "path"])
    # Keep only pa - ap acquisitions
    db_filtered = db[
        (db.acquisition.isin(["ap", "pa"])) & (db.contrast != "preprocessed")
    ]
    # Add alignment column
    db_filtered["alignment"] = False
    db_filtered.loc[db_filtered["task"].isin(ALIGNMENT_TASKS), "alignment"] = (
        True
    )
    # Keep only one session
    db_one_ses = db_filtered.groupby(
        ["subject", "task", "contrast", "acquisition"], as_index=False
    ).tail(1)
    return db_one_ses


def fetch_ibc_vol(
    name: str = "IBC",
    task: str = None,
    n_parcels: int = 400,
    connectivity=None,
) -> Dataset:
    # Get the databases
    # db_bold = load_ibc_db_bold(task)
    db_contrasts = load_ibc_db_contrasts(task)

    # Get the subjects
    subjects = db_contrasts[~db_contrasts.alignment].subject.unique().tolist()

    # Get the masker
    masker, intersect, atlas_resampled = intersect_masker_atlas(
        IBC_GM_MASK, n_parcels
    )
    # Get the labels
    labels = apply_mask_fmri(atlas_resampled, intersect).astype(int)

    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()
    for subject in subjects:
        db_sub_decoding = db_contrasts[db_contrasts.subject == subject]
        dict_alignment[subject] = np.vstack(
            masker.fit_transform(
                db_sub_decoding[db_sub_decoding.alignment].path.tolist()
            )
        )
        dict_decoding[subject] = np.vstack(
            masker.fit_transform(
                db_sub_decoding[~db_sub_decoding.alignment].path.tolist()
            )
        )
        dict_y[subject] = db_sub_decoding[
            ~db_sub_decoding.alignment
        ].contrast.values.astype(str)

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
