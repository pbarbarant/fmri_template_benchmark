import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
from nilearn.masking import apply_mask_fmri
from nilearn.surface import PolyMesh, SurfaceImage
from tqdm import tqdm

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
    target_name: str
    task_name: str
    output_dir: Optional[Path] = None
    time: Optional[float] = None
    dict_aligned: Optional[Dict[str, np.ndarray]] = None
    template: Optional[np.ndarray] = None
    is_surf: bool = False
    target: Optional[np.ndarray] = None
    solver: Optional[str] = None


def check_init_dataset(dataset: Dataset) -> None:
    """Check that the dataset is correctly initialized."""
    first_subject = list(dataset.dict_alignment.keys())[0]
    n_samples_alignment = dataset.dict_alignment[first_subject].shape[-1]
    n_samples_decoding = dataset.dict_decoding[first_subject].img.shape[-1]
    mask_img = dataset.masker.mask_img_

    assert dataset.template is None, (
        "Template should be None at initialization"
    )
    assert dataset.dict_aligned is None, (
        "dict_aligned should be None at initialization"
    )

    for subject in dataset.dict_alignment:
        assert (
            dataset.dict_alignment[subject].shape[-1] == n_samples_alignment
        ), "Inconsistent number of samples in alignment"
        assert (
            dataset.dict_decoding[subject].img.shape[-1] == n_samples_decoding
        ), "Inconsistent number of samples in decoding"
        if dataset.is_surf:
            assert (
                dataset.dict_alignment[subject].shape[0] == mask_img.shape[0]
            ), "Alignment image shape does not match mask shape"
            assert (
                dataset.dict_decoding[subject].img.shape[0]
                == mask_img.shape[0]
            ), "Decoding image shape does not match mask shape"
        else:
            assert (
                dataset.dict_alignment[subject].shape[:-1] == mask_img.shape
            ), "Alignment image shape does not match mask shape"
            assert (
                dataset.dict_decoding[subject].img.shape[:-1] == mask_img.shape
            ), "Decoding image shape does not match mask shape"
        assert (
            dataset.dict_decoding[subject].y.shape[0]
            == dataset.dict_decoding[subject].img.shape[-1]
        ), "Number of labels does not match number of samples"


def log_dataset_info(dataset: Dataset) -> None:
    """Log the dataset information in a log file in the output folder."""
    output_folder = Path(__file__).parent.parent / "outputs/logs"
    output_folder.mkdir(exist_ok=True, parents=True)
    first_subject = list(dataset.dict_alignment.keys())[0]

    with open(output_folder / f"{dataset.name}.log", "w") as f:
        f.write(f"Dataset name: {dataset.name}\n")
        f.write(f"Number of subjects: {len(dataset.dict_alignment)}\n")
        f.write(f"List of subjects: {list(dataset.dict_alignment.keys())}\n")
        f.write(f"Image shape: {dataset.clustering_img.shape}\n")
        if dataset.is_surf:
            parts = dataset.clustering_img.data.parts
            n_parcels = max(parts["left"].max(), parts["right"].max()) - 1
            f.write(f"Number of parcels: {n_parcels}\n")
        else:
            f.write(
                f"Number of parcels: {len(np.unique(dataset.clustering_img.get_fdata())) - 1}\n"
            )
        f.write(
            f"List of decoding conditions: {np.unique(dataset.dict_decoding[first_subject].y)}\n"
        )
        f.write(
            f"Number of alignment samples: {dataset.dict_alignment[first_subject].shape[-1]}\n"
        )
        f.write(
            f"Number of decoding samples: {dataset.dict_decoding[first_subject].img.shape[-1]}\n"
        )


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
    target_name: str,
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
    dict_y = dict()
    subjects = ["sub-01", "sub-02"]

    # Generate a gaussian mixture for sub-01
    mean_1 = np.array([10, 0])
    mean_2 = np.array([-20, 0])
    data1 = np.random.randn(100, 2) + mean_1
    data2 = np.random.randn(100, 2) + mean_2
    data_sub1 = np.concatenate([data1, data2], axis=0)

    # Generate a gaussian mixture for sub-02
    mean_3 = np.array([0, 10])
    mean_4 = np.array([0, -20])
    data3 = np.random.randn(100, 2) + mean_3
    data4 = np.random.randn(100, 2) + mean_4
    data_sub2 = np.concatenate([data3, data4], axis=0)

    # Generate the labels
    y = np.arange(200) >= 100

    dict_alignment["sub-01"] = data_sub1
    dict_decoding["sub-01"] = data_sub1
    dict_y["sub-01"] = y

    dict_alignment["sub-02"] = data_sub2
    dict_decoding["sub-02"] = data_sub2
    dict_y["sub-02"] = y

    if target_name == "template":
        target = None
    else:
        target = dict_alignment[target_name]

    return Dataset(
        name=name,
        subjects=subjects,
        n_subjects=len(subjects),
        labels=np.ones(2),
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        target=target,
        target_name=target_name,
        task_name="simulated_task",
    )


def fetch_ibc(
    name: str = "IBC",
    target_name: str = "template",
    subjects: List[str] = None,
    task: str = None,
    n_parcels: int = 400,
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

    for subject in tqdm(subjects, desc="Processing IBC data"):
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

    if target_name == "template":
        target = None
    else:
        target = dict_alignment[target_name]

    return Dataset(
        name=name,
        subjects=subjects,
        n_subjects=len(subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        target=target,
        target_name=target_name,
        task_name=task,
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
    name: str = "IBC",
    target_name: str = "template",
    subjects: List[str] = None,
    task: str = None,
) -> Dataset:
    df = utils_data.make_surf_db(
        derivatives=IBC_SURF_PATH,
        subject_list=subjects,
        task_list=[task],
        acquisition="all",
    )
    mesh = load_fsaverage("fsaverage5")["pial"]
    atlas = fetch_atlas_surf_destrieux()
    labels = np.hstack([atlas["map_left"], atlas["map_right"] + atlas["map_left"].max()]).astype(int)
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
    for subject in tqdm(subjects, desc="Processing IBC data"):
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
        dict_y[subject] = decoding_df[decoding_df.side == "lh"].contrast.to_numpy()


    if target_name == "template":
        target = None
    else:
        target = dict_alignment[target_name]

    return Dataset(
        name=name,
        subjects=subjects,
        n_subjects=len(subjects),
        labels=labels,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        dict_y=dict_y,
        target=target,
        target_name=target_name,
        task_name=task,
        is_surf=True,
    )


def fetch_neuromod(n_parcels: int, target: str = "template") -> Dataset:
    DATA_PATH = Path(NEUROMOD_PATH)
    subjects = ["sub-01", "sub-02", "sub-03", "sub-05"]

    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing Neuromod data"):
        dict_alignment[subject] = load_img(
            DATA_PATH
            / f"{subject}_task-life_space-MNI152NLin2009cAsym_desc-postproc_bold.nii.gz"
        )
        dict_decoding[subject] = LabeledImage(
            img=load_img(DATA_PATH / f"{subject}.nii.gz"),
            y=pd.read_csv(
                DATA_PATH / f"{subject}_labels.csv", header=None
            ).values.flatten(),
        )

    masker = fit_masker(resolution=3, n_rois=n_parcels, n_jobs=N_JOBS)

    return None
