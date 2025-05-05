import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ibc_public import utils_data
from nibabel.nifti1 import Nifti1Image
from nilearn import image
from nilearn.datasets import (
    fetch_atlas_schaefer_2018,
    fetch_atlas_surf_destrieux,
    load_fsaverage,
    load_mni152_gm_mask,
)
from fmralign._utils import _intersect_clustering_mask
from fmralign.preprocessing import ParcellationMasker
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.maskers._utils import concatenate_surface_images
from nilearn.surface import PolyMesh, SurfaceImage
from tqdm import tqdm

from benchmark_utils.conf import (
    BUDAPEST_PATH,
    FORREST_PATH,
    HCP_PATH,
    IBC_PATH,
    IBC_SURF_PATH,
    MEMORY,
    NEUROMOD_PATH,
    NSD_PATH,
    N_JOBS,
    RAIDERS_PATH,
)


@dataclass
class LabeledImage:
    img: Nifti1Image | SurfaceImage
    y: np.ndarray


@dataclass
class Dataset:
    name: str
    subjects: List[str]
    dict_alignment: Dict[str, Nifti1Image]
    dict_decoding: Dict[str, LabeledImage]
    masker: NiftiMasker | SurfaceMasker
    clustering_img: Nifti1Image
    output_dir: Optional[Path] = None
    time: Optional[float] = None
    parcel_masker: Optional[ParcellationMasker] = None
    dict_aligned: Optional[Dict[str, LabeledImage]] = None
    template: Optional[LabeledImage] = None
    is_surf: bool = False
    paradigm: str = "task"
    target: str = "template"
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


def fetch_clustering_img(
    target_img: Nifti1Image, n_rois: int = 400
) -> Nifti1Image:  # -> FileBasedImage | Nifti1Image | Any:# -> FileBasedImage | Nifti1Image | Any:
    clustering_img = fetch_atlas_schaefer_2018(
        n_rois=n_rois, data_dir=MEMORY.location / "atlas"
    )["maps"]
    resampled_img = image.resample_to_img(clustering_img, target_img)
    int_img = image.math_img("img.astype(int)", img=resampled_img)
    return int_img


def fit_mni152_masker(resolution: int = 3) -> NiftiMasker:
    mask_img = load_mni152_gm_mask(resolution=resolution)
    return NiftiMasker(mask_img=mask_img, memory=MEMORY, memory_level=1).fit()


def fit_masker(imgs, mask_img=None, detrend=False, t_r=None) -> NiftiMasker:
    return NiftiMasker(
        memory=MEMORY,
        mask_img=mask_img,
        memory_level=1,
        standardize=True,
        detrend=detrend,
        t_r=t_r,
        n_jobs=N_JOBS,
    ).fit(imgs)


def get_masker_clustering_img(dict_alignment, subjects, n_parcels):
    masker = fit_masker(
        [dict_alignment[subject] for subject in subjects],
    )
    clustering_img = fetch_clustering_img(masker.mask_img_, n_parcels)
    if 0 in masker.transform(clustering_img):
        reduced_mask = _intersect_clustering_mask(
            clustering_img, masker.mask_img_
        )
        # Update the masker
        masker = fit_masker(
            [dict_alignment[subject] for subject in subjects],
            mask_img=reduced_mask,
        )
    return masker, clustering_img


def sample_movie_segment(n_segments: int, masker: NiftiMasker) -> LabeledImage:
    mask_img = masker.mask_img_
    n_voxels = masker.transform(mask_img).shape[1]
    segment_len = 5
    data = np.random.randn(n_segments * segment_len, n_voxels)
    img = masker.inverse_transform(data)
    y = np.arange(n_segments).repeat(segment_len)
    return LabeledImage(
        img=img,
        y=y,
    )


def sample_dataset(
    name: str,
    target: str,
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
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

    # Reshape the data to (2, 1, 1, 200)
    data_sub1_reshaped = data_sub1.T.reshape(2, 1, 1, 200)
    data_sub2_reshaped = data_sub2.T.reshape(2, 1, 1, 200)

    # Convert to NIfTI images
    img1 = Nifti1Image(data_sub1_reshaped, affine=np.eye(4))
    img2 = Nifti1Image(data_sub2_reshaped, affine=np.eye(4))

    dict_alignment["sub-01"] = img1
    dict_decoding["sub-01"] = LabeledImage(img=img1, y=y)

    dict_alignment["sub-02"] = img2
    dict_decoding["sub-02"] = LabeledImage(img2, y=y)

    # Generate mask of all 1s
    mask_data = np.ones((2, 1, 1))
    mask_img = Nifti1Image(mask_data, affine=np.eye(4))

    # Define and fit the masker
    masker = NiftiMasker(mask_img=mask_img, standardize=False).fit(
        [img1, img2]
    )

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=mask_img,
        target=target,
    )


def sample_movie_dataset(
    name,
    masker,
    clustering_img,
    subjects,
    n_segments_alignement,
    n_segments_decoding,
) -> Dataset:
    print(f"Generating fold {name}")
    dict_alignment = dict()
    dict_decoding = dict()
    for subject in subjects:
        # Generate random surface images for each subject.
        dict_alignment[subject] = sample_movie_segment(
            n_segments_alignement, masker
        ).img
        dict_decoding[subject] = sample_movie_segment(
            n_segments_decoding, masker
        )

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        paradigm="movie",
    )


def fetch_ibc(
    name: str = "IBC",
    target: str = "template",
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
    for subject in tqdm(subjects, desc="Processing IBC data"):
        df_sub = df[(df.subject == subject)]
        # For each contrast, keep randomly one path
        alignment_df = df_sub.groupby(["contrast"]).apply(
            lambda x: x.sample(1, random_state=0)
        )
        # Put the rest in decoding_df
        decoding_df = df_sub[~df_sub.index.isin(alignment_df.index)]
        dict_alignment[subject] = image.concat_imgs(
            alignment_df.path.to_list()
        )
        dict_decoding[subject] = LabeledImage(
            img=image.concat_imgs(decoding_df.path.to_list()),
            y=decoding_df.contrast.to_numpy(),
        )

    masker, clustering_img = get_masker_clustering_img(
        dict_alignment, subjects, n_parcels
    )

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        target=target,
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
    return concatenate_surface_images(surf_imgs)


def fetch_ibc_surf(
    name: str = "IBC",
    target: str = "template",
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
    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing IBC data"):
        alignment_df = df[
            (df.subject == subject) & (df.path.str.contains("_dir-ap"))
        ]
        decoding_df = df[
            (df.subject == subject) & (df.path.str.contains("_dir-pa"))
        ]
        dict_alignment[subject] = load_surface_img(
            alignment_df.path.to_list(), mesh
        )
        dict_decoding[subject] = LabeledImage(
            img=load_surface_img(decoding_df.path.to_list(), mesh),
            y=decoding_df[decoding_df.side == "lh"].contrast.to_numpy(),
        )

    atlas = fetch_atlas_surf_destrieux()
    clustering_img = SurfaceImage(
        mesh=mesh,
        data={
            "left": atlas["map_left"],
            "right": atlas["map_right"],
        },
    )

    masker = SurfaceMasker(
        memory=MEMORY, memory_level=1, standardize=True
    ).fit([dict_alignment[subject] for subject in subjects])

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        is_surf=True,
        target=target,
    )


@MEMORY.cache
def get_valid_hcp_subjects_list(derivatives: str) -> List[str]:
    """Check which subjects have all the tasks."""
    tasks = [
        "EMOTION",
        "GAMBLING",
        "LANGUAGE",
        "MOTOR",
        "RELATIONAL",
        "SOCIAL",
        "WM",
    ]
    valid_subject_list = []
    subject_paths = sorted(glob.glob(derivatives + "**/"))
    subject_list = [Path(path).name for path in subject_paths]
    for subject in tqdm(subject_list, desc="Validating HCP data"):
        for task in tasks:
            lr_zmaps_path = Path(derivatives) / subject / task / "LR/z_maps"
            rl_zmaps_path = Path(derivatives) / subject / task / "RL/z_maps"
            if not lr_zmaps_path.exists() or not rl_zmaps_path.exists():
                break
        else:
            valid_subject_list.append(subject)

    return valid_subject_list


@MEMORY.cache
def make_hcp_db(
    derivatives: str,
    subject_list: List[str],
    tasks: List[str],
    phase_encoding: str = "LR",
) -> pd.DataFrame:
    """Returns a dataframe listing HCP data."""
    paths = []
    contrasts = []
    subjects = []
    if tasks == "all":
        tasks = [
            "EMOTION",
            "GAMBLING",
            "LANGUAGE",
            "MOTOR",
            "RELATIONAL",
            "SOCIAL",
            "WM",
        ]
    for subject in tqdm(subject_list, desc="Loading HCP data"):
        for task in tasks:
            zmaps_path = (
                Path(derivatives) / subject / task / f"{phase_encoding}/z_maps"
            )
            if not zmaps_path.exists():
                raise FileNotFoundError(f"Path {zmaps_path} does not exist.")
            zmaps = glob.glob(str(zmaps_path / "*.nii.gz"))
            for path in zmaps:
                contrast = (
                    Path(path).stem.removeprefix("z_").removesuffix(".nii")
                )
                if (
                    contrast.startswith("neg")
                    or "-" in contrast
                    or "_" in contrast
                ):
                    continue
                else:
                    paths.append(path)
                    contrasts.append(contrast)
                    subjects.append(subject)

    df = pd.DataFrame(
        {
            "path": paths,
            "subject": subjects,
            "contrast": contrasts,
        }
    )
    return df


def fetch_hcp(
    name: str = "HCP",
    target: str = "template",
    n_subjects: int = None,
    task: str = None,
    n_parcels: int = 400,
) -> Dataset:
    valid_subjects_list = get_valid_hcp_subjects_list(HCP_PATH)
    subject_list = valid_subjects_list[:n_subjects]
    alignment_df = make_hcp_db(
        derivatives=HCP_PATH,
        subject_list=subject_list,
        tasks="all",
        phase_encoding="LR",
    )

    decoding_df = make_hcp_db(
        derivatives=HCP_PATH,
        subject_list=subject_list,
        tasks=[task],
        phase_encoding="RL",
    )

    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subject_list, desc="Processing HCP data"):
        sub_alignment_df = alignment_df[(alignment_df.subject == subject)]
        sub_decoding_df = decoding_df[(decoding_df.subject == subject)]
        dict_alignment[subject] = image.concat_imgs(
            sub_alignment_df.path.to_list()
        )
        dict_decoding[subject] = LabeledImage(
            img=image.concat_imgs(sub_decoding_df.path.to_list()),
            y=sub_decoding_df.contrast.to_numpy(),
        )

    masker = fit_masker(
        [dict_alignment[subject] for subject in subject_list],
    )
    clustering_img = fetch_clustering_img(masker.mask_img_, n_parcels)

    return Dataset(
        name=name,
        subjects=subject_list,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        target=target,
    )


def fetch_forrest(
    subjects: List[str] = None,
    target: str = "template",
    n_parcels: int = 400,
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing Forrest data"):
        dict_alignment[subject] = image.load_img(
            Path(FORREST_PATH) / f"gm_3mm_{subject}.nii.gz"
        )
        dict_decoding[subject] = LabeledImage(
            img=image.load_img(
                Path(FORREST_PATH) / f"forrest_gm_3mm_{subject}.nii.gz"
            ),
            y=pd.read_csv(
                Path(FORREST_PATH) / f"{subject}_labels.csv", header=None
            ).to_numpy(),
        )

    masker, clustering_img = get_masker_clustering_img(
        dict_alignment, subjects, n_parcels
    )

    return Dataset(
        name="Forrest",
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        target=target,
    )


def fetch_nsd(
    name: str = "NSD",
    target: str = "template",
    subjects: List[str] = None,
    n_parcels: int = 400,
) -> Dataset:
    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing NSD data"):
        dict_alignment[subject] = image.load_img(
            Path(NSD_PATH) / f"alignment/{subject}.nii.gz"
        )
        dict_decoding[subject] = LabeledImage(
            img=image.load_img(Path(NSD_PATH) / f"decoding/{subject}.nii.gz"),
            y=pd.read_csv(
                Path(NSD_PATH) / f"decoding/{subject}_labels.csv", header=None
            ).values.flatten(),
        )

    masker = fit_masker(
        [dict_alignment[subject] for subject in subjects],
    )
    clustering_img = fetch_clustering_img(masker.mask_img_, n_parcels)

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        target=target,
    )


def load_budapest_img_labels(
    image_path: Path,
) -> tuple[Nifti1Image, np.ndarray]:
    img = image.load_img(image_path)
    img_len = img.shape[-1]
    n_chunks = img_len // 15
    indices = np.concatenate(
        [np.arange(i * 15, i * 15 + 5) for i in range(n_chunks)]
    )
    y = np.arange(n_chunks).repeat(5)
    return image.index_img(img, indices), y


def fetch_budapest(
    n_parcels: int = 400,
    target: str = "template",
    lo_run: int = 1,
) -> Dataset:
    # Fetch subjects
    subjects = sorted(
        [p.name for p in Path(BUDAPEST_PATH).glob("sub-*") if p.is_dir()]
    )[:2]

    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing Budapest data"):
        alignment_imgs = []
        for run in range(1, 3):
            if run == lo_run:
                img, y = load_budapest_img_labels(
                    Path(BUDAPEST_PATH)
                    / subject
                    / "func"
                    / f"{subject}_task-movie_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                )
                dict_decoding[subject] = LabeledImage(img=img, y=y)
            else:
                img = image.load_img(
                    Path(BUDAPEST_PATH)
                    / subject
                    / "func"
                    / f"{subject}_task-movie_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                )
                alignment_imgs.append(img)
        dict_alignment[subject] = image.concat_imgs(alignment_imgs)

    masker, clustering_img = get_masker_clustering_img(
        dict_alignment, subjects, n_parcels
    )

    return Dataset(
        name=f"Budapest_run-{lo_run:02d}",
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        paradigm="movie",
        target=target,
    )


def load_raiders_img_labels(
    image_path: Path,
) -> tuple[
    Nifti1Image, np.ndarray
]:  # -> tuple[FileBasedImage | Nifti1Image | Any, ndarray[Any, dt...:
    img = image.load_img(image_path)
    img_len = img.shape[-1]
    n_chunks = img_len // 15
    indices = np.concatenate(
        [np.arange(i * 15, i * 15 + 5) for i in range(n_chunks)]
    )
    y = np.arange(n_chunks).repeat(5)
    return image.index_img(img, indices), y


def fetch_raiders(
    n_parcels: int = 400,
    target: str = "template",
    lo_run: int = 1,
) -> Dataset:
    # Fetch subjects
    subjects = sorted(
        [p.name for p in Path(RAIDERS_PATH).glob("sub-*") if p.is_dir()]
    )

    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing Raiders data"):
        alignment_imgs = []
        for run in range(1, 9):
            prefix = Path(RAIDERS_PATH) / subject / "func"
            img_path = (
                prefix
                / glob.glob(
                    str(
                        prefix
                        / f"*{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
                    )
                )[0]
            )
            if run == lo_run:
                img, y = load_raiders_img_labels(img_path)
                dict_decoding[subject] = LabeledImage(img=img, y=y)
            else:
                img = image.load_img(img_path)
                alignment_imgs.append(img)
        dict_alignment[subject] = image.concat_imgs(alignment_imgs)

    masker = fit_masker(
        [dict_alignment[subject] for subject in subjects],
        detrend=True,
        t_r=1.0,
    )
    clustering_img = fetch_clustering_img(masker.mask_img_, n_parcels)

    return Dataset(
        name=f"Raiders_run-{lo_run:02d}",
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        paradigm="movie",
        target=target,
    )


def fetch_neuromod(n_parcels: int, target: str = "template") -> Dataset:
    DATA_PATH = Path(NEUROMOD_PATH)
    subjects = ["sub-01", "sub-02", "sub-03", "sub-05"]

    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing Neuromod data"):
        dict_alignment[subject] = image.load_img(
            DATA_PATH
            / f"{subject}_task-life_space-MNI152NLin2009cAsym_desc-postproc_bold.nii.gz"
        )
        dict_decoding[subject] = LabeledImage(
            img=image.load_img(DATA_PATH / f"{subject}.nii.gz"),
            y=pd.read_csv(
                DATA_PATH / f"{subject}_labels.csv", header=None
            ).values.flatten(),
        )

    masker = fit_masker(
        [dict_alignment[subject] for subject in subjects],
    )
    clustering_img = fetch_clustering_img(masker.mask_img_, n_parcels)

    return Dataset(
        name="Neuromod",
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
        target=target,
    )

