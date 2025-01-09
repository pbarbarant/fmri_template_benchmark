import numpy as np
import glob
import pandas as pd
from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018, load_mni152_brain_mask
from nilearn.maskers import NiftiMasker
from ibc_public import utils_data
from tqdm import tqdm
from typing import List, Dict, Optional
from dataclasses import dataclass
from nibabel import Nifti1Image


from joblib import Memory
from pathlib import Path

memory = Memory(
    "/data/parietal/store3/work/pbarbara/fmri_template_benchmark/nilearn_cache",
    verbose=0,
)


@dataclass
class LabeledImage:
    img: Nifti1Image
    y: Optional[np.ndarray]


@dataclass
class Dataset:
    name: str
    subjects: List[str]
    dict_alignment: Dict[str, Nifti1Image]
    dict_decoding: Dict[str, LabeledImage]
    masker: NiftiMasker
    clustering_img: Nifti1Image
    dict_aligned: Optional[Dict[str, LabeledImage]] = None
    template: Optional[LabeledImage] = None


def check_init_dataset(dataset: Dataset) -> None:
    """Check that the dataset is correctly initialized."""
    first_subject = list(dataset.dict_alignment.keys())[0]
    labels = np.unique(dataset.dict_decoding[first_subject].y)
    n_samples_alignment = dataset.dict_alignment[first_subject].shape[-1]
    n_samples_decoding = dataset.dict_decoding[first_subject].img.shape[-1]

    assert (
        dataset.template is None
    ), "Template should be None at initialization"
    assert (
        dataset.dict_aligned is None
    ), "dict_aligned should be None at initialization"

    for subject in dataset.dict_alignment:
        assert (
            dataset.dict_alignment[subject].shape[-1] == n_samples_alignment
        ), "Inconsistent number of samples in alignment"
        assert (
            dataset.dict_decoding[subject].img.shape[-1] == n_samples_decoding
        ), "Inconsistent number of samples in decoding"
        assert (
            dataset.dict_alignment[subject].shape[:-1]
            == dataset.masker.mask_img_.shape
        ), "Alignment image shape does not match mask shape"
        assert (
            dataset.dict_decoding[subject].img.shape[:-1]
            == dataset.masker.mask_img_.shape
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
        f.write(
            f"Number of parcels: {len(np.unique(dataset.clustering_img.get_fdata())) - 1}\n"
        )
        f.write(
            f"List of conditions: {np.unique(dataset.dict_decoding[first_subject].y)}\n"
        )
        f.write(
            f"Number of alignment samples: {dataset.dict_alignment[first_subject].shape[-1]}\n"
        )
        f.write(
            f"Number of decoding samples: {dataset.dict_decoding[first_subject].img.shape[-1]}\n"
        )


def fetch_clustering_img(masker, n_rois=400):
    clustering_img = fetch_atlas_schaefer_2018(n_rois=n_rois)["maps"]
    resampled_img = image.resample_to_img(clustering_img, masker.mask_img_)
    int_img = masker.inverse_transform(
        masker.transform(resampled_img).astype(int)
    )
    return image.index_img(int_img, 0)


def _sample_labels(n_samples):
    return np.arange(3).reshape(1, -1).repeat(n_samples // 3, axis=0).flatten()


def _sample_labeled_image(n_samples, masker):
    mask_img = masker.mask_img_
    n_voxels = masker.transform(mask_img).shape[1]
    data = np.random.randn(n_samples, n_voxels)
    img = masker.inverse_transform(data)
    y = _sample_labels(n_samples)
    return LabeledImage(
        img=img,
        y=y,
    )


def sample_dataset(
    name,
    masker,
    clustering_img,
    subjects,
    n_samples_alignement,
    n_samples_decoding,
):
    print(f"Generating fold {name}")
    dict_alignment = dict()
    dict_decoding = dict()
    for subject in subjects:
        # Generate random surface images for each subject.
        dict_alignment[subject] = _sample_labeled_image(
            n_samples_alignement, masker
        ).img
        dict_decoding[subject] = _sample_labeled_image(
            n_samples_decoding, masker
        )

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
    )


def fit_mni152_masker(resolution=3):
    mask_img = load_mni152_brain_mask(resolution=resolution)
    return NiftiMasker(
        mask_img=mask_img, memory="nilearn_cache", memory_level=1
    ).fit()


def fit_masker_to_data(img):
    return NiftiMasker(memory="nilearn_cache", memory_level=1).fit(img)


@memory.cache
def fetch_ibc(
    name="IBC",
    subjects=None,
    task=None,
    n_parcels=400,
):
    DERIVATIVES = "/data/parietal/store2/data/ibc/3mm"
    df = utils_data.make_vol_db(
        derivatives=DERIVATIVES,
        subject_list=subjects,
        task_list=[task],
        acquisition="all",
    )
    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing IBC data"):
        alignment_df = df[
            (df.subject == subject) & (df.path.str.contains("ffx"))
        ]
        decoding_df = df[
            (df.subject == subject) & ~(df.path.str.contains("ffx"))
        ]
        dict_alignment[subject] = image.concat_imgs(
            alignment_df.path.to_list()
        )
        dict_decoding[subject] = LabeledImage(
            img=image.concat_imgs(decoding_df.path.to_list()),
            y=decoding_df.contrast.to_numpy(),
        )

    masker = fit_masker_to_data(dict_alignment[subjects[0]])
    clustering_img = fetch_clustering_img(masker, n_rois=n_parcels)

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
    )


def make_hcp_db(derivatives, subject_list, task):
    """Returns a dataframe listing HCP data."""
    paths = []
    contrasts = []
    subjects = []
    for subject in tqdm(subject_list):
        zmaps_path = Path(derivatives) / subject / task / "level2/z_maps"
        if not zmaps_path.exists():
            raise FileNotFoundError(f"Path {zmaps_path} does not exist.")
        zmaps = glob.glob(str(zmaps_path / "*.nii.gz"))
        for path in zmaps:
            contrast = Path(path).stem.removeprefix("z_").removesuffix(".nii")
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


@memory.cache
def fetch_hcp(
    name="HCP",
    subjects=None,
    task=None,
    n_parcels=400,
):
    DERIVATIVES = "/data/parietal/store/data/HCP900/glm/"
    df = make_hcp_db(
        derivatives=DERIVATIVES,
        subject_list=subjects,
        task=task,
    )

    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing HCP data"):
        alignment_df = df[(df.subject == subject)]
        # Decoding data is the same as alignment data for HCP
        # since there is only one contrast per subject.
        decoding_df = df[(df.subject == subject)]
        dict_alignment[subject] = image.concat_imgs(
            alignment_df.path.to_list()
        )
        dict_decoding[subject] = LabeledImage(
            img=image.concat_imgs(decoding_df.path.to_list()),
            y=decoding_df.contrast.to_numpy(),
        )

    masker = fit_masker_to_data(dict_alignment[subjects[0]])
    clustering_img = fetch_clustering_img(masker, n_rois=n_parcels)

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
    )


@memory.cache
def fetch_forrest(
    name="Forrest",
    subjects=None,
    n_parcels=400,
):
    DATA_PATH = Path(
        "/data/parietal/store2/work/tbazeill/forrest/derivatives/"
    )
    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing Forrest data"):
        dict_alignment[subject] = image.load_img(
            DATA_PATH / f"forrest_{subject}.nii.gz"
        )
        dict_decoding[subject] = LabeledImage(
            img=image.load_img(DATA_PATH / f"{subject}.nii.gz"),
            y=pd.read_csv(
                DATA_PATH / f"{subject}_labels.csv", header=None
            ).to_numpy(),
        )

    masker = fit_masker_to_data(dict_alignment[subjects[0]])
    clustering_img = fetch_clustering_img(masker, n_rois=n_parcels)

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
    )


@memory.cache
def fetch_nsd(
    name="NSD",
    subjects=None,
    n_parcels=400,
):
    DATA_PATH = Path(
        "/data/parietal/store3/work/pbarbara/datasets/fmralign_benchopt_data/NSD"
    )

    dict_alignment = dict()
    dict_decoding = dict()
    for subject in tqdm(subjects, desc="Processing NSD data"):
        dict_alignment[subject] = image.load_img(
            DATA_PATH / f"alignment/{subject}.nii.gz"
        )
        dict_decoding[subject] = LabeledImage(
            img=image.load_img(DATA_PATH / f"decoding/{subject}.nii.gz"),
            y=pd.read_csv(
                DATA_PATH / f"decoding/{subject}_labels.csv", header=None
            ).values.flatten(),
        )

    masker = fit_masker_to_data(dict_alignment[subjects[0]])
    clustering_img = fetch_clustering_img(masker, n_rois=n_parcels)

    return Dataset(
        name=name,
        subjects=subjects,
        dict_alignment=dict_alignment,
        dict_decoding=dict_decoding,
        masker=masker,
        clustering_img=clustering_img,
    )
