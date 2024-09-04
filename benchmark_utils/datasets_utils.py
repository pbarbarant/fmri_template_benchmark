import joblib
import pandas as pd
from nilearn import masking, maskers, surface
from nilearn.experimental.surface._datasets import load_fsaverage
from nilearn.experimental.surface._surface_image import SurfaceImage


def load_dataset(subject, data_path, mask):
    data_alignment = mask.inverse_transform(
        joblib.load(data_path / "alignment" / f"{subject}.pkl")
    )
    data_decoding = mask.inverse_transform(
        joblib.load(data_path / "decoding" / f"{subject}.pkl")
    )
    labels_decoding = pd.read_csv(
        data_path / "decoding" / f"{subject}_labels.csv",
        header=None,
    ).values.ravel()

    return data_alignment, data_decoding, labels_decoding


def load_mask(data_path, memory):
    masker_path = data_path / "masks" / "mask.nii.gz"
    connected_mask = masking.compute_background_mask(
        masker_path, connected=True
    )
    mask = maskers.NiftiMasker(connected_mask, memory=memory).fit()
    return mask


def project_on_surf(data, mesh_name="fsaverage3"):
    mesh = load_fsaverage(mesh_name)["pial"]
    left_data = surface.vol_to_surf(data, mesh.parts["left"]).T
    right_data = surface.vol_to_surf(data, mesh.parts["right"]).T
    return SurfaceImage(
        mesh=mesh,
        data={
            "left": left_data,
            "right": right_data,
        },
    )


def load_dataset_surf(subject, data_path, mask, mesh_name):
    data_alignment, data_decoding, labels_decoding = load_dataset(
        subject, data_path, mask
    )
    data_alignment_surf = project_on_surf(data_alignment, mesh_name)
    data_decoding_surf = project_on_surf(data_decoding, mesh_name)
    return data_alignment_surf, data_decoding_surf, labels_decoding
