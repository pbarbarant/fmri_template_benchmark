from nilearn import image
from nilearn.datasets import fetch_atlas_schaefer_2018


def fetch_clustering_img(masker, n_rois=400, resolution_mm=2):
    clustering_img = (
        masker.transform(
            fetch_atlas_schaefer_2018(
                n_rois=n_rois,
                resolution_mm=resolution_mm,
            )["maps"]
        )
    ).astype(int)
    return image.index_img(masker.inverse_transform(clustering_img), 0)