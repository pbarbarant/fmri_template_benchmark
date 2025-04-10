# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from joblib import load
from nilearn import image
from scipy.ndimage import label
import numpy as np

plt.rcParams["figure.dpi"] = 300
# plt.style.use(["science", "nature", "no-latex"])

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

N_PARCELS = 400
DATASET = f"IBC_FaceBody_{N_PARCELS}"
METHODS = ["Anatomical", "Procrustes", "ot"]


def get_template_img(dataset, method):
    # Get the dataset folder path
    dataset_folder = data_path / dataset
    # Get the path to the template image
    template_img_path = dataset_folder / method / "template.nii.gz"
    return image.load_img(template_img_path)

def get_avg_weights_img(dataset, method):
    # Get the path to the weights folder
    method_path = data_path / dataset / method / "template"
    # Glob the weights files recursively
    weights_files = sorted(list(method_path.glob("*_weights.nii.gz")))
    # For each contrast, average the weights accross subjects
    data = None
    for weights_file in weights_files:
        weights_img = image.load_img(weights_file)
        if data is None:
            data = weights_img.get_fdata()
        else:
            data += weights_img.get_fdata()
    data /= len(weights_files)
    return image.new_img_like(weights_img, data)

def clusters_sizes(img, idx=0):
    # Get all the voxels clusters on the img
    data = img.get_fdata()
    
    # Threshold is the top 10% of the data
    threshold = np.percentile(data, 10)

    # Binarize if necessary (e.g., thresholding if not already binary)
    binary_data = data[..., idx] > threshold

    # Label connected components
    labeled_array, num_features = label(binary_data)

    # Get sizes of each cluster
    cluster_sizes = np.bincount(labeled_array.ravel())[1:]  # skip label 0 (background)
    return cluster_sizes


clusters_template_dict = {}
clusters_weights_dict = {}
for methods in METHODS:
    print(f"Processing {methods}...")
    template_img = get_template_img(DATASET, methods)
    weights_img = get_avg_weights_img(DATASET, methods)
    # Store the clusters sizes in a dictionary
    clusters_template_dict[methods] = clusters_sizes(template_img, idx=19)
    clusters_weights_dict[methods] = clusters_sizes(weights_img, idx=4)
# %%
fig, axs = plt.subplots(len(METHODS), 2, figsize=(4, 1.5 * len(METHODS)))
for i,method in enumerate(METHODS):
    # Plot template clusters
    sns.kdeplot(
        x=clusters_template_dict[method],
        ax=axs[i, 0],
        color="red",
    )
    axs[i, 0].set_title(method)
    axs[i, 0].set_xlabel("Cluster Size (voxels)")
    axs[i, 0].set_yticklabels([])
    axs[i, 0].set_xlim(0, 5000)

    # Plot weights clusters
    sns.kdeplot(
        x=clusters_weights_dict[method],
        ax=axs[i, 1],
        color="blue",
    )
    axs[i, 1].set_title(method)
    axs[i, 1].set_xlabel("Cluster Size (voxels)")
    axs[i, 1].set_yticklabels([])
    axs[i, 1].set_xlim(0, 5000)
    
# Adjust layout
plt.tight_layout()
plt.show()