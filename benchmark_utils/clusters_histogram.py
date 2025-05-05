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
plt.style.use(["science", "nature", "no-latex"])

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)

N_PARCELS = 400
DATASET = f"IBC_FaceBody_{N_PARCELS}"
METHODS = ["Anatomical", "Procrustes", "ot"]


def get_template_img(dataset, method):
    dataset_folder = data_path / dataset
    template_img_path = dataset_folder / method / "template.nii.gz"
    return image.load_img(template_img_path)


def get_avg_weights_img(dataset, method):
    method_path = data_path / dataset / method / "template"
    weights_files = sorted(list(method_path.glob("*_weights.nii.gz")))
    data = None
    for weights_file in weights_files:
        weights_img = image.load_img(weights_file)
        if data is None:
            data = weights_img.get_fdata()
        else:
            data += weights_img.get_fdata()
    data /= len(weights_files)
    return image.new_img_like(weights_img, data)


def clusters_sizes(img, idx=0, threshold=0):
    data = img.get_fdata()
    binary_data = data[..., idx] > threshold
    labeled_array, _ = label(binary_data)
    cluster_sizes = np.bincount(labeled_array.ravel())[1:]
    return cluster_sizes


def get_threshold(imgs):
    # Get the threshold for each image
    data = np.concatenate([img.get_fdata() for img in imgs], axis=-1)
    # Return the quantile while discarding zeros
    return np.quantile(data[np.abs(data) > 0], 0.80)


threshold_activations = get_threshold(
    [get_template_img(DATASET, method) for method in METHODS]
)

threshold_weights = get_threshold(
    [get_avg_weights_img(DATASET, method) for method in METHODS]
)
print(f"Threshold activations: {threshold_activations}")
print(f"Threshold weights: {threshold_weights}")

clusters_template_dict = {}
clusters_weights_dict = {}
data_template_dict = {}
data_weights_dict = {}
for methods in METHODS:
    print(f"Processing {methods}...")
    template_img = get_template_img(DATASET, methods)
    weights_img = get_avg_weights_img(DATASET, methods)

    # Store the data in a dictionary
    template_data = template_img.get_fdata()
    weights_data = weights_img.get_fdata()
    data_template_dict[methods] = template_data[
        np.abs(template_data) > 0
    ].flatten()
    data_weights_dict[methods] = weights_data[
        np.abs(weights_data) > 0
    ].flatten()

    # Store the clusters sizes in a dictionary
    clusters_template_dict[methods] = clusters_sizes(
        template_img, idx=19, threshold=threshold_activations
    )
    clusters_weights_dict[methods] = clusters_sizes(
        weights_img, idx=4, threshold=threshold_weights
    )


# Convert dictionaries to DataFrames
clusters_template_df = pd.DataFrame.from_dict(
    clusters_template_dict, orient="index"
).T
clusters_weights_df = pd.DataFrame.from_dict(
    clusters_weights_dict, orient="index"
).T

original_palette = sns.color_palette("Paired")
shifted_palette = [
    original_palette[1],
    original_palette[5],
    original_palette[3],
]

# Plot figure
fig, axs = plt.subplots(2, 2, figsize=(3, 2.5))

legend_handles = []

# Plot activation scores and collect handles
for i, method in enumerate(METHODS):
    line1 = sns.kdeplot(
        x=data_template_dict[method],
        label=method,
        color=shifted_palette[i],
        ax=axs[0, 0],
    )
    sns.kdeplot(
        x=data_weights_dict[method],
        label=method,
        color=shifted_palette[i],
        ax=axs[0, 1],
    )
    # Save the line handle for the legend
    legend_handles.append(line1.lines[0])

# Set axis labels
axs[0, 0].set_yticklabels([])
axs[0, 0].set_xlabel("Activation t-values")
axs[0, 0].set_ylabel("")
axs[0, 1].set_yticklabels([])
axs[0, 1].set_xlabel("Weights")
axs[0, 1].set_ylabel("")

# KDE plots for clusters
sns.kdeplot(
    data=clusters_template_df,
    palette=shifted_palette,
    ax=axs[1, 0],
    legend=False,
)
axs[1, 0].set_yticklabels([])
axs[1, 0].set_ylabel("")
axs[1, 0].set_xlim(0, 50)
axs[1, 0].set_xlabel("Activation Cluster Size (voxels)")

sns.kdeplot(
    data=clusters_weights_df,
    palette=shifted_palette,
    ax=axs[1, 1],
    legend=False,
)
axs[1, 1].set_yticklabels([])
axs[1, 1].set_ylabel("")
axs[1, 1].set_xlim(0, 50)
axs[1, 1].set_xlabel("Weights Cluster Size (voxels)")

# Add the legend to the top of the figure
handles, labels = axs[0, 0].get_legend_handles_labels()
labels[2] = "Optimal Transport"
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(METHODS),
    frameon=False,
    bbox_to_anchor=(0.5, 1.02),
)

# Tight layout with room for legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(
    figures_path / "clusters_density.pdf", bbox_inches="tight", dpi=300
)
plt.show()
plt.show()
