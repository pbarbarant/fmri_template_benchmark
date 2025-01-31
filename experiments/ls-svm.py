# %%
import matplotlib.pyplot as plt
import pandas as pd
from fmralign.template_alignment import TemplateAlignment
from neo_ls_svm import NeoLSSVM
from nilearn import datasets
from nilearn.image import index_img
from nilearn.maskers import NiftiMasker

haxby_dataset = datasets.fetch_haxby(subjects=(1, 2))
mask = haxby_dataset.mask
masker = NiftiMasker(mask_img=mask, standardize=True).fit()


def load_subject_data(subject_id, haxby_dataset, masker):
    func_filename = haxby_dataset.func[subject_id]
    behavioral = pd.read_csv(haxby_dataset.session_target[subject_id], sep=" ")
    condition_mask = behavioral["labels"].isin(["face", "house"])
    func_img = index_img(func_filename, condition_mask)
    X = masker.transform(func_img)
    y = behavioral[condition_mask]["labels"].values
    return X, y


X1, y1 = load_subject_data(0, haxby_dataset, masker)
X2, y2 = load_subject_data(1, haxby_dataset, masker)

# Keep only the data where y1 == y2
mask = y1 == y2
X1 = X1[mask]
X2 = X2[mask]
y1 = y1[mask]
y = y1

# Align the second subject
algo = TemplateAlignment(
    n_pieces=500,
    alignment_method="scaled_orthogonal",
    clustering="ward",
    mask=masker,
    n_jobs=-1,
    verbose=11,
)

algo.fit([masker.inverse_transform(X1), masker.inverse_transform(X2)])

X1_template = masker.transform(
    algo.transform(masker.inverse_transform(X1), subject_index=0)
)
X2_template = masker.transform(
    algo.transform(masker.inverse_transform(X2), subject_index=1)
)

template_data = masker.transform(algo.template)

# %%
# Fit on the euclidean mean of the two subjects
clf = NeoLSSVM()
clf.fit(X1, y)
# %%
values_sub2_aligned = clf.decision_function(X2)

fig = plt.figure(figsize=(10, 5))
plt.hist(values_sub2_aligned[y == "face"], bins=30, alpha=0.5, label="face")
plt.hist(values_sub2_aligned[y == "house"], bins=30, alpha=0.5, label="house")
plt.title(f"Subject 2 unprojected, accuracy: {clf.score(X2, y)}")
plt.legend()

# Save before showing
fig.savefig(
    "/data/parietal/store3/work/pbarbara/fmri_template_benchmark/outputs/figures/unprojected.png",
    bbox_inches="tight",
)

fig.show()  # Show after saving

# %%
# Fit on the projected data
clf = NeoLSSVM()
clf.fit(X1_template, y)

values_sub2_aligned = clf.decision_function(X2_template)
fig = plt.figure(figsize=(10, 5))
plt.hist(values_sub2_aligned[y == "face"], bins=30, alpha=0.5, label="face")
plt.hist(values_sub2_aligned[y == "house"], bins=30, alpha=0.5, label="house")
plt.title(
    f"Subject 2 projected on template, accuracy: {clf.score(X2_template, y)}"
)
plt.legend()
fig.savefig(
    "/data/parietal/store3/work/pbarbara/fmri_template_benchmark/outputs/figures/projected.png",
    bbox_inches="tight",
)
fig.show()
