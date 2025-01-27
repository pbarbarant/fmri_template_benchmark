# %%
import numpy as np
import matplotlib.pyplot as plt
from nibabel.nifti1 import Nifti1Image
from fmralign.template_alignment import TemplateAlignment
from nilearn.maskers import NiftiMasker

PATCH_SIZE = (32, 32)
RADIUS = 3
AFFINE = np.eye(4)


def generate_one_patch(center=(16, 16), radius=RADIUS):
    patch = np.zeros(PATCH_SIZE)
    # Generate a gaussian centered at the center
    for i in range(PATCH_SIZE[0]):
        for j in range(PATCH_SIZE[1]):
            patch[i, j] = np.exp(
                -((i - center[0]) ** 2 + (j - center[1]) ** 2)
                / (2 * radius**2)
            )
    return patch


def generate_niftis():
    patches = []
    for i in [-2, 0, 2]:
        for j in [-2, 0, 2]:
            center = (16, 16)
            patch = generate_one_patch(center, RADIUS + (i - j))
            # Convert to nifti of shape (64, 64, 1)
            patch = patch[..., np.newaxis]
            patches.append(Nifti1Image(patch, AFFINE))
    return patches


def simulate_template_alignment(alignment_method="Procrustes"):
    patches = generate_niftis()
    # Remove the center patch
    patches.pop(4)
    mask_img = Nifti1Image(np.ones((*PATCH_SIZE, 1)), AFFINE)
    masker = NiftiMasker(mask_img=mask_img).fit()
    # Compute the template
    algo = TemplateAlignment(
        alignment_method=alignment_method,
        mask=masker,
    )
    algo.fit(patches)
    template = algo.template
    # Insert the template in the patches
    patches.insert(4, template)
    # Add the template to the patches
    patches.append(template)
    return patches


patches = simulate_template_alignment("optimal_transport")

fig, axs = plt.subplots(3, 3, figsize=(12, 12))
# Remove axis
for i in range(3):
    for j in range(3):
        axs[i, j].imshow(
            patches[i * 3 + j].get_fdata()[:, :, 0, 0], cmap="magma"
        )
        axs[i, j].axis("off")
plt.show()
