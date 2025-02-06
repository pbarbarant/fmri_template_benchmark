# %%
import numpy as np
import matplotlib.pyplot as plt
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import MultiNiftiMasker


# Generate a gaussian mixture for sub-01
mean_1 = np.array([20, 0])
mean_2 = np.array([-20, 0])
data1 = np.random.randn(100, 2) + mean_1
data2 = np.random.randn(100, 2) + mean_2
data_sub1 = np.concatenate([data1, data2], axis=0)

# Generate a gaussian mixture for sub-02
mean_3 = np.array([0, 20])
mean_4 = np.array([0, -20])
data3 = np.random.randn(100, 2) + mean_3
data4 = np.random.randn(100, 2) + mean_4
data_sub2 = np.concatenate([data3, data4], axis=0)

# Generate the labels
y = np.arange(200) >= 100
plt.scatter(data_sub1[:, 0], data_sub1[:, 1], c=y)
plt.scatter(data_sub2[:, 0], data_sub2[:, 1], c=y)

# Compute the transport cost between voxels
from scipy.spatial.distance import cdist
import ot

M = cdist(data_sub1.T, data_sub2.T)

# Compute the optimal transport plan
a = np.ones(2) / 2
b = np.ones(2) / 2

# Solve the OT problem
P = ot.sinkhorn(a, b, M, reg=0.1)
P
# %%
# Pushforward the data
data_sub1_transported = data_sub1 @ P * P.shape[0]

# Plot the data
plt.figure()
plt.scatter(
    data_sub1_transported[:, 0], data_sub1_transported[:, 1], label="sub1"
)
plt.scatter(data_sub2[:, 0], data_sub2[:, 1], label="sub2")
plt.legend()
plt.show()
# %%

# %%

# Reshape the data to (2, 1, 1, 200)
data_sub1_reshaped = data_sub1.T.reshape(2, 1, 1, 200)
data_sub2_reshaped = data_sub2.T.reshape(2, 1, 1, 200)

# Create NIfTI images
nifti_img1 = Nifti1Image(data_sub1_reshaped, affine=np.eye(4))
nifti_img2 = Nifti1Image(data_sub2_reshaped, affine=np.eye(4))

# Generate mask of all 1s
mask_data = np.ones((2, 1, 1))
mask_img = Nifti1Image(mask_data, affine=np.eye(4))

# Define and fit the masker
masker = MultiNiftiMasker(mask_img=mask_img, standardize=False).fit(
    [nifti_img1, nifti_img2]
)

# Transform the NIfTI image
transformed_data = masker.transform(nifti_img1)
print("Transformed data shape:", transformed_data.shape)
