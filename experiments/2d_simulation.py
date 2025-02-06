# %%
import numpy as np
import matplotlib.pyplot as plt
from nibabel.nifti1 import Nifti1Image
from nilearn.maskers import NiftiMasker


mean_1 = np.array([-1, 0])
mean_2 = np.array([1, 0])

data1 = np.random.randn(10, 2) + mean_1
data2 = np.random.randn(10, 2) + mean_2

data = np.concatenate([data1, data2], axis=0)
plt.scatter(data[:, 0], data[:, 1])
