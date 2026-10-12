from pathlib import Path

N_JOBS = 1

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Datasets paths
DATA_ROOT = PROJECT_ROOT / "data"
IBC_TRIALS_ROOT = DATA_ROOT / "ibc"
THINGS_CONDITIONS_DIR = DATA_ROOT / "things"
FORREST_CONDITIONS_DIR = DATA_ROOT / "forrest"
HCP_CONDITIONS_DIR = DATA_ROOT / "hcp"

# Path to gray matter mask
GM_MASK = DATA_ROOT / "gm_mask.nii.gz"

# Path to the ffx faces_adult zmap for the FaceBody task,
# used for plotting the surface maps in the paper.
IBC_FACEBODY_GROUP_ZMAP = (
    DATA_ROOT
    / "ibc"
    / "smooth_derivatives"
    / "group"
    / "FaceBody"
    / "ffx_faces_adult.nii.gz",
)
