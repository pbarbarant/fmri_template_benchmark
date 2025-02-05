from pathlib import Path

from joblib import Memory

MEMORY = Memory(Path(__file__).parent.parent / "memory_cache", verbose=0)
N_JOBS = 30

FORREST_PATH = "/data/parietal/store2/work/tbazeill/forrest/derivatives/"

NSD_PATH = (
    "/data/parietal/store3/work/pbarbara/datasets/fmralign_benchopt_data/NSD"
)

BUDAPEST_PATH = "/data/parietal/store3/data/budapest"

RAIDERS_PATH = "/data/parietal/store3/data/raiders_haxby_full/labs/haxby/raiders-fmriprep/"

NEUROMOD_PATH = "/data/parietal/store2/work/tbazeill/neuromod/3mm/"

HCP_PATH = "/data/parietal/store/data/HCP900/glm/"

IBC_PATH = "/data/parietal/store2/data/ibc/3mm"

IBC_SURF_PATH = "/data/parietal/store2/data/ibc/derivatives"
