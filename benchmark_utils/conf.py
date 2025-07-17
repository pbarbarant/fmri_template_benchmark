from pathlib import Path

from joblib import Memory

MEMORY = Memory(Path(__file__).parent.parent / "memory_cache", verbose=0)
N_JOBS = 30

NSD_PATH = (
    "/data/parietal/store3/work/pbarbara/datasets/fmralign_benchopt_data/NSD"
)

IBC_PATH = "/data/parietal/store2/data/ibc/3mm"

IBC_SURF_PATH = "/data/parietal/store2/data/ibc/derivatives"
