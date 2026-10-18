import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from tqdm import tqdm

DATA_PATH = Path(__file__).parent.parent / "outputs"
FIGURES_PATH = DATA_PATH.parent / "outputs" / "figures"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def get_results_dataframe(
    data_path: Path, n_parcels: int | None = None
) -> pd.DataFrame:
    """Load and preprocess all decoding results."""
    results_paths = glob.glob(
        str(data_path / "**" / "decoding_results.pkl"), recursive=True
    )

    res_list = []
    for path in tqdm(results_paths):
        path = Path(path)
        results = load(path)
        res_list.append(pd.DataFrame(results))

    df = pd.concat(res_list)

    # Extract n_parcels into a new column
    df["n_parcels"] = df["dataset_name"].str.extract(r"_(\d+)$").astype("Int64")

    if n_parcels is not None:
        df = df[df["n_parcels"] == n_parcels]

    # Clean dataset names
    df = df[~df["dataset_name"].str.startswith(("Simulated", "HCP"))]
    df["dataset_name"] = df["dataset_name"].str.replace(
        r"_[0-9]+$", "", regex=True
    )

    # Drop any SRM apart from SRM_20
    df = df[
        ~(
            (df["solver_name"].str.startswith("SRM"))
            & (df["solver_name"] != "SRM_20")
        )
    ]

    # Clean solver and task names
    df["solver_name"] = (
        df["solver_name"]
        .str.replace("ot", "Optimal Transport")
        .str.replace("SRM_20", "Shared Response")
        .str.replace("_", " ")
    )
    df["task_name"] = df["task_name"].str.replace("RSVPLanguage", "Language")
    df["task_name"] = df["task_name"].str.replace("Forrest", "StudyForrest")

    # Handle Anatomical solver - only keep template_out_of_sample
    mask_anat = df["solver_name"] == "Anatomical"
    df_anat = df[mask_anat & (df["target"] == "template_out_of_sample")].copy()
    df_non_anat = df[~mask_anat].copy()

    # Create solver_target labels
    df_anat["solver_target"] = df_anat["solver_name"]
    df_non_anat["solver_target"] = np.select(
        [
            df_non_anat["target"] == "template_in_sample",
            df_non_anat["target"] == "template_out_of_sample",
        ],
        [
            df_non_anat["solver_name"] + "\nIn-Sample",
            df_non_anat["solver_name"] + "\nOut-of-Sample",
        ],
        default=df_non_anat["solver_name"] + "\nPairwise",
    )

    df = pd.concat([df_anat, df_non_anat])

    # Add subject counts to task names
    anat_counts = df.groupby("task_name")["subject"].nunique().to_dict()
    df["task_name"] = df["task_name"].apply(
        lambda x: f"{x} (N={anat_counts.get(x, 0)})"
    )

    return df.sort_values(["task_name", "solver_target"])


# Shades per solver, from darkest to lightest:
# (Out-of-Sample, In-Sample, Pairwise)
SOLVER_SHADES = {
    "Anatomical": ("#3182bd",),
    "Optimal Transport": ("#e6550d", "#fd8d3c", "#fdae6b"),
    "Procrustes": ("#31a354", "#74c476", "#a1d99b"),
    "Ridge": ("#756bb1", "#9e9ac8", "#bcbddc"),
    "Shared Response": ("#d4ac0d", "#f1c40f", "#f7dc6f"),
}
TARGET_SHADE = {"Out-of-Sample": 0, "In-Sample": 1, "Pairwise": 2}


def create_palette(df: pd.DataFrame) -> dict:
    """Create color palette for solvers."""
    fallback = iter(sns.color_palette("tab10"))
    palette = {}
    for key in sorted(df.solver_target.unique()):
        solver, _, target = key.partition("\n")
        if solver not in SOLVER_SHADES:
            SOLVER_SHADES[solver] = (next(fallback),) * 3
        shades = SOLVER_SHADES[solver]
        palette[key] = shades[min(TARGET_SHADE.get(target, 0), len(shades) - 1)]
        # Bare solver name uses the Out-of-Sample color
        palette[solver] = shades[0]

    return palette
