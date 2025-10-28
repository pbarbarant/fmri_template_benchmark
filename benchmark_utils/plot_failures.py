# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from joblib import load

sns.set_theme(
    context="paper",
    style="ticks",
    rc={
        "figure.figsize": [7, 6],
        "text.usetex": False,
        "font.family": "sans-serif",
        "savefig.dpi": 300,
    },
)
# Configuration
data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)


def get_results_dataframe(data_path: Path) -> pd.DataFrame:
    """Load and preprocess all decoding results."""
    results_paths = glob.glob(
        str(data_path / "**" / "decoding_results.pkl"), recursive=True
    )

    res_list = []
    for path in results_paths:
        path = Path(path)
        results = load(path)
        res_list.append(pd.DataFrame(results))

    df = pd.concat(res_list)

    # Clean dataset names
    df = df[~df["dataset_name"].str.contains("Simulated")]
    df["dataset_name"] = df["dataset_name"].str.replace(
        r"_[0-9]+$", "", regex=True
    )

    # Clean solver and task names
    df["solver_name"] = (
        df["solver_name"]
        .str.replace("ot", "Optimal Transport")
        .str.replace("SRM", "Shared Response")
        .str.replace("_", " ")
    )
    df["task_name"] = df["task_name"].str.replace("RSVPLanguage", "Language")

    # Handle Anatomical solver - only keep template_in_sample
    mask_anat = df["solver_name"] == "Anatomical"
    df_anat = df[mask_anat & (df["target"] == "template_in_sample")].copy()
    df_non_anat = df[~mask_anat].copy()

    # Create solver_target labels
    df_anat["solver_target"] = df_anat["solver_name"]
    df_non_anat["solver_target"] = np.select(
        [
            df_non_anat["target"] == "template_in_sample",
            df_non_anat["target"] == "template_out_of_sample",
        ],
        [
            df_non_anat["solver_name"],
            df_non_anat["solver_name"] + "\nOut of sample",
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


def average_folds(df: pd.DataFrame):
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = df.select_dtypes(include="number").columns.tolist()

    # Remove 'fold' from the grouping columns if it's categorical
    categorical_cols = [c for c in categorical_cols if c != "fold"]

    # Group by all categorical columns except 'fold' and average the numerical ones
    df = df.groupby(categorical_cols, as_index=False)[numerical_cols].mean()
    df = df.drop("fold", axis=1)
    return df.sort_values(["task_name", "solver_target"])


def make_failure_table(df: pd.DataFrame):
    """
    Generate a LaTeX table showing the number of misclassified subjects per task and solver,
    relative to the Anatomical solver.
    """

    # Average across folds and filter to the in-sample target
    df = average_folds(df)
    df = df[df.target == "template_in_sample"]

    # Get Anatomical scores per subject and task
    anat = df[df["solver_name"] == "Anatomical"][
        ["subject", "task_name", "cv_scores"]
    ].rename(columns={"cv_scores": "anat_score"})

    # Merge Anatomical scores back to all rows by subject and task
    df = df.merge(anat, on=["subject", "task_name"], how="left")

    # Mark a subject as misclassified if cv_scores < Anatomical score
    df["sub_miss"] = (df["cv_scores"] < df["anat_score"]).astype(int)
    df = df.drop(columns=["anat_score"])

    # Pivot table: rows = task_name, columns = solver_name, values = number of failures
    failure_table = df.pivot_table(
        index="task_name",
        columns="solver_name",
        values="sub_miss",
        aggfunc="sum",
        fill_value=0,  # ensure missing solver/task combinations show as 0
    )

    # Remove Anatomical column
    if "Anatomical" in failure_table.columns:
        failure_table = failure_table.drop(columns=["Anatomical"])

    # Add a "Total" row summing all tasks
    total_row = pd.DataFrame(failure_table.sum()).T
    total_row.index = ["Total"]
    failure_table = pd.concat([failure_table, total_row])

    # Ensure integer type
    failure_table = failure_table.astype(int)

    # Generate LaTeX table with centered columns
    latex_table = failure_table.to_latex(
        caption="Failure Cases",
        label="tab:failure_case",
        column_format="l" + "c" * (failure_table.shape[1]),
    )

    print(latex_table)


df = get_results_dataframe(data_path)
make_failure_table(df)
