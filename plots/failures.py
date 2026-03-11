# %%
import pandas as pd
import seaborn as sns
from utils import DATA_PATH, get_results_dataframe

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
    df = df[df.target == "template_out_of_sample"]

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


df = get_results_dataframe(DATA_PATH, n_parcels=400)
make_failure_table(df)
