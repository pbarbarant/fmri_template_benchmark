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
    """Count subjects performing worse than anatomical alignment."""
    df = average_folds(df)
    df = df.loc[df["target"] == "template_out_of_sample"].copy()

    # Adjust these aliases if your solver names differ.
    df["solver_name"] = df["solver_name"].replace(
        {
            "OT": "Optimal Transport",
            "OptimalTransport": "Optimal Transport",
            "SRM": "Shared Response",
            "SharedResponse": "Shared Response",
        }
    )
    methods = [
        "Optimal Transport",
        "Procrustes",
        "Ridge",
        "Shared Response",
    ]

    # One mean score per subject, task, and method.
    scores = df.pivot(
        index=["task_name", "subject"],
        columns="solver_name",
        values="cv_scores",
    )

    required = ["Anatomical", *methods]
    missing = [method for method in required if method not in scores.columns]
    if missing:
        raise ValueError(f"Missing solvers: {missing}")

    # Missing scores must not silently count as zero failures.
    if scores[required].isna().any().any():
        raise ValueError(
            "Some subject/task combinations have missing solver scores."
        )

    failures = scores[methods].lt(scores["Anatomical"], axis=0)
    table = failures.groupby(level="task_name", observed=True).sum().astype(int)
    table.insert(
        0,
        "N Subjects",
        scores.groupby(level="task_name", observed=True).size(),
    )

    # task_order = [
    #     "ArchiSpatial",
    #     "ArchiStandard",
    #     "FaceBody",
    #     "Forrest",
    #     "HcpWm",
    #     "Language",
    #     "THINGS",
    # ]
    # ordered_tasks = [task for task in task_order if task in table.index]
    # ordered_tasks += [task for task in table.index if task not in task_order]
    # table = table.loc[ordered_tasks]

    table.loc["Total"] = table.sum()

    latex_table = table.to_latex(
        escape=False,
        column_format="lccccc",
        formatters={
            "N Subjects": lambda value: str(int(value)),
        },
    )

    # Insert a horizontal rule before the total.
    lines = latex_table.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("Total"):
            lines.insert(i, r"\midrule")
            break

    latex_table = "\n".join(
        [
            r"\begin{table}",
            r"\centering",
            r"\caption{Number of subjects with lower decoding accuracy "
            r"than anatomical alignment.}",
            r"\label{tab:failure_case}",
            r"\vskip 0.12in",
            "\n".join(lines),
            r"\end{table}",
        ]
    )
    print(latex_table)
    return table


df = get_results_dataframe(DATA_PATH, n_parcels=400)
failure_table = make_failure_table(df)
