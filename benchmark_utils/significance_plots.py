# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from joblib import load

plt.rcParams["figure.dpi"] = 500

data_path = Path(__file__).parent.parent / "copy_outputs"
figures_path = data_path.parent / "copy_outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)


def get_results_dataframe(
    data_path: Path, score="cv_scores_classif"
) -> pd.DataFrame:
    # Glob recursively all the decoding_results.pkl files
    results_paths = glob.glob(
        str(data_path / "**" / "decoding_results.pkl"), recursive=True
    )
    # Build a dataframe with all the results
    res_list = []
    for path in results_paths:
        path = Path(path)
        dataset = path.parent.parent.parent.name
        solver = path.parent.parent.name
        target = path.parent.name
        results = load(path)
        # Add the dataset name and solver name to the results
        results["data_name"] = dataset
        results["solver_name"] = solver
        results["target"] = target
        res_list.append(results)

    df = pd.DataFrame(res_list)

    # Remove the simulated data
    df = df[~df["data_name"].str.contains("Simulated")]

    # Remove the SparseOT solver
    df = df[~df["solver_name"].str.contains("Sparse")]

    # For anatomical keep only the template target
    df = df[
        ~((df["solver_name"] == "Anatomical") & (df["target"] == "template"))
    ]

    # Remove the "IBC " prefix on the dataset names
    df["data_name"] = df["data_name"].str.replace("IBC", "")

    # Add (template) to the solver name if target is template
    df["solver_name"] = df.apply(
        lambda x: x["solver_name"] + " (template)"
        if x["target"] == "template"
        else x["solver_name"] + " (pairwise)",
        axis=1,
    )

    # Remove the (template) suffix for the anatomical alignment
    df["solver_name"] = df["solver_name"].str.replace(
        "Anatomical (pairwise)", "Anatomical"
    )

    # Rename ot by Optimal Transport
    df["solver_name"] = df["solver_name"].str.replace(
        "ot", "OT"
    )

    # Rename Wm by WM
    df["data_name"] = df["data_name"].str.replace("Wm", "WM")
    
    # Remove the underscore in the dataset names
    df["data_name"] = df["data_name"].str.replace("_", "")

    # Sort alphabetically by dataset name and solver name
    df.sort_values(by=["data_name", "solver_name"], inplace=True)

    # Fix underscores in the solver names
    df["solver_name"] = df["solver_name"].str.replace("_", " ")

    # Expand the lists in df[score]
    df = df.explode(score)

    return df


df_acc = get_results_dataframe(data_path, score="cv_scores_classif")

# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)

from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

# Prepare a dataframe to store the p-values
pvals = []

# Iterate over each dataset
for dataset in df_acc['data_name'].unique():
    subset = df_acc[df_acc['data_name'] == dataset]
    solvers = subset['solver_name'].unique()
    
    # Pairwise comparison between solvers
    for i in range(len(solvers)):
        for j in range(i+1, len(solvers)):
            solver1 = solvers[i]
            solver2 = solvers[j]
            scores1 = subset[subset['solver_name'] == solver1]['cv_scores_classif'].to_numpy(np.float64)
            scores2 = subset[subset['solver_name'] == solver2]['cv_scores_classif'].to_numpy(np.float64)
            tstat, pval = ttest_ind(scores1, scores2)
            pvals.append({
                'data_name': dataset,
                'solver1': solver1,
                'solver2': solver2,
                'pval': pval
            })

pval_df = pd.DataFrame(pvals)


# First, get all unique datasets and solvers
datasets = pval_df['data_name'].unique()
all_solvers = pd.unique(pval_df[['solver1', 'solver2']].values.ravel())

# Dictionary to hold a matrix for each dataset
pval_matrices = {}

for dataset in datasets:
    matrix = pd.DataFrame(index=all_solvers, columns=all_solvers, dtype=float)
    
    # Fill diagonal with NaNs or 1.0 (no comparison needed)
    np.fill_diagonal(matrix.values, np.nan)
    
    df_subset = pval_df[pval_df['data_name'] == dataset]
    for _, row in df_subset.iterrows():
        s1, s2, pval = row['solver1'], row['solver2'], row['pval']
        matrix.loc[s1, s2] = pval
        matrix.loc[s2, s1] = pval  # symmetric
        
    pval_matrices[dataset] = matrix
    
    

def starify(p):
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

star_matrices = {}
for dataset, matrix in pval_matrices.items():
    stars = matrix.applymap(starify)
    star_matrices[dataset] = stars


# Choose which to plot: raw p-values or stars
plot_matrices = star_matrices  # or pval_matrices

# Set up the number of plots
num_datasets = len(plot_matrices)
ncols = 2
nrows = (num_datasets + ncols - 1) // ncols

# Mask the upper triangle (True = hide)
def get_lower_triangle_mask(df):
    mask = np.triu(np.ones_like(df, dtype=bool))
    return mask

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten()

for i, (dataset, matrix) in enumerate(plot_matrices.items()):
    ax = axes[i]
    mask = get_lower_triangle_mask(matrix)

    sns.heatmap(
        pval_matrices[dataset],  # underlying values for color
        annot=plot_matrices[dataset],  # what to show (stars or numbers)
        fmt="",  # don't format numbers
        cmap="coolwarm_r",
        mask=mask,
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        square=True
    )
    ax.set_title(f"P-value Matrix: {dataset}", fontsize=14)
    ax.set_xlabel("Solver")
    ax.set_ylabel("Solver")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
