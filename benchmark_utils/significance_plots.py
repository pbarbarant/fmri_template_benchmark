# %%
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import seaborn as sns
from joblib import load
import pandas as pd
import numpy as np
from scipy.stats import t

plt.rcParams["figure.dpi"] = 100

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
N_PARCELS = 400

def get_results_dataframe(
    data_path: Path, score:str="cv_scores_classif", n_parcels:int=400
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
    
    # Keep only the results for the specified number of parcels
    df = df[df["data_name"].str.contains(f"{n_parcels}")]

    # Remove parcels numbers from the dataset names
    df["data_name"] = df["data_name"].str.replace(
        r"_[0-9]+$", "", regex=True
    )

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


df_acc = get_results_dataframe(data_path, score="cv_scores_classif", n_parcels=N_PARCELS)


# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)

def corrected_dependent_ttest(data1, data2):
    n = len(data1)
    n_test_folds = 1
    n_training_folds = len(data1) - n_test_folds
    differences = data1 - data2
    sd = np.std(differences)
    divisor = 1 / n * np.sum(differences)
    test_training_ratio = n_test_folds / n_training_folds  
    denominator = np.sqrt(1 / n + test_training_ratio) * sd
    t_stat = divisor / denominator
    # degrees of freedom
    df = n - 1
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    return t_stat, p

def get_p_values(df, method):
    # Prepare a dataframe to store the p-values
    pvals = []

    # Iterate over each dataset
    for dataset in df['data_name'].unique():
        subset = df[df['data_name'] == dataset]
        solvers = subset['solver_name'].unique()
        
        # Pairwise comparison between solvers
        for i in range(len(solvers)):
            for j in range(i+1, len(solvers)):
                solver1 = solvers[i]
                solver2 = solvers[j]
                scores1 = subset[subset['solver_name'] == solver1]['cv_scores_classif'].to_numpy(np.float64)
                scores2 = subset[subset['solver_name'] == solver2]['cv_scores_classif'].to_numpy(np.float64)
                if len(scores1) == len(scores2):
                    tstat, pval = method(scores1, scores2)
                    pvals.append({
                        'data_name': dataset,
                        'solver1': solver1,
                        'solver2': solver2,
                        'pval': pval
                    })
                else:
                    print(f"Warning: Different number of scores for {solver1} and {solver2} in dataset {dataset}. Skipping comparison.")
                    continue

    pval_df = pd.DataFrame(pvals)

    return pval_df


def get_accuracies_diff(df):
    # Prepare a dataframe to store the p-values
    acc_diff = []

    # Iterate over each dataset
    for dataset in df['data_name'].unique():
        subset = df[df['data_name'] == dataset]
        solvers = subset['solver_name'].unique()
        
        # Pairwise comparison between solvers
        for i in range(len(solvers)):
            for j in range(i+1,len(solvers)):
                solver1 = solvers[i]
                solver2 = solvers[j]
                scores1 = subset[subset['solver_name'] == solver1]['cv_scores_classif'].to_numpy(np.float64)
                scores2 = subset[subset['solver_name'] == solver2]['cv_scores_classif'].to_numpy(np.float64)
                if len(scores1) == len(scores2):
                    acc_diff.append({
                        'data_name': dataset,
                        'solver1': solver1,
                        'solver2': solver2,
                        'diff': np.mean(scores1) - np.mean(scores2)
                    })
                else:
                    print(f"Warning: Different number of scores for {solver1} and {solver2} in dataset {dataset}. Skipping comparison.")
                    continue

    acc_diff_df = pd.DataFrame(acc_diff)

    return acc_diff_df

# Get p-values using Wilcoxon test
pval_df = get_p_values(df_acc, corrected_dependent_ttest)

# Get accuracies differences
df_acc = get_accuracies_diff(df_acc)

# First, get all unique datasets and solvers
datasets = pval_df['data_name'].unique()
all_solvers = pd.unique(pval_df[['solver1', 'solver2']].values.ravel())

# Dictionary to hold a matrix for each dataset
acc_diff_matrices = {}
# Dictionary to hold p-value matrices for each dataset
p_value_matrices = {}
for dataset in datasets:
    acc_matrix = pd.DataFrame(index=all_solvers, columns=all_solvers, dtype=float)
    p_value_matrix = pd.DataFrame(index=all_solvers, columns=all_solvers, dtype=float)
    
    # Fill diagonal with NaNs or 1.0 (no comparison needed)
    np.fill_diagonal(acc_matrix.values, np.nan)
    np.fill_diagonal(p_value_matrix.values, np.nan)
    
    df_subset = df_acc[df_acc['data_name'] == dataset]
    for _, row in df_subset.iterrows():
        s1, s2, diff = row['solver1'], row['solver2'], row['diff']
        acc_matrix.loc[s1, s2] = diff
        acc_matrix.loc[s2, s1] = -diff  # symmetric
        
    # Fill the p-value matrix
    pval_subset = pval_df[pval_df['data_name'] == dataset]
    for _, row in pval_subset.iterrows():
        s1, s2, pval = row['solver1'], row['solver2'], row['pval']
        p_value_matrix.loc[s1, s2] = pval
        p_value_matrix.loc[s2, s1] = pval  # symmetric
        
    p_value_matrices[dataset] = p_value_matrix
    acc_diff_matrices[dataset] = acc_matrix
    
    

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
for dataset, pval_matrix in p_value_matrices.items():
    stars = pval_matrix.map(starify)
    star_matrices[dataset] = stars


# Choose which to plot: raw p-values or stars
plot_matrices = star_matrices  # or acc_diff_matrices

# Set up the number of plots
num_datasets = len(plot_matrices)
ncols = 2
nrows = (num_datasets + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
axes = axes.flatten()

for i, (dataset, acc_matrix) in enumerate(plot_matrices.items()):
    ax = axes[i]

    sns.heatmap(
        acc_diff_matrices[dataset],  # underlying values for color
        annot=plot_matrices[dataset],  # what to show (stars or numbers)
        fmt="",  # don't format numbers
        cmap="bwr",
        cbar=True,
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        square=True,
        annot_kws={"fontsize":12},
        vmin=-0.25,
        vmax=0.25,
    )
    ax.set_title(f"Dataset: {dataset}", fontsize=14)
    ax.set_xlabel("Solver")
    ax.set_ylabel("Solver")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig(figures_path / f"significance_plots_{N_PARCELS}.pdf", dpi=100, bbox_inches="tight")
