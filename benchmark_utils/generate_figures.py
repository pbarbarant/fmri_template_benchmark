# %%
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from PyPDF2 import PdfMerger

plt.rcParams["figure.dpi"] = 500

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
# Parse the latest file
file_list = glob.glob(os.path.join(data_path, "*.parquet"))
latest_file = max(file_list, key=os.path.getmtime)
df = pd.read_parquet(latest_file)

# Remove the simulated data
df.drop(df[df["data_name"].str.contains("Simulated")].index, inplace=True)

# Merge all BOLD5000 folds into one
df.loc[df["data_name"].str.contains("BOLD5000"), "data_name"] = "BOLD5000"

# Expand the lists in df["objective_cv_scores"]
df = df.explode("objective_cv_scores")

# Sort alphabetically by dataset name and solver name
df.sort_values(by=["data_name", "solver_name"], inplace=True)

# Fix underscores in the solver names
df["solver_name"] = df["solver_name"].str.replace("_", " ")

# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)

# Create the figure and axes with a specific size
fig, ax = plt.subplots(figsize=(12, 20))

# Create the box plot
sns.boxplot(
    data=df,
    x="objective_cv_scores",
    y="data_name",
    hue="solver_name",
    showfliers=False,
    ax=ax,
    # fill=False,
    # legend=False,
    # color="k",
)
# Create the scatter plot
# sns.stripplot(
#     data=df,
#     x="objective_cv_scores",
#     y="data_name",
#     size=4,
#     hue="solver_name",
#     dodge=True,
#     jitter=True,
# )

# Customize the plot
ax.set_xlabel("Accuracy", fontweight="bold")
ax.set_ylabel("Dataset", fontweight="bold")
ax.set_title(
    "Prediction accuracies for various template estimators",
    fontweight="bold",
    fontsize="large",
)
# Set the legend title
ax.legend(title="Alignment method", title_fontsize="large")

# Move the legend outside the plot
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

# Add gray rectangles to separate the datasets
for i, data_name in enumerate(df["data_name"].unique()):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.05)

# Fix underscores in the dataset names
ax.set_yticklabels(
    [name.replace("_", " ") for name in df["data_name"].unique()]
)

# Adjust the layout to prevent the legend from being cut off
plt.tight_layout()

# Save the figure with high resolution
plt.savefig(
    figures_path / "boxplot_accuracies.pdf", dpi=500, bbox_inches="tight"
)

# Display the plot
plt.show()


# %%
# Keep only the FUGW solvers with eps=0.01 and rho=100.0 and
# the Procrustes and Anatomical solvers
df_short = df[
    df["solver_name"].str.contains("FUGW")
    & df["solver_name"].str.contains("eps=0.01")
    & df["solver_name"].str.contains("rho=100.0]")
    | df["solver_name"].str.contains("Procrustes")
    | df["solver_name"].str.contains("Anatomical")
]

# Remove the useless annotations
df_short["solver_name"] = df_short["solver_name"].str.replace(
    r"FUGW\[alpha=([\d.]+).*?\]", r"FUGW(alpha=\1)", regex=True
)

df_short["solver_name"] = df_short["solver_name"].str.replace(
    r"\[.*\]", "", regex=True
)
# Sort alphabetically by dataset name and solver name
df_short.sort_values(by=["solver_name", "data_name"], inplace=True)

# Drop the Neuromod dataset
df_short.drop(
    df_short[df_short["data_name"].str.contains("Neuromod")].index,
    inplace=True,
)


# Create the figure and axes with a specific size
fig, ax = plt.subplots(figsize=(10, 6))

# Create the box plot
sns.boxplot(
    data=df_short,
    x="objective_cv_scores",
    y="data_name",
    hue="solver_name",
    showfliers=False,
    ax=ax,
    fill=False,
    legend=False,
    color="k",
)
# Create the scatter plot
sns.stripplot(
    data=df_short,
    x="objective_cv_scores",
    y="data_name",
    size=4,
    hue="solver_name",
    dodge=True,
    jitter=True,
)

# Customize the plot
ax.set_xlabel("Accuracy", fontweight="bold")
ax.set_ylabel("Dataset", fontweight="bold")
ax.set_title(
    "Prediction accuracies for various template estimators",
    fontweight="bold",
    fontsize="large",
)
# Set the legend title
ax.legend(title="Alignment method", title_fontsize="large")

# Move the legend outside the plot
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

# Add gray rectangles to separate the datasets
for i, data_name in enumerate(df_short["data_name"].unique()):
    if i % 2 == 0:
        ax.axhspan(i - 0.5, i + 0.5, color="gray", alpha=0.05)

# Fix underscores in the dataset names
ax.set_yticklabels(
    [name.replace("_", " ") for name in df_short["data_name"].unique()]
)

# Adjust the layout to prevent the legend from being cut off
plt.tight_layout()

# Save the figure with high resolution
plt.savefig(
    figures_path / "boxplot_accuracies_summary.pdf",
    dpi=500,
    bbox_inches="tight",
)


# %%
# Concatenate pdfs per dataset and solver
aligned_dataset_paths = figures_path / "aligned_datasets"
# Get the list of folders
aligned_datasets = [
    folder for folder in aligned_dataset_paths.iterdir() if folder.is_dir()
]
# Get the subfolders in one list
solvers_paths = [
    solver
    for dataset in aligned_datasets
    for solver in dataset.iterdir()
    if solver.is_dir()
]


def concat_pdf(folder):
    subject_paths = [
        subject for subject in folder.iterdir() if subject.is_dir()
    ]
    # Get the list of contrast_names
    contrasts_paths = glob.glob(str(subject_paths[0] / "*.pdf"))
    # Keep only the contrast names
    contrast_names = [
        Path(contrast_path).stem for contrast_path in contrasts_paths
    ]
    for contrast in contrast_names:
        pdf_files = []
        for subject_path in subject_paths:
            pdf_files.append(*glob.glob(str(subject_path / f"{contrast}.pdf")))
        # Sort the pdf_files
        pdf_files.sort()
        # Concatenate the pdf_files
        merger = PdfMerger()
        for pdf_file in pdf_files:
            merger.append(pdf_file)
        output_filename = folder / f"{contrast}.pdf"
        merger.write(output_filename)
        # Close the PdfMerger object
        merger.close()


for solver_path in solvers_paths:
    print(f"Concatenating pdfs for {solver_path}")
    concat_pdf(solver_path)
