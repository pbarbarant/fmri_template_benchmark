# %%
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns


plt.rcParams["figure.dpi"] = 500

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path.parent / "outputs" / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
# Parse the latest file
file_list = glob.glob(os.path.join(data_path, "*.parquet"))
latest_file = max(file_list, key=os.path.getmtime)

# %% Plot the boxplot for the accuracies
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
fig, ax = plt.subplots(figsize=(12, 5))

# Create the box plot
sns.boxplot(
    data=df,
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
    data=df,
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
# Do the same for the Pearson correlation
df = pd.read_parquet(latest_file)

# Remove the simulated data
df.drop(df[df["data_name"].str.contains("Simulated")].index, inplace=True)

# Merge all BOLD5000 folds into one
df.loc[df["data_name"].str.contains("BOLD5000"), "data_name"] = "BOLD5000"

# Expand the lists in df["objective_cv_scores"]
df = df.explode("objective_pearson_corrs")

# Sort alphabetically by dataset name and solver name
df.sort_values(by=["data_name", "solver_name"], inplace=True)

# Fix underscores in the solver names
df["solver_name"] = df["solver_name"].str.replace("_", " ")

# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)

# Create the figure and axes with a specific size
fig, ax = plt.subplots(figsize=(12, 5))

# Create the box plot
sns.boxplot(
    data=df,
    x="objective_pearson_corrs",
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
    data=df,
    x="objective_pearson_corrs",
    y="data_name",
    size=4,
    hue="solver_name",
    dodge=True,
    jitter=True,
)

# Customize the plot
ax.set_xlabel("Pearson Correlation", fontweight="bold")
ax.set_ylabel("Dataset", fontweight="bold")
ax.set_title(
    "Average voxel-wise Pearson correlation to the template",
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
    figures_path / "boxplot_correlation.pdf", dpi=500, bbox_inches="tight"
)

# Display the plot
plt.show()
