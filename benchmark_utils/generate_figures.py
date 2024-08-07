# %%
import pandas as pd
import os
import glob
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

data_path = Path(__file__).parent.parent / "outputs"
figures_path = data_path / "figures"
figures_path.mkdir(parents=True, exist_ok=True)
# Parse the latest file
file_list = glob.glob(os.path.join(data_path, "*.parquet"))
latest_file = max(file_list, key=os.path.getmtime)
df = pd.read_parquet(latest_file)

# Remove the simulated data
df.drop(df[df["data_name"].str.contains("Simulated")].index, inplace=True)

# Expand the lists in df["objective_scores"]
df = df.explode("objective_scores")

# Set the style and font scale for better readability
plt.style.use(["science", "nature", "no-latex"])
sns.set_context("paper", font_scale=1.3)

# Create the figure and axes with a specific size
fig, ax = plt.subplots(figsize=(10, 6))

# Create the box plot
sns.boxplot(
    data=df,
    x="objective_scores",
    y="data_name",
    hue="solver_name",
    showfliers=False,
    ax=ax,
)

# Customize the plot
ax.set_xlabel("Accuracy", fontweight="bold")
ax.set_ylabel("Dataset", fontweight="bold")
ax.set_title(
    "Prediction accuracies for various template estimators", fontweight="bold"
)

# Move the legend outside the plot
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

# Adjust the layout to prevent the legend from being cut off
plt.tight_layout()

# Save the figure with high resolution
# plt.savefig("boxplot_paper_ready.png", dpi=300, bbox_inches="tight")

# Display the plot
plt.show()
