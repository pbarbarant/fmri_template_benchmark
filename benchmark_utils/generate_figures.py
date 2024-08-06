# %%
import pandas as pd
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

path_fugw = Path(
    "/home/mind/pbarbara/.paths/pbarbara/fmri_template_benchmark/outputs/benchopt_run_2024-08-06_21h39m08.parquet"
)
path_anat = Path(
    "/home/mind/pbarbara/.paths/pbarbara/fmri_template_benchmark/outputs/benchopt_run_2024-08-06_21h31m32.parquet"
)

df_anat = pd.read_parquet(path_anat)
df_fugw = pd.read_parquet(path_fugw)
df = pd.concat([df_anat, df_fugw])

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
ax.set_title("Prediction accuracies for various template estimators", fontweight="bold")

# Move the legend outside the plot
sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))

# Adjust the layout to prevent the legend from being cut off
plt.tight_layout()

# Save the figure with high resolution
# plt.savefig("boxplot_paper_ready.png", dpi=300, bbox_inches="tight")

# Display the plot
plt.show()
