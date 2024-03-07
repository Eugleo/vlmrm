import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# Creates grouped bar chart, reporting same metric for various tasks for group of experiments

def main():
    target_metric = "balanced_accuracy"
    result_dir = Path("out/")

    #experiment_names = ["s3d", "viclip"]
    experiment_names = ["habitat_s3d", "habitat_viclip"]

# Create a dataframe
    rows = []

    for name in experiment_names:
        with open(result_dir / name / "metrics.json", "r") as f:
            task_name2title2metrics = json.load(f)
        
        for task_name, title2metrics in task_name2title2metrics.items():
            # We can select any of particular ways of using the model -- logit, projection, etc
            first_key = min(title2metrics.keys())
            
            rows.append({"experiment_name": name, "task_name": task_name, target_metric: title2metrics[first_key][target_metric]})

    summary = pd.DataFrame(rows)

# Create a plot
    plt.figure(figsize=(12, 5))
    g = sns.catplot(data=summary, kind="bar", x="task_name", y=target_metric, hue="experiment_name", palette="deep", alpha=0.6, legend_out=False)
    g.set_axis_labels("Task", target_metric)
    g.set_xticklabels(rotation=30, ha="right")
    g.legend.set_title("")

    plt.title(f"{target_metric} per task")
    plt.tight_layout()

    (result_dir / "summary_plots").mkdir(exist_ok=True)
    plt.savefig(result_dir / "summary_plots" / ("-".join(experiment_names) + ".pdf"), dpi=350)

if __name__ == "__main__":
    main()