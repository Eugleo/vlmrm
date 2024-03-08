import json
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# Creates grouped bar chart, reporting same metric for various tasks for group of experiments

def main():
    target_metric = "balanced_accuracy"
    result_dir = Path("out/")
    target_sub_dir = "summary_plots"
    path_to_task_info = "data/habitat/tasks.yaml"

    #experiment_names = ["s3d", "viclip"]
    experiment_names = ["habitat_s3d", "habitat_viclip"]

    (result_dir / target_sub_dir).mkdir(exist_ok=True)

    tasks = []
    with open(path_to_task_info) as stream:
        try:
            tasks = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    task_name2baseline = {task["name"]: task["baseline_prompt"] for task in tasks}
    task_name2label2description = {
        task["name"]: task["label_prompts"] for task in tasks
    }
    task_name2prompt = {task["name"]: task["gpt4_prompt"] for task in tasks}

# Create a dataframe
    rows = []

    for name in experiment_names:
        with open(result_dir / name / "metrics.json", "r") as f:
            task_name2title2metrics = json.load(f)
        
        for task_name, title2metrics in task_name2title2metrics.items():
            # We can select any of particular ways of using the model -- logit, projection, etc
            first_key = min(title2metrics.keys())
            print(f"Using {first_key}")
            
            rows.append({"experiment_name": name, "task_name": task_name, target_metric: title2metrics[first_key][target_metric]})

            plt.figure(figsize=(10, 9))
            g = sns.heatmap(
                title2metrics[first_key]["confusion_matrix"],
                annot=True,
                fmt="2d",
                cmap="crest",
                xticklabels=task_name2label2description[task_name].keys(),
                yticklabels=task_name2label2description[task_name].keys()
            )
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.title(f"Confusion matrix\n{task_name}, {first_key}")
            plt.tight_layout()
            plt.savefig(result_dir / target_sub_dir / f"{task_name}_{first_key}_confusion_matrix.pdf", dpi=350)

    summary = pd.DataFrame(rows)

# Create a plot
    plt.figure(figsize=(12, 5))
    g = sns.catplot(data=summary, kind="bar", x="task_name", y=target_metric, hue="experiment_name", palette="deep", alpha=0.6, legend_out=False)
    g.set_axis_labels("Task", target_metric)
    g.set_xticklabels(rotation=30, ha="right")
    g.legend.set_title("")

    plt.title(f"{target_metric} per task")
    plt.tight_layout()

    plt.savefig(result_dir / target_sub_dir / ("-".join(experiment_names) + ".pdf"), dpi=350)

if __name__ == "__main__":
    main()