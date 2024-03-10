from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score

sns.set_theme(style="whitegrid")


def _balanced_accuracy(group: pl.DataFrame):
    return group.with_columns(
        balanced_accuracy=balanced_accuracy_score(
            group["true_label"].to_numpy(), group["predicted_label"].to_numpy()
        )
    )


def _max_prob(df: pl.DataFrame, col: str):
    return df.group_by(["video", "task", "model", "reward"]).agg(
        # Extract the label with the highest probability
        pl.col("label").sort_by(col).last()
    )


def main():
    # Load the latest experiment
    latest_experiment = sorted(Path("out").iterdir())[-1]
    experiment_dir = latest_experiment

    df = pl.read_csv(experiment_dir / "results.csv")

    predicted_labels = _max_prob(df, "probability").rename({"label": "predicted_label"})
    true_labels = _max_prob(df, "true_probability").rename({"label": "true_label"})

    results = (
        predicted_labels.join(true_labels, on=["video", "task", "model", "reward"])
        .group_by("task", "model", "reward")
        .map_groups(_balanced_accuracy)
        .with_columns(
            pl.concat_str("model", "reward", separator="+").alias("evaluator")
        )
    )

    plot_dir = Path(experiment_dir / "plots")
    plot_dir.mkdir(exist_ok=True)

    for task in results["task"].unique():
        task_results = results.filter(pl.col("task") == task).sort("evaluator")
        plt.figure(figsize=(20, 10))
        g = sns.catplot(
            data=task_results.to_pandas(),
            kind="bar",
            x="evaluator",
            y="balanced_accuracy",
            hue="model",
            palette="deep",
            alpha=0.6,
            legend_out=False,
        )
        g.set_axis_labels("Model", "Balanced accuracy")
        g.set_xticklabels(rotation=30, ha="right")
        # g.legend.set_title("")

        plt.title(f"Balanced accuracy, {task}")
        plt.tight_layout()

        plt.savefig(plot_dir / f"{task}_balanced_accuracy.pdf", dpi=350)

    # # Create a dataframe
    # rows = []

    # for name in experiment_names:
    #     with open(result_dir / name / "metrics.json", "r") as f:
    #         task_name2title2metrics = json.load(f)

    #     for task_name, title2metrics in task_name2title2metrics.items():
    #         # We can select any of particular ways of using the model -- logit, projection, etc
    #         first_key = min(title2metrics.keys())
    #         print(f"Using {first_key}")

    #         rows.append(
    #             {
    #                 "experiment_name": name,
    #                 "task_name": task_name,
    #                 target_metric: title2metrics[first_key][target_metric],
    #             }
    #         )

    #         plt.figure(figsize=(10, 9))
    #         g = sns.heatmap(
    #             title2metrics[first_key]["confusion_matrix"],
    #             annot=True,
    #             fmt="2d",
    #             cmap="crest",
    #             xticklabels=task_name2label2description[task_name].keys(),
    #             yticklabels=task_name2label2description[task_name].keys(),
    #         )
    #         plt.xticks(rotation=30, ha="right")
    #         plt.ylabel("True")
    #         plt.xlabel("Predicted")
    #         plt.title(f"Confusion matrix\n{task_name}, {first_key}")
    #         plt.tight_layout()
    #         plt.savefig(
    #             result_dir
    #             / target_sub_dir
    #             / f"{task_name}_{first_key}_confusion_matrix.pdf",
    #             dpi=350,
    #         )


if __name__ == "__main__":
    main()
