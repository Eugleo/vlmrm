import argparse
import base64
import json
import re
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Union

import cv2
import dotenv
import numpy as np
import openai
import pandas as pd
import torch
import vlmrm.reward.rewards as rewards
import yaml
from einops import rearrange
from evaluation import util
from evaluation.evaluator import gpt4, gpt4v_load_video
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from torch import Tensor
from torch.amp.autocast_mode import autocast
from vlmrm.reward.encoders import CLIP, S3D, Encoder, ViCLIP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a set of zero-shot multiclass evaluations."
    )
    parser.add_argument(
        "-d",
        "--data",
        help="Path to a csv table containing video paths and their labels for each multiclass task.",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--tasks",
        help="Path to a yaml file containing task definitions (labels, prompts).",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Name of the model to evaluate (ViCLIP, S3D, CLIP)",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--n_frames",
        help="How many frames to use for the video encoding. Only used in CLIP.",
    )
    parser.add_argument(
        "-r",
        "--rewards",
        help="Name of the reward to calculate (logit, projection)",
    )
    parser.add_argument(
        "-a",
        "--alphas",
        help="If using projection reward, the value of alpha to use.",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--experiment-id",
        help="Name of current experiment (used to save the results)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save evaluation results.",
        default="out",
    )
    parser.add_argument(
        "--standardize",
        help="Directory to save evaluation results.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--average-by-video",
        action="store_true",
        help="Use several videos of same action and average similarity",
    )
    parser.add_argument(
        "--max-n-videos",
        type=int,
        default=10,
        help="Max number of videos taken from each directory when `--average-by-video` is True",
    )
    parser.add_argument("--cache-dir", default=".cache")

    args = parser.parse_args()
    return args

ConfusionMatrix = List[List[int]]

def compute_multiclass_metrics(
    average_similarities: torch.Tensor, true_labels: np.ndarray, verbose: bool
) -> Dict[str, Union[float, ConfusionMatrix]]:
    n_samples, n_classes = average_similarities.shape
    predictions = np.argmax(average_similarities, axis=1)

    if verbose:
        # note: when using prpjection reward, numbers are very large by modulo, seems like a bug
        print(
            "in compute_multiclass_metrics: average_similarities\n",
            average_similarities,
        )
        print("=" * 70)
        print(f"{true_labels.shape=}, {predictions.shape=}")
        print(f"{true_labels=}, {predictions=}")

    one_hot_true_labels = np.zeros((n_samples, n_classes))
    one_hot_true_labels[range(n_samples), true_labels] = 1

    # random performance will score 0 in adjusted_balanced_accuracy
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predictions),
        "adjusted_balanced_accuracy": balanced_accuracy_score(
            true_labels, predictions, adjusted=True
        ),
        "roc_auc_ovr_micro": roc_auc_score(
            one_hot_true_labels,
            average_similarities,
            multi_class="ovr",
            average="micro",
        ),
        "confusion_matrix": confusion_matrix(true_labels, predictions).tolist(),
    }


@autocast("cuda", enabled=torch.cuda.is_available())
def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(args.data)

    tasks = []
    with open(args.tasks) as stream:
        try:
            tasks = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    task_name2baseline = {task["name"]: task["baseline_prompt"] for task in tasks}
    task_name2label2description = {
        task["name"]: task["label_prompts"] for task in tasks
    }
    task_name2prompt = {task["name"]: task["gpt4_prompt"] for task in tasks}

    # Getting videos
    video_paths = data["path"].to_list()
    # TODO: Groups are probably not needed (we will group by task), remove
    video_group_borders = list(range(len(video_paths) + 1))

    videos = util.get_video_batch(video_paths, device)

    # Setting up output directory
    experiment_dir = Path(args.output_dir) / args.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    task_name2title2metrics = {}

    # Choosing model
    if args.model.lower() == "gpt4":
        for task_name in task_name2baseline:
            title = f"gpt4_{task_name}_{args.experiment_id}"

            label_name_to_index = {
                label: i
                for i, label in enumerate(task_name2label2description[task_name])
            }
            true_labels = np.array([label_name_to_index[l] for l in data[task_name]])
            descriptions = list(task_name2label2description[task_name].values())

            reward_matrix = gpt4(video_paths, descriptions, task_name2prompt[task_name])
            # average_similarities, std_similarities = (
            #     util.aggregate_similarities_many_video_groups(
            #         reward_matrix,
            #         prompt_group_borders=range(len(descriptions) + 1),
            #         video_group_borders=video_group_borders,
            #         do_normalize=args.standardize,
            #     )
            # )

            average_similarities = reward_matrix

            # util.make_heatmap(
            #     average_similarities,
            #     groups=data["group"].to_list(),
            #     trajectories_names=video_names,
            #     labels=descriptions,
            #     result_dir=str(experiment_dir),
            #     experiment_id=title,
            # )

            metrics = compute_multiclass_metrics(
                average_similarities, true_labels, args.verbose
            )

            if task_name in task_name2title2metrics:
                task_name2title2metrics[task_name][title] = metrics
            else:
                task_name2title2metrics[task_name] = {title: metrics}

        with open(experiment_dir / "metrics.json", "w") as f:
            json.dump(task_name2title2metrics, f, indent=2)

        # np.save(experiment_dir / "reward_matrix.npy", reward_matrix)
        # np.save(experiment_dir / "average_similarities.npy", average_similarities)
        # np.save(experiment_dir / "std_similarites.npy", std_similarites)

        return

    assert isinstance(args.model, str)

    if args.model.lower() == "viclip":
        encoder = ViCLIP(args.cache_dir)
    elif args.model.lower() == "s3d":
        encoder = S3D(args.cache_dir)
    elif args.model.lower() == "clip":
        if args.n_frames is None:
            raise ValueError("Number of frames must be provided when using CLIP.")
        model = "ViT-bigG-14/laion2b_s39b_b160k"
        model_name_prefix, pretrained = model.split("/")
        encoder = CLIP(
            model_name_prefix,
            pretrained,
            args.cache_dir,
            expected_n_frames=int(args.n_frames),
        )
    encoder = encoder.to(device)

    # Parsing which reward functions we should try
    task_name2named_reward_functions = {}

    for task_name in task_name2baseline:
        task_name2named_reward_functions[task_name] = []

        for reward_name in args.rewards.split(","):
            if reward_name == "logit":
                task_name2named_reward_functions[task_name].append(
                    (
                        util.logit_reward,
                        f"{args.model}_logit_{task_name}_{args.experiment_id}",
                    )
                )
            elif reward_name == "projection":
                baselines = encoder.encode_text(
                    [task_name2baseline[task_name]]
                    * len(task_name2label2description[task_name])
                )
                if args.alphas is None:
                    raise ValueError(
                        "Alpha must be provided when using projection reward."
                    )
                for alpha in args.alphas.split(","):
                    reward_fun = util.mk_projection_reward(float(alpha), baselines)
                    title = f"{args.model}_projection_{alpha}_{task_name}_{args.experiment_id}"
                    task_name2named_reward_functions[task_name].append(
                        (reward_fun, title)
                    )
            else:
                raise ValueError(f"Unknown reward name {reward_name}")

    # Running evaluations
    for i, task_name in enumerate(task_name2named_reward_functions):
        if args.verbose:
            print(
                f"({i + 1}/{len(task_name2named_reward_functions)})  Task {task_name}"
            )

        descriptions = list(task_name2label2description[task_name].values())

        label_name_to_index = {
            label: i for i, label in enumerate(task_name2label2description[task_name])
        }
        true_labels = np.array([label_name_to_index[l] for l in data[task_name]])

        for j, (reward_fun, title) in enumerate(
            task_name2named_reward_functions[task_name]
        ):
            if args.verbose:
                print(
                    f"  ({j + 1}/{len(task_name2named_reward_functions[task_name])})   Evaluating {title}"
                )

            reward_matrix = util.evaluate(encoder, videos, descriptions, reward_fun)

            # TODO: Groups are probably not needed (we will group by task), remove
            # ...unless we want to test many alternative labels, but maybe we don't need to
            # average_similarities, std_similarities = (
            #     util.aggregate_similarities_many_video_groups(
            #         reward_matrix,
            #         prompt_group_borders=range(len(descriptions) + 1),
            #         video_group_borders=video_group_borders,
            #         do_normalize=args.standardize,
            #     )
            # )

            average_similarities = reward_matrix.cpu().numpy()

            # util.make_heatmap(
            #     average_similarities,
            #     groups=data["group"].to_list(),
            #     trajectories_names=video_group_names,
            #     labels=descriptions,
            #     result_dir=str(experiment_dir),
            #     experiment_id=title,
            # )

            metrics = compute_multiclass_metrics(
                average_similarities, true_labels, args.verbose
            )

            if task_name in task_name2title2metrics:
                task_name2title2metrics[task_name][title] = metrics
            else:
                task_name2title2metrics[task_name] = {title: metrics}

    with open(experiment_dir / "metrics.json", "w") as f:
        json.dump(task_name2title2metrics, f, indent=2)


if __name__ == "__main__":
    main()
