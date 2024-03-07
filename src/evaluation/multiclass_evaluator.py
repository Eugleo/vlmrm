import argparse
import base64
import json
import re
from pathlib import Path
from typing import Callable, Tuple, Dict

import cv2
import dotenv
import numpy as np
import openai
import pandas as pd
import torch
from torch import Tensor
from torch.amp.autocast_mode import autocast
from einops import rearrange
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from evaluation import util
import vlmrm.reward.rewards as rewards
from vlmrm.reward.encoders import CLIP, S3D, Encoder, ViCLIP
from evaluation.evaluator import evaluate, logit_reward, mk_projection_reward, subsample, gpt4v_load_video, gpt4

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a set of zero-shot multiclass evaluations."
    )
    parser.add_argument(
        "-t",
        "--table-path",
        help="Path to a csv table containing video paths and their labels for each multiclass task.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--descriptions-for-class-labels",
        help="Path to a csv table containing class descriptions for all tasks.",
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
        help="Use several videos of same action and average similarity"
    )
    parser.add_argument(
        "--max-n-videos",
        type=int,
        default=10,
        help="Max number of videos taken from each directory when `--average-by-video` is True"
    )
    parser.add_argument("--cache-dir", default=".cache")

    args = parser.parse_args()
    return args

def get_descriptions_and_baselines_for_tasks(label_descriptions: pd.DataFrame) -> Tuple[Dict[str, Dict[int, str]], Dict[str, str]]:
    name2label2description = {}
    name2baseline = {}

    for task_name, label_id, baseline_prompt, label_prompt in label_descriptions.itertuples(index=False):
        if task_name not in name2label2description:
            name2label2description[task_name] = {label_id: label_prompt}
            name2baseline[task_name] = baseline_prompt
        else:
            assert name2baseline[task_name] == baseline_prompt, \
                f"Found differing baseline prompts within task {task_name}, {name2baseline[task_name]} and {baseline_prompt}"
            
            name2label2description[task_name][label_id] = label_prompt

    return name2label2description, name2baseline

def compute_all_vs_all_metrics(average_similarities: torch.Tensor) -> Dict[str, float]:
    n_video_groups, n_descriptions = average_similarities.shape

    identity = np.eye(n_descriptions)
    # Need to treat broadcasting carefully here
    binary_matrix = np.where(np.isclose(average_similarities, average_similarities.max(axis=1, keepdims=True)), 1, 0)

    cosine_similarity = (binary_matrix * identity / np.sqrt(n_descriptions * binary_matrix.sum())).sum()
    l2_distance = np.sqrt(np.sum(np.square((identity - binary_matrix)))) / n_descriptions
    l1_distance = np.sum(np.abs(identity - binary_matrix)) / n_descriptions

    # If we predict constant label for all videos, cosine similarity will be equal to 1/n_descriptions, since only one guess will be correct
    return {
        "constant_baseline_cosine_similarity": 1 / n_descriptions,
        "cosine_similarity": cosine_similarity,
        "l2_distance": l2_distance,
        "l1_distance": l1_distance,
    }

def compute_multiclass_metrics(average_similarities: torch.Tensor, true_labels: np.ndarray, verbose: bool) -> Dict[str, float]:
    n_samples, n_classes = average_similarities.shape
    predictions = np.argmax(average_similarities, axis=1)

    one_hot_true_labels = np.zeros((n_samples, n_classes))
    one_hot_true_labels[range(n_samples), true_labels] = 1

    if verbose:
        print("in compute_multiclass_metrics: average_similarities\n", average_similarities)
        print("="*70)

    # random performance will score 0 in adjusted_balanced_accuracy
    return {
        "accuracy": accuracy_score(true_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(true_labels, predictions),
        "adjusted_balanced_accuracy": balanced_accuracy_score(true_labels, predictions, adjusted=True),
        "roc_auc_ovr_micro": roc_auc_score(one_hot_true_labels, average_similarities, multi_class="ovr", average="micro"),
    }

@autocast("cuda", enabled=torch.cuda.is_available())
def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = pd.read_csv(args.table_path)
    label_descriptions = pd.read_csv(args.descriptions_for_class_labels)

    task_name2label2description, task_name2baseline = get_descriptions_and_baselines_for_tasks(label_descriptions)

# Getting videos
    video_paths = []
    video_group_borders = [0]

    for dir_path in data["path"].values:
        # Sorted to ensure deterministic results
        video_paths_group = sorted(list(Path(dir_path).glob("*.avi")))
        video_paths.extend(video_paths_group[:args.max_n_videos] if args.average_by_video else video_paths_group[:1])
        video_group_borders.append(len(video_paths))

    videos = util.get_video_batch(video_paths, device)
    video_group_names = [Path(p).stem for p in data["path"]]

# Choosing model
    if args.model.lower() == "gpt4":
        raise ValueError(f"gpt4 is not supported for multiclass evaluation yet")

        reward_matrix = gpt4(video_paths, descriptions)
        title = f"gpt4_{args.experiment_id}"
        util.make_heatmap(
            reward_matrix,
            groups=data["group"].to_list(),
            trajectories_names=video_names,
            labels=descriptions,
            result_dir=args.output_dir,
            experiment_id=title,
        )
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
                    (logit_reward, f"{args.model}_logit_{task_name}_{args.experiment_id}")
                )
            elif reward_name == "projection":
                baselines = encoder.encode_text([task_name2baseline[task_name]] * len(data))
                if args.alphas is None:
                    raise ValueError("Alpha must be provided when using projection reward.")
                for alpha in args.alphas.split(","):
                    reward_fun = mk_projection_reward(float(alpha), baselines)
                    title = f"{args.model}_projection_{alpha}_{task_name}_{args.experiment_id}"
                    task_name2named_reward_functions[task_name].append((reward_fun, title))
            else:
                raise ValueError(f"Unknown reward name {reward_name}")

# Running evaluations
    experiment_dir = Path(args.output_dir) / args.experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    task_name2title2metrics = {}

    for i, task_name in enumerate(task_name2named_reward_functions):
        if args.verbose:
            print(f"({i + 1}/{len(task_name2named_reward_functions)})  Task {task_name}")

        true_labels = data[task_name]
        descriptions = [task_name2label2description[task_name][i]
            for i in range(len(task_name2label2description[task_name]))]

        for j, (reward_fun, title) in enumerate(task_name2named_reward_functions[task_name]):
            if args.verbose:
                print(f"  ({j + 1}/{len(task_name2named_reward_functions[task_name])})   Evaluating {title}")

            reward_matrix = evaluate(encoder, videos, descriptions, reward_fun)

            average_similarities, std_similarities = util.aggregate_similarities_many_video_groups(
                reward_matrix,
                prompt_group_borders=range(len(descriptions) + 1),
                video_group_borders=video_group_borders,
                do_normalize=args.standardize,
            )

            # util.make_heatmap(
            #     average_similarities,
            #     groups=data["group"].to_list(),
            #     trajectories_names=video_group_names,
            #     labels=descriptions,
            #     result_dir=str(experiment_dir),
            #     experiment_id=title,
            # )

            metrics = compute_multiclass_metrics(average_similarities, true_labels, args.verbose)

            if task_name in task_name2title2metrics:
                task_name2title2metrics[task_name][title] = metrics
            else:
                task_name2title2metrics[task_name] = {title: metrics}

    with open(experiment_dir / "metrics.json", "w") as f:
        json.dump(task_name2title2metrics, f, indent=2)


if __name__ == "__main__":
    main()
