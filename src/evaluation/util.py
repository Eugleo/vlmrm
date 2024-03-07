import os
from typing import List, Union, Tuple, Callable

from pathlib import Path
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import Tensor
from einops import rearrange
from matplotlib.patches import Circle, Rectangle

from vlmrm.reward.encoders import Encoder
import vlmrm.reward.rewards as rewards

def load_video(path: str):
    if path.endswith(".mp4"):
        return iio.imread(path, plugin="pyav")
    elif path.endswith(".avi"):
        return iio.imread(path)


def get_video_batch(video_paths: List[Union[str, Path]], device) -> List[torch.Tensor]:
    return [torch.from_numpy(load_video(str(p))).to(device) for p in video_paths]


def make_heatmap(
    similarity_matrix: np.ndarray,
    groups: List[str],
    trajectories_names: List[str],
    labels: List[str],
    result_dir: str,
    experiment_id: str,
):
    fig, ax = plt.subplots(figsize=(25, 25))

    new_similarity_matrix = similarity_matrix.copy()
    new_labels = labels.copy()
    new_trajectory_names = trajectories_names.copy()
    shift = 0
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            new_similarity_matrix = np.insert(
                new_similarity_matrix, i + shift, np.nan, axis=0
            )
            new_similarity_matrix = np.insert(
                new_similarity_matrix, i + shift, np.nan, axis=1
            )
            new_labels.insert(i + shift, "")
            new_trajectory_names.insert(i + shift, "")
            shift += 1

    sns.heatmap(
        new_similarity_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=new_labels,
        yticklabels=new_trajectory_names,
        cbar=False,
    )

    for i in range(similarity_matrix.shape[0] + len(groups)):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor="red", lw=3))

    # Add borders around the cells with the highest values
    for i in range(new_similarity_matrix.shape[0]):
        row = new_similarity_matrix[i, :]
        if not np.isnan(row).all():
            mask = np.isfinite(row)
            # Use the mask to ignore nan values when sorting
            # Use the mask to ignore nan values when sorting
            n = 2
            # Get unique values and their indices
            unique_values, unique_indices = np.unique(row[mask], return_inverse=True)
            # Sort the unique values and get the indices of the top n
            top_n_indices = np.argsort(unique_values)[-n:]
            # Get the indices of the top n unique values in the original array
            top_n_indices = np.where(np.isin(unique_indices, top_n_indices))[0]
            # Adjust the indices to account for the mask
            top_n_indices = np.arange(len(row))[mask][top_n_indices]
            for j, index in enumerate(top_n_indices):
                ax.add_patch(
                    Circle(
                        (index + 0.5, i + 0.5),
                        0.5,
                        fill=False,
                        edgecolor="violet",
                        lw=6,
                    )
                )

    plt.title(f"{experiment_id}")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    plt.savefig(f"{result_dir}/{experiment_id}.pdf", dpi=350)

def aggregate_similarities_many_video_groups(similarities: np.ndarray, prompt_group_borders: List[int], video_group_borders: List[int], do_normalize: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes a all-to-all similarity matrix between grouped videos and prompts. Aggregates the similarities by groups.
    Videos in one group are supposed to demonstrate same behavior (moving the kettle, or toppling something over) with variations in unimportant details (background, camera angle, ...).
    Prompts in one group are supposed to be different wordings of same idea (e.g. "robotic hand moves the kettle" and "robot moves the kettle" are in one group in Franka Kitchen environment).
    Input:
        similarities: (n_total_videos, n_total_prompts)
        prompt_group_borders: List[int] of length (n_prompt_groups + 1) -- prompt group `i` is between `prompt_group_borders[i]` and `prompt_group_borders[i+1]`
        video_group_borders: List[int] of length (n_video_groups + 1) -- analogous
        do_normalize: bool -- whether to normalize similarity for each prompt over all videos

    Output:
        average_similarities: (n_video_groups, n_prompt_groups) -- average similarity between a group of videos and prompts
        std_similarities: (n_video_groups, n_prompt_groups) -- standard deviation of similarity betwenn a group of videos and prompts
    """
    n_total_videos, n_total_prompts = similarities.shape
    n_video_groups = len(video_group_borders) - 1
    n_prompt_groups = len(prompt_group_borders) - 1

    if do_normalize:
        bias = similarities.mean(0, keepdims=True)
        scale = similarities.std(0, keepdims=True)
        similarities = (similarities - bias) / scale

    average_similarities = np.empty((n_video_groups, n_prompt_groups))
    std_similarities = np.empty((n_video_groups, n_prompt_groups))

    for video_group_idx in range(n_video_groups):
        for prompt_group_idx in range(n_prompt_groups):
            p_start, p_end = prompt_group_borders[prompt_group_idx], prompt_group_borders[prompt_group_idx + 1]
            v_start, v_end = video_group_borders[video_group_idx], video_group_borders[video_group_idx + 1]

            average_similarities[video_group_idx, prompt_group_idx] = similarities[v_start:v_end, p_start:p_end].mean()
            std_similarities[video_group_idx, prompt_group_idx] = similarities[v_start:v_end, p_start:p_end].std()

    return average_similarities, std_similarities


def evaluate(
    encoder: Encoder,
    videos: List[Tensor],
    descriptions: List[str],
    reward: Callable[[Tensor, Tensor], Tensor],
):
    subsampled_videos = torch.stack([encoder.subsample(video) for video in videos])
    # The encoder expects the input to be (frames, windows, episodes, c h w)
    subsampled_videos = rearrange(subsampled_videos, "b f c h w -> f 1 b c h w")
    # (f w e c h w) -> (w e d)
    video_encodings = encoder.encode_video(subsampled_videos)
    video_encodings = rearrange(video_encodings, "1 b d -> b d")

    description_encodings = encoder.encode_text(descriptions)
    description_encodings.to(video_encodings.device)

    return reward(video_encodings, description_encodings)


def logit_reward(video_encodings: Tensor, description_encodings: Tensor):
    return rewards.logit_reward(
        video_encodings, description_encodings, torch.arange(len(description_encodings))
    )


def mk_projection_reward(alpha: float, baselines: Tensor):
    def reward(video_encodings: Tensor, description_encodings: Tensor) -> Tensor:
        reward_cols = [
            rewards.projection_reward(video_encodings, b, t.unsqueeze(0), alpha)
            for t, b in zip(description_encodings, baselines)
        ]
        return torch.stack(reward_cols, dim=1)

    return reward


def subsample(x: torch.Tensor, frames: int) -> torch.Tensor:
    n_frames, *_ = x.shape
    step = n_frames // frames
    x = x[::step, ...][:frames, ...]
    return x
