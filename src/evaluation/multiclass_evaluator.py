import argparse
import base64
import datetime
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import cv2
import polars as pl
import torch
import yaml
from einops import rearrange
from evaluation import util
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torchvision.io import read_video
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
        "--models",
        help="Names of the models to evaluate (ViCLIP, S3D, CLIP, GPT4)",
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
        help="If using projection reward, the values of alpha to use.",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save evaluation results.",
        default="out",
    )
    parser.add_argument(
        "--log-gpt-inputs",
        help="Whether to log the frames that GPT4 gets",
        action="store_true",
        default=False,
    )

    parser.add_argument("--cache-dir", default=".cache")

    args = parser.parse_args()
    return args


@dataclass
class Task:
    id: str
    gpt4_prompt: str
    baseline_prompt: str
    labels: Dict[str, str]

    @staticmethod
    def from_dict(val):
        return Task(val["id"], val["gpt4_prompt"], val["baseline"], val["labels"])


@dataclass
class Reward:
    id: str
    _fun: Callable[..., Tensor]

    def __call__(self, **kwargs):
        return self._fun(**kwargs)


@dataclass
class Evaluator:
    id: str
    rewards: list[Reward]


class VideoDataset(IterableDataset):
    def __init__(self, paths: list[str], device: torch.device, transforms=[]) -> None:
        self.paths = paths
        self.device = device
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for path in self.paths:
            video = read_video(path, pts_unit="sec")[0].to(self.device)
            for transform in self.transforms:
                video = transform(video)
            yield video

    def __getitem__(self, idx: int) -> Tensor:
        video = read_video(self.paths[idx], pts_unit="sec")[0].to(self.device)
        for transform in self.transforms:
            video = transform(video)
        return video


def _load_tasks(path) -> list[Task]:
    with open(path) as f:
        return [Task.from_dict(task) for task in yaml.safe_load(f)]


def _load_rewards(args) -> list[Reward]:
    rewards = []
    for reward_name in args.rewards.split(","):
        if reward_name == "logit":
            rewards.append(Reward("logit", util.logit_reward))
        elif reward_name == "projection":
            if args.alphas is None:
                raise ValueError("Alpha must be provided when using projection reward.")
            for alpha in args.alphas.split(","):
                rewards.append(
                    Reward(
                        f"projection_{alpha}",
                        partial(util.projection_reward, alpha=float(alpha)),
                    )
                )
        else:
            raise ValueError(f"Unknown reward name {reward_name}")
    return rewards


def _load_evaluators(args) -> list[Evaluator]:
    evaluators = []
    for model_name in args.models.split(","):
        if model_name not in ["viclip", "s3d", "clip", "gpt4"]:
            raise ValueError(f"Unknown model name {model_name}")

        if model_name == "gpt4":
            rewards = [
                Reward("default", partial(util.gpt4, log_inputs=args.log_gpt_inputs))
            ]
        else:
            rewards = _load_rewards(args)

        logging.info(f"Loading evaluator {model_name} with rewards {rewards}")
        evaluators.append(Evaluator(model_name, rewards))

    return evaluators


def _load_encoder(model_name: str, args, device: torch.device) -> Encoder:
    assert model_name in ["viclip", "s3d", "clip"]

    logging.info(f"Loading encoder {model_name}")

    if model_name == "viclip":
        encoder = ViCLIP(args.cache_dir)
    elif model_name == "s3d":
        encoder = S3D(args.cache_dir)
    elif model_name == "clip":
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
    return encoder.to(device)


def _frames_to_encodings(frames: Tensor, encoder: Encoder) -> Tensor:
    frames = rearrange(frames, "b f c h w -> f 1 b c h w")
    logging.info(f"Encoding {frames.shape=} using {encoder.__class__.__name__}")
    encodings = encoder.encode_video(frames)
    logging.info(f"Encoding complete, {encodings.shape=}")
    encodings = rearrange(encodings, "1 b d -> b d")
    return encodings


def _frames_to_b64(frames: Tensor):
    frames_np = frames.numpy()
    # Convert RGB to BGR
    frames_np = frames_np[:, :, :, ::-1]

    b64_frames = []
    for frame in frames_np:
        _, buffer = cv2.imencode(".jpg", frame)
        b64_frames.append(base64.b64encode(buffer).decode("utf-8"))  # type: ignore

    return b64_frames


def main():
    args = parse_args()
    experiment_id = "Exp_{}".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    experiment_dir = Path(args.output_dir) / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=experiment_dir / "experiment.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Starting experiment {experiment_id} with args {args}")

    data = pl.read_csv(args.data)
    logging.info(f"Loading {len(data)} videos from {args.data}")
    video_paths = data["path"].to_list()
    tasks = _load_tasks(args.tasks)
    # This doesn't load the models themselves, just the rewards and configs
    evaluators = _load_evaluators(args)

    # Put GPT first because it needs the videos to be on a CPU unlike the other models
    evaluators = sorted(evaluators, key=lambda r: r.id != "gpt4")

    results = []
    # Loading the model is slow, so we do it only once, then score all tasks and rewards,
    # and then free the memory before loading the next model

    for evaluator in evaluators:
        logging.info(f"Starting evaluation for model {evaluator.id}")

        if evaluator.id == "gpt4":
            # TODO: Make the number of frames to use for GPT4 a parameter
            device = torch.device("cpu")
            transforms = [partial(util.subsample, frames=5), _frames_to_b64]
            frames_enc = list(VideoDataset(video_paths, device, transforms))
        else:
            # Using even the small models on CPU is very slow
            device = torch.device("cuda:0")
            encoder = _load_encoder(evaluator.id, args, device)
            transforms = [
                encoder.subsample,
                partial(rearrange, pattern="f h w c -> f c h w", c=3),
                encoder._transform,
            ]
            dataset = VideoDataset(video_paths, device, transforms)
            dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
            framec_enc = []
            for frames in dataloader:
                framec_enc.append(_frames_to_encodings(frames, encoder))
            frames_enc = torch.cat(framec_enc)

        for task in tasks:
            label_ids = list(task.labels.keys())
            labels = list(task.labels.values())
            logging.info(
                f"Starting evaluation for task {task.id}, {label_ids=}, {labels=}"
            )
            frame_mask = torch.Tensor(data[task.id].is_null().to_list())

            if evaluator.id != "gpt4":
                assert isinstance(frames_enc, torch.Tensor)
                device = frames_enc.device

                frames_enc_masked = frames_enc[frame_mask == 0]
                logging.info(f"Masked {frames_enc_masked.count_nonzero()} frames")

                labels_enc = encoder.encode_text(labels).to(device)
                baselines_enc = encoder.encode_text(
                    [task.baseline_prompt] * len(task.labels)
                ).to(device)
                logging.info(
                    f"Encoded labels and baselines, {labels_enc.shape=}, {baselines_enc.shape=}"
                )

            for reward in evaluator.rewards:
                if evaluator.id == "gpt4":
                    # The objects below represent the task and the evaluator config
                    # and are used to load and save the cache
                    task_info = {
                        "id": task.id,
                        "gpt4_prompt": task.gpt4_prompt,
                        "labels": labels,
                    }
                    eval_info = {"id": evaluator.id, "reward": reward.id, "n_frames": 5}
                    cache = util.load_cache(args.cache_dir, task_info, eval_info)
                    scores = reward(
                        frames_enc=[
                            v for i, v in enumerate(frames_enc) if not frame_mask[i]
                        ],
                        paths=video_paths,  # Only used for logging and cache
                        cache=cache,  # Is modified inside the function
                        labels=labels,
                        prompt=task.gpt4_prompt,
                    )
                    util.save_cache(cache, args.cache_dir, task_info, eval_info)
                else:
                    scores = reward(
                        frames_enc=frames_enc_masked,
                        labels_enc=labels_enc,
                        baselines_enc=baselines_enc,
                    )

                for data_row, score_row in zip(data.rows(named=True), scores):
                    for label, score in zip(label_ids, score_row):
                        metadata = {
                            k: v
                            for k, v in data_row.items()
                            if k != task.id and v != pl.Null
                        }
                        results.append(
                            {
                                "task": task.id,
                                "model": evaluator.id,
                                "reward": reward.id,
                                "label": label,
                                "probability": score.item(),
                                "true_probability": int(data_row[task.id] == label),
                                **metadata,
                            }
                        )

        # Free GPU memory since we are done with this model
        if evaluator.id != "gpt4":
            logging.info(f"Freeing memory for model {evaluator.id}")
            del encoder
            torch.cuda.empty_cache()

    logging.info(f"Saving results to {experiment_dir / 'results.csv'}")
    pl.DataFrame(results).write_csv(experiment_dir / "results.csv")


if __name__ == "__main__":
    main()
