from typing import List, Literal, Optional, Tuple, overload

import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, reduce, repeat
from torch.amp.autocast_mode import autocast

from vlmrm.contrib.open_clip.transform import image_transform
from vlmrm.trainer.config import CLIPRewardConfig


class BaseModel(nn.Module):
    def embed_text(self, x) -> torch.Tensor:
        raise NotImplementedError

    def embed_image(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CLIP(BaseModel):
    _model: open_clip.model.CLIP

    def __init__(self, model_name: str, pretrained: str, cache_dir: str):
        super().__init__()
        self._model = open_clip.create_model(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
        )  # type: ignore

    @torch.inference_mode()
    def embed_text(self, x: List[str]) -> torch.Tensor:
        tokens = open_clip.tokenize(x)
        encoded = self._model.encode_text(tokens).float()
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        return encoded

    @torch.inference_mode()
    def embed_image(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._model.encode_image(x, normalize=True)
        return encoded


class Embed(nn.Module):
    def __init__(self, embed_model):
        super().__init__()
        self.embed_model = embed_model

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of image chunks.

        Args:
            x (Tensor): Tensor of shape (n_frames, n_chunks, n_episodes, channels, height, width).

        Returns:
            Tensor: Tensor of shape (n_chunks, n_episodes, embedding_dim).
        """
        raise NotImplementedError


class AvgCLIPEmbed(Embed):
    _base_model: CLIP

    def __init__(self, base_model: CLIP):
        """Generate embeddings for a batch of image chunks
        by averaging the embeddings of all frames in a given chunk.
        """
        self._base_model = base_model
        size = base_model._model.visual.image_size
        image_size: int = size if isinstance(size, int) else size[0]  # type: ignore
        self.transform = image_transform(image_size)

    @torch.inference_mode()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[3] != 3:
            frames = frames.permute(0, 1, 2, 5, 3, 4)
        with torch.no_grad(), autocast("cuda", enabled=torch.cuda.is_available()):
            n_frames, n_chunks, n_episodes, *_ = frames.shape
            frames = rearrange(frames, "n_f n_ch n_e c h w -> (n_f n_ch n_e) c h w")
            # Embed every frame using CLIP
            frame_embed = self._base_model._model.encode_image(frames, normalize=True)
            # Calculate a per-chunk embedding by averaging all frame embeddings of a chunk
            chunk_embed = reduce(
                frame_embed,
                "(n_f n_ch n_e) d -> n_ch n_e d",
                reduction="mean",
                n_f=n_frames,
                n_ch=n_chunks,
                n_e=n_episodes,
            )
        return chunk_embed


class Reward(nn.Module):
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the reward for a batch of chunks of embeddings.

        Args:
            x (Tensor): Tensor of shape (n_chunks, n_episodes, channels, height, width).

        Returns:
            Tensor: Tensor of shape (n_chunks, n_episodes).
        """
        raise NotImplementedError


class ProjectionReward(Reward):
    def __init__(self, baseline, target, direction, projection, alpha):
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)
        self.register_buffer("projection", projection)
        self.alpha = alpha

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        return y

    @staticmethod
    def from_embed(
        target_prompts: list[str],
        baseline_prompts: list[str],
        embed_base: BaseModel,
        alpha: float,
    ) -> "ProjectionReward":
        target = embed_base.embed_text(target_prompts).mean(dim=0, keepdim=True)
        baseline = embed_base.embed_text(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline
        projection = ProjectionReward._compute_projection(direction, alpha)

        return ProjectionReward(baseline, target, direction, projection, alpha)

    @staticmethod
    def _compute_projection(direction: torch.Tensor, alpha: float) -> torch.Tensor:
        projection = direction.T @ direction / torch.norm(direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection


class RewardModel(nn.Module):
    def __init__(
        self,
        embed: Embed,
        reward: Reward,
        window_size: int,
        window_step: int,
        episode_length: int,
    ) -> None:
        super().__init__()
        self.embed = embed
        self.reward = reward

        self.episode_length = episode_length
        self.window_size = window_size
        self.window_step = window_step

    @staticmethod
    def from_config(config: CLIPRewardConfig, episode_length: int) -> "RewardModel":
        model_name_prefix, pretrained = config.pretrained_model.split("/")
        base_model = CLIP(model_name_prefix, pretrained, config.cache_dir)

        if config.embed_type == "avg_frame":
            embed = AvgCLIPEmbed(base_model)
        else:
            raise ValueError(f"Unknown embed_type: {config.embed_type}")

        if config.reward_type == "projection":
            reward = ProjectionReward.from_embed(
                target_prompts=config.target_prompts,
                baseline_prompts=config.baseline_prompts,
                embed_base=base_model,
                alpha=config.alpha,
            )
        else:
            raise ValueError(f"Unknown reward_type: {config.reward_type}")

        return RewardModel(
            embed, reward, config.window_size, config.window_step, episode_length
        )

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the reward for a batch of episodes.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Tensor of shape (batch_size,).
        """
        batch_size = x.shape[0]
        n_episodes = x.shape[0] // self.episode_length
        n_chunks = 1 + (self.episode_length - self.window_size) // self.window_step

        # Un-flatten the batch into episodes
        x = rearrange(
            x,
            "(n_steps n_episodes) ... -> n_s n_e ...",
            n_steps=self.episode_length,
            n_episodes=n_episodes,
        )

        # Unfold the episodes into (potentially overlapping) chunks
        # -> (n_chunks, n_frames, n_episodes, c, h, w)
        x = x.unfold(0, size=self.window_size, step=self.window_step)

        # Rearrange the dimensions to match the expected input shape of the embed model
        x = rearrange(
            x,
            "n_chunks n_frames n_episodes ... -> n_frames n_chunks n_episodes ...",
            n_frames=self.window_size,
            n_chunks=n_chunks,
            n_episodes=n_episodes,
        )

        # Embed the chunks -> (n_chunks, n_episodes, embedding_dim)
        x = self.embed(x)

        # Compute the reward for each chunk -> (n_chunks, n_episodes)
        chunk_rewards = self.reward(x)

        # Assign the reward of each chunk to its last frame
        flat_rewards = rearrange(chunk_rewards, "n_ch n_e -> (n_ch n_e)")

        # TODO: Check what the dimension needs to be
        rewards = torch.zeros(batch_size, device=x.device)

        # Calculate the end indices for each chunk
        indices = (
            torch.arange(n_chunks * n_episodes, device=x.device) * self.window_step
            + self.window_size
            - 1
        )

        assert len(indices) == len(flat_rewards)

        # Assign chunk rewards to the last frame of each chunk
        rewards[indices] = flat_rewards

        return rewards


def compute_rewards(
    model: RewardModel,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute rewards for all frames using the provided reward model.
    Handles splitting into batches and distributing each batch across multiple workers.

    Args:
        model (CLIPReward): reward model
        frames (torch.Tensor): frames to compute rewards for
        batch_size (int): frames will be split into batch_size sized chunks
        num_workers (int): each batch will be split into num_workers chunks
        worker_frames_tensor (Optional[torch.Tensor], optional): no idea what these do, maybe for logging?. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0
    n_samples = len(frames)
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            render_dim = tuple(frames_batch.shape[1:])
            assert len(render_dim) == 3
            rewards_batch = dist_worker_compute_reward(
                rank=0,
                reward_model=model,
                render_dim=render_dim,
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            assert rewards_batch is not None
            rewards_batch = rewards_batch.cpu()
            rewards[i : i + batch_size] = rewards_batch
    return rewards


def dist_worker_compute_reward(
    rank: int,
    reward_model: RewardModel,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: Optional[torch.Tensor] = None,
    worker_frames_tensor: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Compute rewards for a batch of frames using the provided reward model in parallel.

    Args:
        rank (int): the identifier of the current process
        batch_size (int): the batch size here describes the number of frames one process gets

    Returns:
        Optional[torch.Tensor]: the computed rewards, only returned by the master process (rank 0)
    """
    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        if len(frames) != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = [t.cuda(rank) for t in torch.chunk(frames, num_workers, dim=0)]
    else:
        # TODO: Check wheter this should be None or []
        scatter_list = None

    worker_frames = (
        worker_frames_tensor
        if worker_frames_tensor is not None
        else torch.zeros((batch_size, *render_dim), dtype=torch.uint8).cuda(rank)
    )
    dist.scatter(worker_frames, scatter_list=scatter_list, src=0)

    with torch.no_grad():
        rewards = reward_model(worker_frames)

    def zero_t():
        return torch.zeros_like(rewards)

    # TODO: Check wheter this should be None or []
    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else None
    dist.gather(rewards, gather_list=recv_rewards, dst=0)

    if rank == 0:
        assert recv_rewards is not None
        return torch.cat(recv_rewards, dim=0).cuda(rank)
