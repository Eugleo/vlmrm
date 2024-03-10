import base64
import logging
import re
from datetime import datetime
from pathlib import Path

import backoff
import dotenv
import openai
import torch
import vlmrm.reward.rewards as rewards
from torch import Tensor


def logit_reward(frames_enc: Tensor, labels_enc: Tensor, **_kwargs):
    logging.info(f"Computing logit reward, {frames_enc.shape=}, {labels_enc.shape=}")
    scores = rewards.logit_reward(frames_enc, labels_enc, torch.arange(len(labels_enc)))
    logging.info(f"Computed logit reward, {scores.shape=}")
    return scores


def projection_reward(
    frames_enc: Tensor,
    labels_enc: Tensor,
    baselines_enc: Tensor,
    alpha: float,
    **_kwargs,
):
    logging.info(
        f"Computing projection reward, {frames_enc.shape=}, {labels_enc.shape=}, {baselines_enc.shape=}, {alpha=}"
    )
    scores = rewards.projection_reward(frames_enc, baselines_enc, labels_enc, alpha)
    logging.info(f"Computed projection reward, {scores.shape=}")
    return scores


@backoff.on_exception(
    backoff.expo,
    (
        openai.ConflictError,
        openai.APIStatusError,
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.UnprocessableEntityError,
        openai.APIResponseValidationError,
    ),
)
def send_request(client: openai.Client, message, history):
    logging.info(
        f"Sending a new message to GPT-4V: {message if isinstance(message, str) else message[:1] + ['...']}"
    )
    messages = history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(
        model="gpt-4-vision-preview", messages=messages, max_tokens=1200
    )
    reponse_text = response.choices[0].message.content
    logging.info(f"Received response from GPT-4V: {reponse_text}")

    if reponse_text is None:
        raise ValueError(f"Empty response from GPT-4. {messages=}, {response=}")

    return reponse_text, [*messages, {"role": "assistant", "content": reponse_text}]


def _frame_to_payload(image):
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image}",
            "detail": "low",
            "resize": 512,
        },
    }


# def compute_multiclass_metrics(
#     average_similarities: torch.Tensor, true_labels: np.ndarray, verbose: bool
# ) -> Dict[str, Union[float, ConfusionMatrix]]:
#     n_samples, n_classes = average_similarities.shape
#     predictions = np.argmax(average_similarities, axis=1)

#     if verbose:
#         # note: when using prpjection reward, numbers are very large by modulo, seems like a bug
#         print(
#             "in compute_multiclass_metrics: average_similarities\n",
#             average_similarities,
#         )
#         print("=" * 70)
#         print(f"{true_labels.shape=}, {predictions.shape=}")
#         print(f"{true_labels=}, {predictions=}")

#     one_hot_true_labels = np.zeros((n_samples, n_classes))
#     one_hot_true_labels[range(n_samples), true_labels] = 1

#     # random performance will score 0 in adjusted_balanced_accuracy
#     return {
#         "accuracy": accuracy_score(true_labels, predictions),
#         "balanced_accuracy": balanced_accuracy_score(true_labels, predictions),
#         "adjusted_balanced_accuracy": balanced_accuracy_score(
#             true_labels, predictions, adjusted=True
#         ),
#         "roc_auc_ovr_micro": roc_auc_score(
#             one_hot_true_labels,
#             average_similarities,
#             multi_class="ovr",
#             average="micro",
#         ),
#         "confusion_matrix": confusion_matrix(true_labels, predictions).tolist(),
#     }


def gpt4(frames_enc, labels, prompt, log_inputs=False):
    dotenv.load_dotenv()
    client = openai.OpenAI()

    scores = torch.zeros((len(frames_enc), len(labels)))

    logger_path = logging.getLogger().handlers[0].baseFilename  # type: ignore
    log_path = Path(logger_path).parent / "frames"
    if log_inputs:
        log_path.mkdir(parents=True, exist_ok=True)

    for i, video in enumerate(frames_enc):
        logging.info(f"Starting GPT-4V evaluation for video {i + 1}/{len(frames_enc)}")
        if log_inputs:
            for j, frame in enumerate(video):
                time = datetime.now().strftime("%Y%m%d_%H%M%S")
                with open(log_path / f"{time}_v{i+1}_f{j}.jpg", "wb") as f:
                    logging.info(f"Writing frame {j} to {f.name}")
                    f.write(base64.b64decode(frame))

        logging.info(f"Using system prompt: {prompt}")
        history = [{"role": "system", "content": [{"type": "text", "text": prompt}]}]

        heading = {"type": "text", "text": "# TASK"}
        frames = [_frame_to_payload(f) for f in video]
        _, history = send_request(client, [heading, *frames], history)

        classes = "\n".join([f"- {d}" for d in set(labels)])
        scoring_prompt = f"""
Now, given the original frames and your description, score the following potential video descriptions from 0 to 1 based on how well they fit the frames you've seen. Feel free to use values between 0 and 1, too. Usually, there should be exactly one 'correct' description with score 1.

{classes}

Your answer format:
- description: score
"""

        score_answer, history = send_request(client, scoring_prompt, history)

        label_scores = {
            m.group(1): float(m.group(2))
            for m in re.finditer(r"- (.+): ([\d.]+)", score_answer)
        }

        vals = []
        for d in labels:
            vals.append(label_scores.get(d, 1e-6))
            if d not in label_scores:
                print("WARNING: description not found in scores:", d)
                print(f"{history=}")

        scores[i, :] = torch.Tensor([label_scores[d] for d in labels])

    return scores.softmax(dim=1).cpu().numpy()


def subsample(x: torch.Tensor, frames: int) -> torch.Tensor:
    n_frames, *_ = x.shape
    step = n_frames // frames
    x = x[::step, ...][:frames, ...]
    return x
