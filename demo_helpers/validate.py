import re
import string
from collections import Counter
from typing import List
from datasets import load_metric
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import paired_cosine_distances
from scipy.stats import spearmanr


def formatted_score(
    pred, test, ids=None, tokenizer=None, inputs=None, task="classification"
):
    sc = score(pred, test, ids=ids, tokenizer=tokenizer, inputs=inputs, task=task)
    if task in ["classification", "qa", "ner", "keyword_spotting"]:
        sc = f"{sc:.2%}"
    elif task in ["regression", "sentence_similarity"]:
        sc = f"{sc:.4f}"
    elif task == "semantic_segmentation":
        sc = sc["mean_iou"]
        sc = f"{sc:.4f}"
    else:
        raise Exception(f"unrecognized task: {task}")

    return sc


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    From official SQuAD evaluation script.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """From official SQuAD evaluation script."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """From official SQuAD evaluation script."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def score(pred, test, ids=None, tokenizer=None, inputs=None, task="classification"):
    if task == "classification":
        sc = np.mean(pred.argmax(axis=-1).reshape(test.shape) == test)
    elif task == "keyword_spotting":
        sc = np.equal(pred.argmax(axis=-1).ravel(), test).mean()
    elif task == "ner":
        # unroll gt labels across time steps
        flat_test = np.array(test).ravel()

        # get best label for each time step
        pred_labels = np.argmax(pred, -1)
        # unroll pred labels across time steps
        flat_preds = pred_labels.ravel()

        # all samples are padded to max_seq_len. reduce to valid
        # time steps only
        valid_indices = flat_test >= 0
        flat_test, flat_preds = flat_test[valid_indices], flat_preds[valid_indices]

        # calculate score
        sc = np.equal(flat_preds, flat_test).mean()
    elif task == "regression":
        sc = np.mean(np.square(test - pred))
    elif task == "qa":
        pred = pred.argmax(axis=-1)

        def answers(y):
            return [
                tokenizer.decode(id[start:end])
                for (id, start, end) in zip(ids, y[:, 0], y[:, 1])
            ]

        pred = answers(pred)

        sc = np.mean(
            [
                metric_max_over_ground_truths(f1_score, p, t)
                for (p, t) in zip(pred, test)
            ]
        )
    elif task == "semantic_segmentation":
        sc = calculate_miou_score(pred, test)
    elif task == "sentence_similarity":
        sc = calculate_spearman_correlation(pred, test, inputs)
    else:
        raise Exception(f"Unrecognized task: {task}")
    return sc


def resolve_score_label(task: str) -> str:
    if task in ["classification", "ner", "keyword_spotting"]:
        label = "Accuracy"
    elif task == "regression":
        label = "MSE"
    elif task == "qa":
        label = "F1 Score"
    elif task == "semantic_segmentation":
        label = "Mean IoU"
    elif task == "sentence_similarity":
        label = "Spearman Rank Correlation Coefficient"
    else:
        raise Exception(f"Unrecognized task: {task}")
    return label


def calculate_miou_score(pred: List, test: List):
    metric = load_metric("mean_iou")

    upsample_size = test[0].shape[-2:]
    num_labels = pred[0].shape[1]

    for p, t in zip(pred, test):
        p = _upsample_logits(torch.tensor(p), upsample_size).squeeze()
        t = t.squeeze()
        metric.add(prediction=p, reference=t)

    score = metric.compute(
        num_labels=num_labels,
        ignore_index=255,
        reduce_labels=False,
    )
    return score


def calculate_spearman_correlation(pred, test, encoded_input):
    sentence_1_embeddings = []
    sentence_2_embeddings = []
    for p, i in zip(pred, encoded_input):
        p = torch.tensor(p)

        sentence_embeddings = _mean_pooling(p, i["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_1_embeddings.append(sentence_embeddings[0].reshape(1, -1))
        sentence_2_embeddings.append(sentence_embeddings[1].reshape(1, -1))

    cosine_scores = 1 - (
        paired_cosine_distances(
            torch.stack(sentence_1_embeddings).squeeze(),
            torch.stack(sentence_2_embeddings).squeeze(),
        )
    )

    spearman_cosine, _ = spearmanr(test, cosine_scores)

    return spearman_cosine


def _upsample_logits(logits, size):
    return F.interpolate(
        logits.double(),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)


def _mean_pooling(model_output, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(model_output.shape).float()
    )

    return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def formatted_ips(ips):
    return f"{ips:.2f}"
