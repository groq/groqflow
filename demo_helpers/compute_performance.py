from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime
from prettytable import PrettyTable
from tqdm import tqdm
import torch

from demo_helpers.dataset import Dataset, create_dataset
from demo_helpers.validate import formatted_score, resolve_score_label


@dataclass
class PerformanceResult:
    name: str
    batch_size: int
    total_number_of_samples: int
    predictions: List = field(repr=False)


def generate_result_comparison_table(
    performance_result: List[PerformanceResult], dataset: Dataset, task: str
) -> List[Tuple]:
    pretty_table = PrettyTable()
    row_data = []

    score_label = resolve_score_label(task)
    pretty_table.field_names = [
        "Source",
        score_label,
    ]

    for performance in performance_result:
        if isinstance(performance.predictions[0], torch.Tensor):
            prediction = torch.stack(performance.predictions).numpy()
        else:
            prediction = np.concatenate(performance.predictions, axis=0)
        score = formatted_score(prediction, dataset, task=task)
        row_data.append(
            (
                performance.name,
                score,
            )
        )

    for row in row_data:
        pretty_table.add_row(row)

    print(pretty_table)

    return row_data


def compute_performance(
    groq_model,
    pytorch_model,
    dataset,
    tokenizer=None,
    max_seq_length=None,
    feature_extractor=None,
    task=None,
):
    print("Preprocessing data.")
    input_names = list(groq_model.state.expected_input_shapes.keys())
    dataset = create_dataset(
        dataset,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        feature_extractor=feature_extractor,
        input_names=input_names,
    )

    groq_performance_result = timed_inference_end_to_end_latency(
        dataset, groq_model, chip_type="groq", task=task
    )

    host_performance_result = timed_inference_end_to_end_latency(
        dataset, pytorch_model, chip_type="cpu"
    )

    result_table = generate_result_comparison_table(
        [host_performance_result, groq_performance_result], dataset, task
    )
    return result_table


def groq_model_inference(dataset, model, task: Optional[str] = None):
    print("Running inference on GroqChip.")
    pred = model.run_abunch(dataset.x)
    if isinstance(pred, torch.Tensor):
        pred = [pred]

    if isinstance(pred[0], tuple):
        if task == "sentence_similarity":
            pred = [p[0] for p in pred]
        else:
            pred = list(map(torch.vstack, pred))
    return dataset.postprocess(pred)


def onnx_model_inference(dataset, model):
    print("Running inference on CPU (ONNX).")
    session = onnxruntime.InferenceSession(model)
    result = []
    for inputs in tqdm(dataset.x):
        out = session.run(None, inputs)
        if len(out) == 1:
            result.append(torch.tensor(out[0]))
        else:
            result.append(tuple([torch.tensor(out[i]) for i in range(len(out))]))
    return dataset.postprocess(result)


def pytorch_model_inference(dataset, model):
    pred = []
    with torch.no_grad():
        print("Running inference using PyTorch model (CPU).")
        for inputs in tqdm(dataset.x):
            out = model(**inputs)
            if not isinstance(out, torch.Tensor):
                if "logits" in out:
                    out = out.logits
                elif "start_logits" in out and "end_logits" in out:
                    out = torch.vstack((out["start_logits"], out["end_logits"]))
                elif "last_hidden_state" in out:
                    out = out.last_hidden_state
                else:
                    raise ValueError(
                        "Unknown output key. List of keys:", list(out.keys())
                    )
            pred.append(out)
    return dataset.postprocess(pred)


def timed_inference_end_to_end_latency(
    dataset, model, chip_type: str, task: Optional[str] = None
) -> PerformanceResult:
    if chip_type == "groq":
        result = groq_model_inference(dataset, model, task)
    elif chip_type == "cpu":
        if isinstance(model, str):  # ONNX
            result = onnx_model_inference(dataset, model)
        else:
            result = pytorch_model_inference(dataset, model)

    return PerformanceResult(
        name=chip_type,
        batch_size=1,
        total_number_of_samples=len(dataset.x),
        predictions=result,
    )
