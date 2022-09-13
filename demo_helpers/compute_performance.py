from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    y_test = dataset.y

    score_label = resolve_score_label(task)
    pretty_table.field_names = [
        "Source",
        score_label,
    ]

    for performance in performance_result:
        prediction = torch.stack(performance.predictions).numpy()
        score = formatted_score(prediction, y_test, inputs=dataset.x, task=task)
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
        dataset.x, groq_model, chip_type="groq", task=task
    )

    host_performance_result = timed_inference_end_to_end_latency(
        dataset.x, pytorch_model, chip_type="cpu"
    )

    result_table = generate_result_comparison_table(
        [host_performance_result, groq_performance_result], dataset, task
    )
    return result_table


def groq_model_inference(x, model, task: Optional[str] = None):
    print("Running inference on GroqChip.")
    pred = model.run_abunch(x)
    if isinstance(pred, torch.Tensor):
        pred = [pred]

    if isinstance(pred[0], tuple):
        if task == "sentence_similarity":
            pred = [p[0] for p in pred]
        else:
            pred = list(map(torch.vstack, pred))
    return pred


def pytorch_model_inference(x, model):
    pred = []
    with torch.no_grad():
        print("Running inference using PyTorch model (CPU).")
        for inputs in tqdm(x):
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
    return pred


def timed_inference_end_to_end_latency(
    x, model, chip_type: str, task: Optional[str] = None
) -> PerformanceResult:
    if chip_type == "groq":
        result = groq_model_inference(x, model, task)
    elif chip_type == "cpu":
        result = pytorch_model_inference(x, model)

    return PerformanceResult(
        name=chip_type,
        batch_size=1,
        total_number_of_samples=len(x),
        predictions=result,
    )
