import timeit
from functools import partial

# pylint: disable=import-error, no-name-in-module
from termgraph.termgraph import chart
from prettytable import PrettyTable, PLAIN_COLUMNS

import torch


def easydemo(input_dict, groq_model, pytorch_model, repetitions=100):

    print()
    data_table = PrettyTable()
    data_table.set_style(PLAIN_COLUMNS)
    data_table.align = "l"
    data_table.field_names = ["Input Name", "Shape", "Data"]

    def formatlist(x):
        fmtstr = "{:.4f}" if torch.is_floating_point(x) else "{}"
        x = x.flatten()[:5].tolist()
        x = [fmtstr.format(itm) for itm in x]
        if len(x) >= 5:
            x += ["..."]
        return ", ".join(x)

    for k, v in input_dict.items():
        data_table.add_row((f'"{k}"', tuple(v.shape), formatlist(v)))

    print(data_table)
    print()
    print(
        f"Running {type(pytorch_model).__name__} model across {repetitions} repetitions ..."
    )

    print("Running on GroqChip...")
    #    groq_outputs = groq_model(**input_dict)
    groq_time = groq_model.benchmark(input_dict, repetitions=repetitions).latency

    print("Running on CPU...")
    #    cpu_outputs = pytorch_model(**input_dict)
    cpu_time = (
        timeit.timeit(partial(pytorch_model, **input_dict), number=repetitions)
        / repetitions
    )

    print("Running on A100...")
    pytorch_model.to("cuda")
    input_dict = {k: v.to("cuda") for k, v in input_dict.items()}
    #    gpu_outputs = pytorch_model(**input_dict)
    gpu_time = (
        timeit.timeit(partial(pytorch_model, **input_dict), number=repetitions)
        / repetitions
    )

    print()
    print("Mean End-to-End latency:")
    print()

    labels = ["CPU", "A100", "Groq"]
    data = [cpu_time, gpu_time, groq_time]
    labels = [f"{label:8s}{d:.6f} sec" for label, d in zip(labels, data)]
    data = [[d] for d in data]
    line_length = 120
    args = {
        "stacked": False,
        "width": line_length - len(labels[0]) - 2,
        "no_labels": False,
        "format": "{}",
        "suffix": "",
        "vertical": False,
        "histogram": False,
        "no_values": True,
        "different_scale": False,
    }
    chart(colors=None, data=data, args=args, labels=labels)
    print("Groq speedup vs CPU: {:.2f}x".format(cpu_time / groq_time))
    print("Groq speedup vs GPU: {:.2f}x".format(gpu_time / groq_time))
    print()


#    gpu_error = torch.mean(torch.abs((gpu_outputs.cpu() - cpu_outputs) / cpu_outputs))
#    print("GPU Error", gpu_error)
#    groq_error = torch.mean(torch.abs((groq_outputs - cpu_outputs) / cpu_outputs))
#    print("Groq Error", groq_error)
