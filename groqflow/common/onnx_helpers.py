"""
Helper functions for dealing with ONNX files and ONNX models
"""

import subprocess
import ast
import onnxflow.common.printing as printing
import groqflow.common.sdk_helpers as sdk


def check_ops(input_onnx, use_sdk=False):

    print("Checking unsupported ops...")

    # Select either bake or SDK
    if use_sdk:
        cmd = sdk.find_tool("onnxmodelanalyzer")
    else:
        cmd = [
            "bake",
            "r",
            "//Groq/Compiler:OnnxModelAnalyze",
        ]
    cmd = cmd + ["-u", "-i", input_onnx]

    # Run process and decode outputs
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = p.communicate()
    out = out.decode("utf-8").split("\n")
    all_ops = ast.literal_eval(out[-4])
    unsupported_ops = ast.literal_eval(out[-2])

    # print results accordingly
    num_ops = len(all_ops)
    num_unsupported = len(unsupported_ops)
    num_supported = num_ops - num_unsupported
    if num_unsupported == 0:
        printing.logn("\t\tDONE", printing.Colors.OKGREEN)
        printing.logn(
            "\t" + f"{num_supported}/{num_ops} ops supported", printing.Colors.OKGREEN
        )
    else:
        printing.logn("\t\tDONE", printing.Colors.OKGREEN)
        printing.logn(
            "\t" + f"{num_supported}/{num_ops} ops supported", printing.Colors.WARNING
        )
        printing.logn(
            "\tUnsupported ops: " + ", ".join(unsupported_ops),
            printing.Colors.WARNING,
        )
    return all_ops, unsupported_ops
