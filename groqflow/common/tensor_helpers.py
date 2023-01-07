"""
Helper functions for dealing with tensors
"""

import os
import copy
import torch
import numpy as np
import groqflow.common.exceptions as exp
import groqflow.common.build as build

try:
    import tensorflow as tf
except ModuleNotFoundError as module_error:
    raise exp.GroqitEnvError(
        "GroqFlow added a dependence on tensorflow in version 2.1.2. "
        "You must install tensorflow to continue."
    )

# Checks whether a given input has the expected shape
def check_shapes_and_dtypes(inputs, expected_shapes, expected_dtypes):
    current_shapes, current_dtypes = build.get_shapes_and_dtypes(inputs)
    if not expected_shapes == current_shapes:
        msg = f"""
        Groq Model compiled to always take input of shape
        {expected_shapes} but got {current_shapes}
        """
        raise exp.GroqFlowError(msg)
    elif not expected_dtypes == current_dtypes:
        msg = f"""
        Groq Model compiled to always take input of types
        {expected_dtypes} but got {current_dtypes}
        """
        raise exp.GroqFlowError(msg)


def save_inputs(inputs, inputs_file, input_dtypes=None, downcast=True):

    # Convert inputs to fp16 and int32
    inputs_converted = copy.deepcopy(inputs)
    for i in range(len(inputs_converted)):
        inputs_converted[i] = {
            k: v for k, v in inputs_converted[i].items() if v is not None
        }
        for k in inputs_converted[i].keys():
            if not hasattr(inputs_converted[i][k], "dtype"):
                continue
            if torch.is_tensor(inputs_converted[i][k]):
                inputs_converted[i][k] = inputs_converted[i][k].cpu().detach().numpy()
            if tf.is_tensor(inputs_converted[i][k]):
                inputs_converted[i][k] = inputs_converted[i][k].numpy()
            if downcast:
                if input_dtypes is not None and input_dtypes[k] is not None:
                    inputs_converted[i][k] = inputs_converted[i][k].astype(
                        input_dtypes[k]
                    )
                    continue
                if (
                    inputs_converted[i][k].dtype == np.float32
                    or inputs_converted[i][k].dtype == np.float64
                ):
                    inputs_converted[i][k] = inputs_converted[i][k].astype("float16")
                if inputs_converted[i][k].dtype == np.int64:
                    inputs_converted[i][k] = inputs_converted[i][k].astype("int32")

    # Save models inputs to file for later profiling
    if os.path.isfile(inputs_file):
        os.remove(inputs_file)
    np.save(inputs_file, inputs_converted)

    return inputs_converted
