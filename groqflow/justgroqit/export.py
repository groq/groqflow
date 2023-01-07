import os
import inspect
import shutil
import re
import warnings
import sys
import copy
from typing import Union
import torch
import onnxruntime
import onnxmltools
import onnx
import groqflow.justgroqit.stage as stage
import groqflow.common.exceptions as exp
import groqflow.common.build as build
import groqflow.common.tensor_helpers as tensor_helpers
import groqflow.common.onnx_helpers as onnx_helpers
import groqflow.common.sdk_helpers as sdk
import groqflow.common.quantization_helpers as quant_helpers

try:
    import tensorflow as tf
    import tf2onnx
except ModuleNotFoundError as module_error:
    raise exp.GroqitEnvError(
        "GroqFlow added a dependence on tensorflow and tf2onnx in version 2.1.2. "
        "You must install tensorflow and tf2onnx to continue."
    )


def _check_model(onnx_file, success_message, fail_message) -> bool:
    if os.path.isfile(onnx_file):
        print(success_message)
    else:
        print(fail_message)
        return False
    try:
        onnx.checker.check_model(onnx_file)
        print("\tSuccessfully checked onnx file")
        return True
    except onnx.checker.ValidationError as e:
        print("\tError while checking generated ONNX file")
        print(e)
        return False


def get_output_names(onnx_model: Union[str, onnx.ModelProto]):
    # Get output names of ONNX file/model
    if not isinstance(onnx_model, onnx.ModelProto):
        onnx_model = onnx.load(onnx_model)
    return [node.name for node in onnx_model.graph.output]  # pylint: disable=no-member


class ReceiveOnnxModel(stage.GroqitStage):
    """
    Stage that takes an ONNX model as input.

    Expected inputs:
     - state.model is a path to the ONNX model
     - state.inputs is a dict that represents valid inputs for the onnx model

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs.
    """

    def __init__(self):
        super().__init__(
            unique_name="receive_onnx",
            monitor_message="Receiving ONNX Model",
        )

    def fire(self, state: build.State):
        if not isinstance(state.model, str):
            msg = f"""
            The current stage (ReceiveOnnxModel) is only compatible with
            ONNX files, however the stage received a model of type
            {type(state.model)}.
            """
            raise exp.GroqitStageError(msg)
        if not state.model.endswith(".onnx"):
            msg = f"""
            The current stage (ReceiveOnnxModel) expects a path to ONNX
            model, however the stage received {state.model}.
            """
            raise exp.GroqitStageError(msg)

        dummy_inputs = tuple(state.inputs.values())
        dummy_input_names = tuple(state.inputs.keys())
        state.inputs = dict(zip(dummy_input_names, dummy_inputs))

        model = onnx.load(state.model)
        opset_str = str(model.opset_import)  # pylint: disable=no-member
        opset = int(re.search(r"\d+", opset_str).group())
        input_shapes = [
            [d.dim_value for d in _input.type.tensor_type.shape.dim]
            for _input in model.graph.input  # pylint: disable=no-member
        ]

        # Save output node names
        state.expected_output_names = get_output_names(model)

        # Check for Dynamic shapes in the model. They can be represented as 0, -1, "unk__".
        for input in input_shapes:
            for dimension in input:
                if dimension < 1 or not isinstance(dimension, int):
                    msg = f"""
                    The received model has dynamic input dimensions. Please freeze the model with static
                    input dimensions.
                    More information may be available in the log file at **{self.logfile_path}**
                    """
                    raise exp.GroqitStageError(msg)

        if opset < build.DEFAULT_ONNX_OPSET and opset >= build.MINIMUM_ONNX_OPSET:
            print(
                " \n The received model has an opset {opset}. Though this opset is supported \
                we recommend upgrading the model to opset {build.MINIMUM_ONNX_OPSET}"
            )
        elif opset < build.MINIMUM_ONNX_OPSET:
            msg = f"""
            The received model has an opset {opset}. Opset < 11 is not supported. Please
            try upgrading the model to opset 13.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        shutil.copy(state.model, state.base_onnx_file)

        # Save inputs and convert to fp16 and int32 (no int64 nor float32/64)
        to_downcast = False if state.quantization_samples else True
        tensor_helpers.save_inputs(
            [state.inputs], state.original_inputs_file, downcast=to_downcast
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess receiving ONNX Model"
        fail_msg = "\tFailed receiving ONNX Model"
        state.info.base_onnx_exported = _check_model(
            state.base_onnx_file, success_msg, fail_msg
        )

        if state.info.base_onnx_exported:
            state.intermediate_results = [state.base_onnx_file]
        else:
            msg = f"""
            Unable to process ONNX Model. We recommend that you verify the source of the model.
            Any optimizations performed on the model could result in an error.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        return state


class ExportPytorchModel(stage.GroqitStage):
    """
    Stage that takes a PyTorch model instance, in state.model, and
    exports it to an ONNX file.

    Expected inputs:
     - state.model is a torch.nn.Module or torch.jit.ScriptModule
     - state.inputs is a dict that represents valid kwargs to the forward
        function of state.model

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs
    """

    def __init__(self):
        super().__init__(
            unique_name="export_pytorch",
            monitor_message="Exporting PyTorch to ONNX",
        )

    def fire(self, state: build.State):
        if not isinstance(state.model, (torch.nn.Module, torch.jit.ScriptModule)):
            msg = f"""
            The current stage (ExportPytorchModel) is only compatible with
            models of type torch.nn.Module or torch.jit.ScriptModule, however
            the stage received a model of type {type(state.model)}.
            """
            raise exp.GroqitStageError(msg)

        # TODO: check that state.inputs is valid
        # https://git.groq.io/code/Groq/-/issues/13947

        # The `torch.onnx.export()` function accepts a tuple of positional inputs
        # followed by a dictionary with all keyword inputs.
        # The dictionary must be last item in tuple.
        user_provided_args = list(state.inputs.keys())

        if isinstance(state.model, torch.nn.Module):
            # Validate user provided args
            all_args = list(inspect.signature(state.model.forward).parameters.keys())

            for inp in user_provided_args:
                if inp not in all_args:
                    msg = f"""
                    Input name {inp} not found in the model's forward method. Available
                    input names are: {all_args}"
                    """
                    raise ValueError(msg)

            # Most pytorch models have args that are kind = positional_or_keyword.
            # The `torch.onnx.export()` function accepts model args as
            #     (all_positional_args_value,{keyword_arg:value}).
            # To map the input_args correctly and to build an accurate model
            # the order of the input_names must reflect the order of the model args.

            # Collect order of pytorch model args.
            all_args_order_mapping = {arg: idx for idx, arg in enumerate(all_args)}

            # Sort the user provided inputs with respect to model args and store as tuple.
            sorted_user_inputs = sorted(
                user_provided_args, key=lambda x: all_args_order_mapping[x]
            )
            dummy_input_names = tuple(sorted_user_inputs)

            # If a single input is provided torch.onnx.export will
            # not accept a dictionary, so pop the first arg
            user_args = copy.deepcopy(state.inputs)
            first_input = user_args.pop(dummy_input_names[0])

            # Create tuple: (first input, {rest of user_args dict as keyword args})
            dummy_inputs = (first_input, user_args)

        # TODO: Test with optional inputs
        # https://git.groq.io/code/Groq/-/issues/14581

        else:  # state.model is a torch.jit.ScriptModule
            dummy_inputs = tuple(state.inputs.values())

            # Collect input names
            dummy_input_names = tuple(state.inputs.keys())

        state.info.opset = build.DEFAULT_ONNX_OPSET

        # Send torch export warnings to stdout (and therefore the log file)
        # so that they don't fill up the command line
        def warn_to_stdout(message, category, filename, line_number, _, __):
            sys.stdout.write(
                warnings.formatwarning(message, category, filename, line_number)
            )

        default_warnings = warnings.showwarning
        warnings.showwarning = warn_to_stdout

        # Export the model to ONNX
        torch.onnx.export(
            state.model,
            dummy_inputs,
            state.base_onnx_file,
            input_names=dummy_input_names,
            do_constant_folding=True,
            opset_version=state.info.opset,
            verbose=False,
        )

        # Save output names to ensure we are preserving the order of the outputs
        state.expected_output_names = get_output_names(state.base_onnx_file)

        # Restore default warnings behavior
        warnings.showwarning = default_warnings

        # Save inputs and convert to fp16 and int32 (no int64 nor float32/64)
        to_downcast = False if state.quantization_samples else True
        print("to_downcast: ", to_downcast)
        tensor_helpers.save_inputs(
            [state.inputs], state.original_inputs_file, downcast=to_downcast
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess exporting model to ONNX"
        fail_msg = "\tFailed exporting model to ONNX"
        state.info.base_onnx_exported = _check_model(
            state.base_onnx_file, success_msg, fail_msg
        )

        if state.info.base_onnx_exported:
            state.intermediate_results = [state.base_onnx_file]
        else:
            msg = f"""
            Unable to export model to ONNX using Torch's ONNX exporter.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run groqit().
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        return state


class ExportKerasModel(stage.GroqitStage):
    """
    Stage that takes a Keras model instance, in state.model, and
    exports it to an ONNX file.

    Expected inputs:
     - state.model is a tf.keras.Model
     - state.inputs is a dict that represents valid kwargs to the forward
        function of state.model

    Outputs:
     - A *-base.onnx file that implements state.model given state.inputs
    """

    def __init__(self):
        super().__init__(
            unique_name="export_keras",
            monitor_message="Exporting Keras to ONNX",
        )

    def fire(self, state: build.State):
        if not isinstance(state.model, (tf.keras.Model)):
            msg = f"""
            The current stage (ExportKerasModel) is only compatible with
            models of type tf.keras.Model, however
            the stage received a model of type {type(state.model)}.
            """
            raise exp.GroqitStageError(msg)

        user_provided_args = state.inputs.keys()

        all_args = []

        # Check the model inputs member
        if state.model.inputs:
            all_args = [x.name for x in state.model.inputs]

        # If the input name(s) cannot be extracted from the inputs variable
        # than try to find them in the call() method
        if len(all_args) == 0:
            all_args = list(inspect.signature(state.model.call).parameters.keys())

        inputs = []
        input_names = []

        for inp in user_provided_args:
            if inp not in all_args:
                msg = f"""
                Input name {inp} not found in the model's forward method. Available
                input names are: {all_args}"
                """
                raise ValueError(msg)

        for _, arg in enumerate(all_args):
            if arg in user_provided_args:
                inputs.append(state.inputs[arg])
                input_names.append(arg)

        input_specs = []
        for inp, name in zip(inputs, input_names):
            dtype = inp.dtype
            shape = inp.shape
            if inp.dtype == tf.float64:
                print(f"Converting input {name} from float64 to float32")
                dtype = tf.float32
            if inp.dtype == tf.int64:
                print(f"Converting input {name} from int64 to int32")
                dtype = tf.int32
            if inp.shape[0] is None:
                print("Found batch size None and setting it to 1")
                shape = (1, shape[1:])

            input_specs.append(tf.TensorSpec(shape, dtype, name))

        state.info.opset = build.DEFAULT_ONNX_OPSET

        # Export the model to ONNX
        tf2onnx.convert.from_keras(
            state.model,
            input_signature=input_specs,
            opset=state.info.opset,
            output_path=state.base_onnx_file,
        )

        # Save output names to ensure we are preserving the order of the outputs
        state.expected_output_names = get_output_names(state.base_onnx_file)

        state.inputs = dict(zip(tuple(input_names), tuple(inputs)))

        # Save inputs and convert to fp16 and int32 (no int64 nor float32/64)
        to_downcast = False if state.quantization_samples else True
        tensor_helpers.save_inputs(
            [state.inputs], state.original_inputs_file, downcast=to_downcast
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess exporting model to ONNX"
        fail_msg = "\tFailed exporting model to ONNX"
        state.info.base_onnx_exported = _check_model(
            state.base_onnx_file, success_msg, fail_msg
        )

        if state.info.base_onnx_exported:
            state.intermediate_results = [state.base_onnx_file]
        else:
            msg = f"""
            Unable to export model to ONNX using tf2onnx exporter.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run groqit().
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        return state


class OptimizeOnnxModel(stage.GroqitStage):
    """
    Stage that takes an ONNX file and uses ONNX Runtime to optimize it.
    Important because this helps to perform constant folding, Redundant
    node eliminations, Semantics-preserving node fusions

    Expected inputs:
     - state.intermediate_results contains a single .onnx file

    Outputs:
     - A *-opt.onnx file
    """

    def __init__(self):
        super().__init__(
            unique_name="optimize_onnx",
            monitor_message="Optimizing ONNX file",
        )

    def fire(self, state: build.State):

        # TODO: validate this input
        # https://git.groq.io/code/Groq/-/issues/13947
        input_onnx = state.intermediate_results[0]

        # Perform some basic optimizations on the model to remove shape related
        # information inserted for dynamic shape inference.
        # Given that we're compiling against a fixed sequence length the dynamic
        # shape information is not necessary
        session_options = onnxruntime.SessionOptions()

        # Set graph optimization level
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )

        # To enable model serialization after graph optimization set this
        session_options.optimized_model_filepath = state.opt_onnx_file

        # Optimize graph
        onnxruntime.InferenceSession(input_onnx, session_options)

        # Check that the converted model is still valid
        success_msg = "\tSuccess optimizing ONNX model"
        fail_msg = "\tFailed optimizing ONNX model"
        state.info.opt_onnx_exported = _check_model(
            state.opt_onnx_file, success_msg, fail_msg
        )

        if state.info.opt_onnx_exported:
            state.intermediate_results = [state.opt_onnx_file]
        else:
            msg = f"""
            Unable to optimize ONNX file using ONNX runtime.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run groqit().
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        return state


class CheckOnnxCompatibility(stage.GroqitStage):
    """
    Stage that takes an ONNX file, checks whether it is compatible
    with Groq Compiler, and raises an exception if the ONNX file is
    not compatible.

    Expected inputs:
     - state.intermediate_results contains a single .onnx file

    Outputs:
     - The same ONNX file as the input
    """

    def __init__(self):
        super().__init__(
            unique_name="check_compatibility",
            monitor_message="Checking for Op support",
        )

    def fire(self, state: build.State):

        sdk.check_dependencies(
            require_devtools=True, exception_type=exp.GroqitStageError
        )

        # TODO: validate this input
        # https://git.groq.io/code/Groq/-/issues/13947
        input_onnx = state.intermediate_results[0]

        (
            state.info.opt_onnx_ops,
            state.info.opt_onnx_unsupported_ops,
        ) = onnx_helpers.check_ops(input_onnx, state.use_sdk)
        print(f"Model has {len(state.info.opt_onnx_unsupported_ops)} unsupported ops")

        state.info.opt_onnx_all_ops_supported = (
            len(state.info.opt_onnx_unsupported_ops) == 0
            and len(state.info.opt_onnx_ops) != 0
        )

        if not state.info.opt_onnx_all_ops_supported:
            ops = ", ".join(state.info.opt_onnx_unsupported_ops)
            msg = f"""
            You model contains ONNX operation(s) that are not supported by Groq Compiler:
            **{ops}**
            Please replace these operation(s) in your model or contact
            sales@groq.com to request improved operation support in Groq Compiler.
            """
            raise exp.GroqitStageError(msg)

        return state


class ConvertOnnxToFp16(stage.GroqitStage):
    """
    Stage that takes an ONNX file and converts its trained parameters
    to fp16.

    Expected inputs:
     - state.intermediate_results contains a single .onnx file

    Outputs:
     - A *-f16.onnx file with FP16 trained parameters
    """

    def __init__(self):
        super().__init__(
            unique_name="fp16_conversion",
            monitor_message="Converting to FP16",
        )

    def fire(self, state: build.State):

        # TODO: validate this input
        # https://git.groq.io/code/Groq/-/issues/13947
        input_onnx = state.intermediate_results[0]

        # Convert the model to FP16
        # Some ops will not be converted to fp16 because they are in a block list
        # The latest list can be found here. It is not neccesarily the list that
        # our version of onnxmltools sees
        # https://github.com/microsoft/onnxconverter-common/blob/master/onnxconverter_common/float16.py#L82

        # Legalize ops are ops that have been or are currently in the block list
        # that we explicitly want removed
        legalize_ops = ["InstanceNormalization", "Resize", "Max"]
        op_block_list = onnxmltools.utils.float16_converter.DEFAULT_OP_BLOCK_LIST.copy()
        for op in legalize_ops:
            # Check to see that they are not in the block list before we remove them
            # Neccesary because the block list may be updated, and not in the state we expect
            if op in op_block_list:
                op_block_list.remove(op)

        # Infer shapes before converting to FP16 to enable models with >2GB
        onnx.shape_inference.infer_shapes_path(input_onnx)

        fp32_model = onnx.load_model(input_onnx)
        fp16_model = onnxmltools.utils.float16_converter.convert_float_to_float16(
            fp32_model, op_block_list=op_block_list, disable_shape_infer=True
        )

        # Save FP16 model (use external data format if needed)
        output_path = state.converted_onnx_file
        try:
            onnxmltools.utils.save_model(fp16_model, output_path)
        except ValueError:
            onnx.save_model(fp16_model, output_path, save_as_external_data=True)

        # Check that the converted model is still valid
        success_msg = "\tSuccess converting ONNX model to fp16"
        fail_msg = "\tFailed converting ONNX model to fp16"
        state.info.converted_onnx_exported = _check_model(
            state.converted_onnx_file, success_msg, fail_msg
        )

        if state.info.converted_onnx_exported:
            state.intermediate_results = [output_path]
        else:
            msg = f"""
            Attempted to use onnxmltools, a third party library, to convert your
            model to the float16 datatype, however this operation was not successful.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        return state


class QuantizeONNXModel(stage.GroqitStage):
    """
    Stage that takes an ONNX model and a dataset of quantization samples as inputs,
    and performs static post-training quantization to the model to int8 precision.

    Expected inputs:
     - state.model is a path to the ONNX model
     - state.quantization_dataset is a dataset that is used for static quantization

    Outputs:
     - A *_quantized.onnx file => the quantized onnx model.
    """

    def __init__(self):
        super().__init__(
            unique_name="quantize_onnx",
            monitor_message="Quantizing ONNX model",
        )

    def fire(self, state: build.State):
        input_path = state.intermediate_results[0]
        output_path = state.quantized_onnx_file

        quant_helpers.quantize(
            input_file=input_path,
            data=state.quantization_samples,
            output_file=output_path,
        )

        # Check that the converted model is still valid
        success_msg = "\tSuccess quantizing ONNX model to int8"
        fail_msg = "\tFailed quantizing ONNX model to int8"
        state.info.quantized_onnx_exported = _check_model(
            state.quantized_onnx_file, success_msg, fail_msg
        )

        if state.info.quantized_onnx_exported:
            state.intermediate_results = [output_path]
        else:
            msg = f"""
            Attempted to use {state.quantization_dataset} to statically quantize
            model to int8 datatype, however this operation was not successful.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        return state
