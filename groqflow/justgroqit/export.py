import inspect
import os
import sys
import warnings
import torch
import onnxflow.justbuildit.stage as stage
import onnxflow.common.exceptions as exp
import onnxflow.common.tensor_helpers as tensor_helpers
import groqflow.common.build as build
import groqflow.common.onnx_helpers as onnx_helpers
import groqflow.common.sdk_helpers as sdk


def _warn_to_stdout(message, category, filename, line_number, _, line):
    sys.stdout.write(
        warnings.formatwarning(message, category, filename, line_number, line)
    )


class CheckOnnxCompatibility(stage.Stage):
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

    def fire(self, state: build.GroqState):

        sdk.check_dependencies(require_devtools=True, exception_type=exp.StageError)

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
            raise exp.StageError(msg)

        return state


class ExportPytorchToTorchScript(stage.Stage):
    """
    Stage that takes a Pytorch module and exports it to TorchScript using
    torch.jit API.

    Expected inputs:
     - state.model is a torch.nn.Module or torch.jit.ScriptModule
     - state.inputs is a dict that represents valid kwargs to the forward
        function of state.model

    Outputs:
     - A *.pt file that implements state.model given state.inputs
    """

    def __init__(self):
        super().__init__(
            unique_name="export_pytorch_to_torch_script",
            monitor_message="Exporting PyTorch to TorchScript",
        )

    @staticmethod
    def _check_model(torch_script_file, success_message, fail_message) -> bool:
        if os.path.isfile(torch_script_file):
            print(success_message)
            return True
        else:
            print(fail_message)
            return False

    def fire(self, state: build.GroqState):
        if not isinstance(state.model, (torch.nn.Module, torch.jit.ScriptModule)):
            msg = f"""
            The current stage (ExportPytorchToTorchScript) is only compatible
            with models of type torch.nn.Module or torch.jit.ScriptModule,
            however the stage received a model of type {type(state.model)}.
            """
            raise exp.StageError(msg)

        if isinstance(state.model, torch.nn.Module):
            # Validate user provided args
            all_args = list(inspect.signature(state.model.forward).parameters.keys())

            for inp in list(state.inputs.keys()):
                if inp not in all_args:
                    msg = f"""
                    Input name {inp} not found in the model's forward method. Available
                    input names are: {all_args}"
                    """
                    raise ValueError(msg)

        # Send torch export warnings to stdout (and therefore the log file)
        # so that they don't fill up the command line
        default_warnings = warnings.showwarning
        warnings.showwarning = _warn_to_stdout

        # Export the model to TorchScript
        jit_module = torch.jit.trace(
            state.model,
            example_kwarg_inputs=state.inputs,
        )

        # Save model to disk
        os.makedirs(state.torch_script_dir, exist_ok=True)
        jit_module.save(state.torch_script_file)

        # Save output names to ensure we are preserving the order of the outputs.
        # We have to re-load the torchscript module because the output names
        # will change during serialization.
        loaded_jit_module = torch.jit.load(state.torch_script_file)
        state.expected_output_names = [
            output.debugName() for output in loaded_jit_module.graph.outputs()
        ]

        # Restore default warnings behavior
        warnings.showwarning = default_warnings

        tensor_helpers.save_inputs(
            [state.inputs], state.original_inputs_file, downcast=False
        )

        # Check the if the base mode has been exported successfully
        success_msg = "\tSuccess exporting model to TorchScript"
        fail_msg = "\tFailed exporting model to TorchScript"
        state.info.torch_script_exported = self._check_model(
            state.torch_script_file, success_msg, fail_msg
        )

        if state.info.torch_script_exported:
            state.intermediate_results = [state.torch_script_file]
        else:
            msg = f"""
            Unable to export model to TorchScript using Torch's jit exporter.
            We recommend that you modify your model until it is
            compatible with this third party software, then re-run.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state
