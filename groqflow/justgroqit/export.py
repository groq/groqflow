import onnxflow.justbuildit.stage as stage
import onnxflow.common.exceptions as exp
import groqflow.common.build as build
import groqflow.common.onnx_helpers as onnx_helpers
import groqflow.common.sdk_helpers as sdk


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
