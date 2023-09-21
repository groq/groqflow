from typing import Optional, List, Tuple, Union, Dict, Any
from collections.abc import Collection
from typeguard import typechecked

import onnxflow.common.build as of_build
import onnxflow.common.exceptions as exp
import onnxflow.justbuildit.export as of_export
import onnxflow.justbuildit.hummingbird as hummingbird
import onnxflow.justbuildit.stage as stage
import onnxflow.justbuildit.ignition as of_ignition

import groqflow.common.build as build
import groqflow.justgroqit.compile as compile
import groqflow.justgroqit.export as gf_export


from groqflow.version import __version__ as groqflow_version

default_pytorch_export_sequence = stage.Sequence(
    "default_pytorch_export_sequence",
    "Exporting PyTorch Model",
    [
        of_export.ExportPytorchModel(),
        of_export.OptimizeOnnxModel(),
        gf_export.CheckOnnxCompatibility(),
        of_export.ConvertOnnxToFp16(),
    ],
)

default_pytorch_sequence = stage.Sequence(
    "default_pytorch_sequence",
    "Building PyTorch Model",
    [
        default_pytorch_export_sequence,
        compile.CompileOnnx(),
        compile.Assemble(),
    ],
)

pytorch_export_sequence_with_quantization = stage.Sequence(
    "pytorch_export_sequence_with_quantization",
    "Exporting PyTorch Model and Quantizing Exported ONNX",
    [
        of_export.ExportPytorchModel(),
        of_export.OptimizeOnnxModel(),
        gf_export.CheckOnnxCompatibility(),
        of_export.QuantizeONNXModel(),
    ],
)

pytorch_sequence_with_quantization = stage.Sequence(
    "pytorch_sequence_with_quantization",
    "Building PyTorch Model",
    [
        pytorch_export_sequence_with_quantization,
        compile.CompileOnnx(),
        compile.Assemble(),
    ],
)

default_keras_export_sequence = stage.Sequence(
    "default_keras_export_sequence",
    "Exporting Keras Model",
    [
        of_export.ExportKerasModel(),
        of_export.OptimizeOnnxModel(),
        gf_export.CheckOnnxCompatibility(),
        of_export.ConvertOnnxToFp16(),
    ],
)

default_keras_sequence = stage.Sequence(
    "default_keras_sequence",
    "Building Keras Model",
    [
        default_keras_export_sequence,
        compile.CompileOnnx(),
        compile.Assemble(),
    ],
)


default_onnx_sequence = stage.Sequence(
    "default_onnx_sequence",
    "Building ONNX Model",
    [
        of_export.ReceiveOnnxModel(),
        of_export.OptimizeOnnxModel(),
        gf_export.CheckOnnxCompatibility(),
        of_export.ConvertOnnxToFp16(),
        compile.CompileOnnx(),
        compile.Assemble(),
    ],
)

default_hummingbird_sequence = stage.Sequence(
    "default_hummingbird_sequence",
    "Building Hummingbird Model",
    [
        hummingbird.ConvertHummingbirdModel(),
        of_export.OptimizeOnnxModel(),
        gf_export.CheckOnnxCompatibility(),
        compile.CompileOnnx(),
        compile.Assemble(),
    ],
)

default_compiler_flags: List[str] = []

default_assembler_flags = [
    "--ifetch-from-self",
    "--ifetch-slice-ordering=round-robin",
]


@typechecked
def _validate_args(  # pylint: disable = unused-argument
    build_name: Optional[str] = None,
    compiler_flags: Optional[List[str]] = None,
    assembler_flags: Optional[List[str]] = None,
    groqview: bool = False,
    groqcard: Optional[str] = build.GROQCARD_A14,
    num_chips: Optional[int] = None,
    topology: Optional[str] = build.TOPOLOGY,
):

    if groqcard is not build.GROQCARD_A14:
        msg = f"""
        You set groqit()'s groqcard argument to {groqcard}, which is not a supported value. The
        currently supported value is: {build.GROQCARD_A14}.
        """
        raise exp.ArgError(msg)

    if num_chips is not None and num_chips > 1:
        if topology is not build.DRAGONFLY and topology is not build.ROTATIONAL:
            msg = f"""
            You set groqit()'s topology argument to {topology}
            for build {build_name}, which is not a supported value. Choose from the
            currently supported values: {build.DRAGONFLY}, {build.ROTATIONAL}.
            """
            raise exp.ArgError(msg)

        supported_topology = build.supported_topology(groqcard, topology)
        if num_chips not in supported_topology.keys():
            msg = f"""
            You set groqit()'s num_chips argument to {num_chips} with topology {topology}
            for build {build_name}, which is not a supported value. Choose from the
            currently supported chip counts: {supported_topology.keys()}.
            """
            raise exp.ArgError(msg)

    if compiler_flags:
        if "--auto-asm" in compiler_flags:
            if assembler_flags:
                msg = """
                The --auto-asm compiler flag is mutually exclusive with the assembler_flags argument
                argument to groqit(). Either set assembler_flags=None or do not use --auto-asm.
                """
                raise exp.ArgError(msg)

        # groqit() may automatically apply certain Groq Compiler flags to each build
        # This check makes sure the user isn't creating a collision by also applying
        # any of these flags
        disallowed_compiler_flags = [
            "--multichip",
            "--groqview",
            "--save-stats",
            "-o",
        ]
        for user_flag in compiler_flags:
            for disallowed_flag in disallowed_compiler_flags:
                if user_flag.startswith(disallowed_flag):
                    msg = f"""
                    The following compiler flags are reserved by groqit() and cannot be used
                    in the groqit(compiler_flags=...) argument: {disallowed_compiler_flags}.
                    However, your compiler_flags argument includes {user_flag}.
                    """
                    raise exp.ArgError(msg)

    if assembler_flags and num_chips != 1:
        msg = """
        The assembler_flags argument is incompatible with multi-chip models.
        Either set num_chips=1 or do not use assembler_flags.
        """
        raise exp.ArgError(msg)


def lock_config(
    model,
    build_name: Optional[str] = None,
    compiler_flags: Optional[List[str]] = None,
    assembler_flags: Optional[List[str]] = None,
    groqview: bool = False,
    groqcard: Optional[str] = build.GROQCARD_A14,
    num_chips: Optional[int] = None,
    topology: Optional[str] = build.TOPOLOGY,
    sequence: stage.Sequence = None,
) -> build.GroqConfig:

    """
    Process the user's configuration arguments to groqit():
    1. Raise exceptions for illegal arguments
    2. Replace unset arguments with default values
    3. Lock the configuration into an immutable object
    """

    _validate_args(
        build_name=build_name,
        compiler_flags=compiler_flags,
        assembler_flags=assembler_flags,
        groqview=groqview,
        groqcard=groqcard,
        num_chips=num_chips,
        topology=topology,
    )

    # Override the onnxflow default opset with GroqFlow's default
    of_build.DEFAULT_ONNX_OPSET = build.DEFAULT_ONNX_OPSET

    of_config = of_ignition.lock_config(
        model=model,
        build_name=build_name,
        sequence=sequence,
    )

    # Use default compiler flags if no flags were specified
    if compiler_flags is None:
        compiler_flags = default_compiler_flags

    # Use default assembler flags if no flags were specified
    if assembler_flags is None:
        assembler_flags = default_assembler_flags

    # Store the args that should be immutable
    config = build.GroqConfig(  # pylint: disable=unexpected-keyword-arg
        build_name=of_config.build_name,
        auto_name=of_config.auto_name,
        compiler_flags=compiler_flags,
        assembler_flags=assembler_flags,
        groqview=groqview,
        groqcard=groqcard,
        topology=topology,
        num_chips=num_chips,
        sequence=of_config.sequence,
        onnx_opset=of_config.onnx_opset,
    )

    return config


def _validate_cached_model(
    config: build.GroqConfig,
    model_type: of_build.ModelType,
    state: build.GroqState,
    model: of_build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Verify whether anything in the call to groqit changed
    We require the user to resolve the discrepancy when such a
    change occurs, so the purpose of this function is simply to
    detect these conditions and raise an appropriate error.
    If this function returns without raising an exception then
    the cached model is valid to use in the build.
    """

    result = of_ignition.validate_cached_model(
        config=config,
        model_type=model_type,
        state=state,
        model=model,
        inputs=inputs,
    )

    current_version_decoded = of_ignition.decode_version_number(groqflow_version)
    state_version_decoded = of_ignition.decode_version_number(state.groqflow_version)

    out_of_date: Union[str, bool] = False
    if current_version_decoded["major"] > state_version_decoded["major"]:
        out_of_date = "major"
    elif current_version_decoded["minor"] > state_version_decoded["minor"]:
        out_of_date = "minor"

    if out_of_date:
        msg = (
            f"Your build {state.config.build_name} was previously built against "
            f"GroqFlow version {state.groqflow_version}, "
            f"however you are now using GroqFlow version {groqflow_version}. The previous build is "
            f"incompatible with this version of GroqFlow, as indicated by the {out_of_date} "
            "version number changing. See **docs/versioning.md** for details."
        )
        result.append(msg)

    return result


def load_or_make_state(
    config: build.GroqConfig,
    cache_dir: str,
    rebuild: str,
    model_type: of_build.ModelType,
    monitor: bool,
    use_sdk: bool,
    model: of_build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
    quantization_samples: Optional[Collection] = None,
) -> build.GroqState:
    """
    Decide whether we can load the model from the GroqFlow model cache
    (return a valid State instance) or whether we need to rebuild it (return
    a new State instance).
    """

    return of_ignition.load_or_make_state(
        config=config,
        cache_dir=cache_dir,
        rebuild=rebuild,
        model_type=model_type,
        monitor=monitor,
        model=model,
        inputs=inputs,
        quantization_samples=quantization_samples,
        state_type=build.GroqState,
        cache_validation_func=_validate_cached_model,
        extra_state_args={"use_sdk": use_sdk},
    )


groq_model_type_to_sequence = {
    of_build.ModelType.PYTORCH: default_pytorch_sequence,
    of_build.ModelType.KERAS: default_keras_sequence,
    of_build.ModelType.ONNX_FILE: default_onnx_sequence,
    of_build.ModelType.HUMMINGBIRD: default_hummingbird_sequence,
}

groq_model_type_to_sequence_with_quantization = {
    of_build.ModelType.PYTORCH: pytorch_sequence_with_quantization,
}


def model_intake(
    user_model,
    user_inputs,
    user_sequence: Optional[stage.Sequence],
    config: build.GroqConfig,
    user_quantization_samples: Optional[Collection] = None,
) -> Tuple[Any, Any, stage.Sequence, of_build.ModelType]:

    model, inputs, sequence, model_type = of_ignition.model_intake(
        user_model=user_model,
        user_inputs=user_inputs,
        user_sequence=user_sequence,
        user_quantization_samples=user_quantization_samples,
        override_quantization_sequence_map=groq_model_type_to_sequence_with_quantization,
        override_sequence_map=groq_model_type_to_sequence,
    )

    if "--auto-asm" in config.compiler_flags:
        sequence.stages = [
            stage
            for stage in sequence.stages
            if not isinstance(stage, compile.Assemble)
        ]

    return (model, inputs, sequence, model_type)
