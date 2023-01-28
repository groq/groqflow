from typing import Optional, List, Tuple, Union, Dict, Any
from collections.abc import Collection
import sys
import os
import importlib.machinery
import pathlib
import types
import torch
from typeguard import typechecked
import groqflow.common.build as build
import groqflow.common.cache as cache
import groqflow.common.exceptions as exp
import groqflow.common.printing as printing
import groqflow.justgroqit.compile as compile
import groqflow.justgroqit.export as export
import groqflow.justgroqit.stage as stage
import groqflow.justgroqit.hummingbird as hummingbird
from groqflow import __version__ as groqflow_version

try:
    import tensorflow as tf
except ModuleNotFoundError as module_error:
    raise exp.GroqitEnvError(
        "GroqFlow added a dependence on tensorflow in version 2.1.2. "
        "You must install tensorflow to continue."
    )

default_pytorch_export_sequence = stage.Sequence(
    "default_pytorch_export_sequence",
    "Exporting PyTorch Model",
    [
        export.ExportPytorchModel(),
        export.OptimizeOnnxModel(),
        export.CheckOnnxCompatibility(),
        export.ConvertOnnxToFp16(),
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
    "Exporting PyTorch Model and Quantizating Exported ONNX",
    [
        export.ExportPytorchModel(),
        export.OptimizeOnnxModel(),
        export.CheckOnnxCompatibility(),
        export.QuantizeONNXModel(),
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
        export.ExportKerasModel(),
        export.OptimizeOnnxModel(),
        export.CheckOnnxCompatibility(),
        export.ConvertOnnxToFp16(),
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
        export.ReceiveOnnxModel(),
        export.OptimizeOnnxModel(),
        export.CheckOnnxCompatibility(),
        export.ConvertOnnxToFp16(),
        compile.CompileOnnx(),
        compile.Assemble(),
    ],
)

default_hummingbird_sequence = stage.Sequence(
    "default_hummingbird_sequence",
    "Building Hummingbird Model",
    [
        hummingbird.ConvertHummingbirdModel(),
        export.OptimizeOnnxModel(),
        export.CheckOnnxCompatibility(),
        compile.CompileOnnx(),
        compile.Assemble(),
    ],
)

default_compiler_flags = []

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
    groqcard: Optional[build.Groqcard] = build.Groqcard.A14,
    num_chips: Optional[int] = None,
):

    if num_chips is not None:
        supported_topology = build.supported_topology(groqcard)
        if num_chips not in supported_topology:
            msg = f"""
            You set groqit()'s num_chips argument to {num_chips} for build {build_name}, which is
            not a supported value. Choose from the currently supported chip counts: {supported_topology}.
            """
            raise exp.GroqitArgError(msg)

    if compiler_flags:
        if "--auto-asm" in compiler_flags:
            if assembler_flags:
                msg = """
                The --auto-asm compiler flag is mutually exclusive with the assembler_flags argument
                argument to groqit(). Either set assembler_flags=None or do not use --auto-asm.
                """
                raise exp.GroqitArgError(msg)

            if num_chips is None or num_chips > 1:
                msg = """
                The --auto-asm compiler flag is incompatible with multi-chip models.
                Either set num_chips=1 or do not use --auto-asm.
                """
                raise exp.GroqitArgError(msg)

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
                    raise exp.GroqitArgError(msg)

    if assembler_flags and num_chips != 1:
        msg = """
        The assembler_flags argument is incompatible with multi-chip models.
        Either set num_chips=1 or do not use assembler_flags.
        """
        raise exp.GroqitArgError(msg)


def lock_config(
    build_name: Optional[str] = None,
    compiler_flags: Optional[List[str]] = None,
    assembler_flags: Optional[List[str]] = None,
    groqview: bool = False,
    groqcard: Optional[build.Groqcard] = build.Groqcard.A14,
    num_chips: Optional[int] = None,
    sequence: stage.Sequence = None,
) -> Tuple[build.Config, bool]:

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
    )

    # The default model name is the name of the python file that calls GroqIt
    auto_name = False
    if build_name is None:
        build_name = sys.argv[0].split("/")[-1].split(".")[0]
        auto_name = True

    # Use default compiler flags if no flags were specified
    if compiler_flags is None:
        compiler_flags = default_compiler_flags

    # Use default assembler flags if no flags were specified
    if assembler_flags is None:
        assembler_flags = default_assembler_flags

    if sequence is None:
        # The value ["default"] indicates that groqit will be assigning some
        # default sequence later in the program
        stage_names = ["default"]
    else:
        stage_names = sequence.get_names()

    # Store the args that should be immutable
    config = build.Config(
        build_name=build_name,
        compiler_flags=compiler_flags,
        assembler_flags=assembler_flags,
        groqview=groqview,
        groqcard=groqcard,
        num_chips=num_chips,
        sequence=stage_names,
    )

    return config, auto_name


def _validate_cached_model(
    config: build.Config,
    model_type: build.ModelType,
    state: build.State,
    version: str,
    model: build.UnionValidModelInstanceTypes = None,
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

    result = []

    current_version_decoded = _decode_version_number(version)
    state_version_decoded = _decode_version_number(state.groqflow_version)

    out_of_date: Union[str, bool] = False
    if current_version_decoded["major"] > state_version_decoded["major"]:
        out_of_date = "major"
    elif current_version_decoded["minor"] > state_version_decoded["minor"]:
        out_of_date = "minor"

    if out_of_date:
        msg = (
            f"Your build {state.config.build_name} was previously built against "
            f"GroqFlow version {state.groqflow_version}, "
            f"however you are now using GroqFlow version {version}. The previous build is "
            f"incompatible with this version of GroqFlow, as indicated by the {out_of_date} "
            "version number changing. See **docs/versioning.md** for details."
        )
        result.append(msg)

    if model is not None:
        model_changed = state.model_hash != build.hash_model(model, model_type)
    else:
        model_changed = False

    if inputs is not None:
        input_shapes, input_dtypes = build.get_shapes_and_dtypes(inputs)
        input_shapes_changed = state.expected_input_shapes != input_shapes
        input_dtypes_changed = state.expected_input_dtypes != input_dtypes
    else:
        input_shapes_changed = False
        input_dtypes_changed = False

    changed_args = []
    for key in vars(state.config):
        if vars(config)[key] != vars(state.config)[key]:
            changed_args.append((key, vars(config)[key], vars(state.config)[key]))

    # Show an error if the model changed

    build_conditions_changed = (
        model_changed
        or input_shapes_changed
        or input_dtypes_changed
        or len(changed_args) > 0
    )
    if build_conditions_changed:
        # Show an error if build_name is not specified for different models on the same script
        if (
            state.uid == build.unique_id()
            and state.build_status != build.Status.PARTIAL_BUILD
        ):
            msg = (
                "You are building multiple different models in the same script "
                "without specifying a unique groqit(..., build_name=) for each build."
            )
            result.append(msg)

        if model_changed:
            msg = (
                f'Model "{config.build_name}" changed since the last time it was built.'
            )
            result.append(msg)

        if input_shapes_changed:
            msg = (
                f'Input shape of model "{config.build_name}" changed from '
                f"{state.expected_input_shapes} to {input_shapes} "
                f"since the last time it was built."
            )
            result.append(msg)

        if input_dtypes_changed:
            msg = (
                f'Input data type of model "{config.build_name}" changed from '
                f"{state.expected_input_dtypes} to {input_dtypes} "
                f"since the last time it was built."
            )
            result.append(msg)

        if len(changed_args) > 0:
            for key_name, current_arg, previous_arg in changed_args:
                msg = (
                    f'groqit() argument "{key_name}" for build '
                    f"{config.build_name} changed from "
                    f"{previous_arg} to {current_arg} since the last build."
                )
                result.append(msg)
    else:

        if (
            state.build_status == build.Status.FAILED_BUILD
            or state.build_status == build.Status.BUILD_RUNNING
        ) and version == state.groqflow_version:
            msg = (
                "groqit() has detected that you already attempted building this model with the "
                "exact same model, inputs, options, and version of GroqFlow, and that build failed."
            )
            result.append(msg)

    return result


def _decode_version_number(version: str) -> Dict[str, int]:
    numbers = [int(x) for x in version.split(".")]
    return {"major": numbers[0], "minor": numbers[1], "patch": numbers[0]}


def _begin_fresh_build(
    model: build.UnionValidModelInstanceTypes,
    inputs: Optional[Dict[str, Any]],
    monitor: bool,
    rebuild: str,
    use_sdk: bool,
    cache_dir: str,
    config: build.Config,
    model_type: build.ModelType,
    corpus: str,
    groqflow_version: str,
    quantization_samples: Collection,
) -> build.State:
    # Wipe this model's directory in the cache and start with a fresh State.
    cache.rmdir(build.output_dir(cache_dir, config.build_name))
    state = build.State(
        model=model,
        inputs=inputs,
        monitor=monitor,
        rebuild=rebuild,
        use_sdk=use_sdk,
        cache_dir=cache_dir,
        config=config,
        model_type=model_type,
        corpus=corpus,
        groqflow_version=groqflow_version,
        quantization_samples=quantization_samples,
    )
    state.save()

    return state


def _rebuild_if_needed(problem_report: str, state_args: Dict):
    build_name = state_args["config"].build_name
    msg = (
        f"groqit() discovered a cached build of {build_name}, but decided to "
        "rebuild for the following reasons: \n\n"
        f"{problem_report} \n\n"
        "groqit() will now rebuild your model to ensure correctness. You can change this "
        "policy by setting the groqit(rebuild=...) argument."
    )
    printing.log_warning(msg)

    return _begin_fresh_build(**state_args)


def load_or_make_state(
    config: build.Config,
    cache_dir: str,
    rebuild: str,
    model_type: build.ModelType,
    monitor: bool,
    use_sdk: bool,
    corpus: str,
    model: build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
    quantization_samples: Optional[Collection] = None,
) -> build.State:
    """
    Decide whether we can load the model from the GroqFlow model cache
    (return a valid State instance) or whether we need to rebuild it (return
    a new State instance).
    """

    # Put all the args for making a new State instance into a dict
    # to help the following code be cleaner
    state_args = {
        "model": model,
        "inputs": inputs,
        "monitor": monitor,
        "rebuild": rebuild,
        "use_sdk": use_sdk,
        "cache_dir": cache_dir,
        "config": config,
        "model_type": model_type,
        "corpus": corpus,
        "groqflow_version": groqflow_version,
        "quantization_samples": quantization_samples,
    }

    if rebuild == "always":
        return _begin_fresh_build(**state_args)
    else:
        # Try to load state and check if model successfully built before
        if os.path.isfile(build.state_file(cache_dir, config.build_name)):
            try:
                state = build.load_state(cache_dir, config.build_name)

                # if the previous build is using quantization while the current is not
                # or vice versa
                if state.quantization_samples and quantization_samples is None:
                    if rebuild == "never":
                        msg = (
                            f"Model {config.build_name} was built in a previous call to "
                            "groqit() with post-training quantization sample enabled."
                            "However, post-training quantization is not enabled in the "
                            "current build. Rebuild is necessary but currently the rebuild"
                            "policy is set to 'never'. "
                        )
                        raise exp.GroqitCacheError(msg)

                    msg = (
                        f"Model {config.build_name} was built in a previous call to "
                        "groqit() with post-training quantization sample enabled."
                        "However, post-training quantization is not enabled in the "
                        "current build. Starting a fresh build."
                    )

                    printing.log_info(msg)
                    return _begin_fresh_build(**state_args)

                if not state.quantization_samples and quantization_samples is not None:
                    if rebuild == "never":
                        msg = (
                            f"Model {config.build_name} was built in a previous call to "
                            "groqit() with post-training quantization sample disabled."
                            "However, post-training quantization is enabled in the "
                            "current build. Rebuild is necessary but currently the rebuild"
                            "policy is set to 'never'. "
                        )
                        raise exp.GroqitCacheError(msg)

                    msg = (
                        f"Model {config.build_name} was built in a previous call to "
                        "groqit() with post-training quantization sample disabled."
                        "However, post-training quantization is enabled in the "
                        "current build. Starting a fresh build."
                    )

                    printing.log_info(msg)
                    return _begin_fresh_build(**state_args)

            except exp.GroqitStateError as e:
                problem = (
                    "- groqit() failed to load "
                    f"{build.state_file(cache_dir, config.build_name)}"
                )

                if rebuild == "if_needed":
                    return _rebuild_if_needed(problem, state_args)
                else:
                    # Give the rebuild="never" users a chance to address the problem
                    raise exp.GroqitCacheError(e)

            if (
                model_type == build.ModelType.UNKNOWN
                and state.build_status == build.Status.SUCCESSFUL_BUILD
            ):
                msg = (
                    "Model caching is disabled for successful builds against custom Sequences. "
                    "Your model will rebuild whenever you call groqit() on it."
                )
                printing.log_warning(msg)

                return _begin_fresh_build(**state_args)
            elif (
                model_type == build.ModelType.UNKNOWN
                and state.build_status == build.Status.PARTIAL_BUILD
            ):
                msg = (
                    f"Model {config.build_name} was partially built in a previous call to "
                    "groqit(). This call to groqit() found that partial build and is loading "
                    "it from the GroqFlow model cache."
                )

                printing.log_info(msg)
                return state
            else:
                cache_problems = _validate_cached_model(
                    config=config,
                    model_type=model_type,
                    state=state,
                    version=groqflow_version,
                    model=model,
                    inputs=inputs,
                )

                if len(cache_problems) > 0:
                    cache_problems = [f"- {msg}" for msg in cache_problems]
                    problem_report = "\n".join(cache_problems)

                    if rebuild == "if_needed":
                        return _rebuild_if_needed(problem_report, state_args)
                    if rebuild == "never":
                        msg = (
                            "groqit() discovered a cached build of "
                            f"{config.build_name}, and found that it "
                            "is likely invalid for the following reasons: \n\n"
                            f"{problem_report} \n\n"
                            'However, since you have set rebuild="never", groqit() will attempt '
                            "to load the build from cache anyways (with no guarantee of "
                            "functionality or correctness). "
                        )
                        printing.log_warning(msg)
                        return state
                else:
                    return state

        else:
            # No state file found, so we have to build
            return _begin_fresh_build(**state_args)


def load_model_dot_py(full_path):
    # pylint: disable = no-member
    loader = importlib.machinery.SourceFileLoader("a_b", full_path)
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)
    model, inputs = mod.mlagility_model()

    # Get a corpus name (parent folder name) for the model
    corpus = pathlib.Path(os.path.abspath(full_path)).parent.parts[-1]

    return model, inputs, corpus


def _load_model_from_file(path_to_model, user_inputs):
    if not os.path.isfile(path_to_model):
        msg = f"""
        groqit() model argument was passed a string (path to a model file),
        however no file was found at {path_to_model}.
        """
        raise exp.GroqitIntakeError(msg)

    if path_to_model.endswith(".py"):
        if user_inputs is not None:
            msg = """
            groqit() received a path to a model.py file as the model argument, as well as
            an non-None value for the inputs argument. However, when providing a
            model.py file, it is required for inputs=None (because the model.py file
            is expected to return inputs).
            """
            raise exp.GroqitIntakeError(msg)

        model, inputs, corpus = load_model_dot_py(path_to_model)

        if isinstance(model, str):

            if model.endswith(".onnx"):
                return model, inputs, corpus
            else:
                msg = f"""
                groqit() received a model argument that was a path to a model file,
                which returned a path to another file: {model}. However,
                it is required for these returned paths to point to a ".onnx" file.
                """
                raise exp.GroqitIntakeError(msg)

        elif isinstance(
            model, (torch.nn.Module, torch.jit.ScriptModule, tf.keras.Model)
        ):
            return model, inputs, corpus

        else:
            msg = f"""
            groqit() received a model argument that was a path to a model.py file.
            All model.py files are required to return either an ONNX file path,
            a PyTorch model object, or a Keras model object, however the model.py
            provided none of these: {model}
            """
            raise exp.GroqitIntakeError(msg)

    elif path_to_model.endswith(".onnx"):
        return path_to_model, user_inputs, None

    else:
        msg = f"""
        groqit() received a model argument that was a string. However, model string
        arguments are required to be a path to either a .py or .onnx file, and the
        following argument is neither: {path_to_model}
        """
        raise exp.GroqitIntakeError(msg)


model_type_to_sequence = {
    build.ModelType.PYTORCH: default_pytorch_sequence,
    build.ModelType.KERAS: default_keras_sequence,
    build.ModelType.ONNX_FILE: default_onnx_sequence,
    build.ModelType.HUMMINGBIRD: default_hummingbird_sequence,
}

model_type_to_sequence_with_quantization = {
    build.ModelType.PYTORCH: pytorch_sequence_with_quantization,
}


def _validate_inputs(inputs: Dict, model_dot_py_used: bool):
    """
    Check the model's inputs and make sure they are legal. Raise an exception
    if they are not legal.
    TODO: it may be wise to validate the inputs against the model, or at least
    the type of model, as well.
    """

    if inputs is None:
        if model_dot_py_used:
            msg = """
            groqit() requires model inputs. Check your model.py file and make sure it
            returns inputs.
            """
            raise exp.GroqitIntakeError(msg)
        else:
            msg = """
            groqit() requires model inputs. Check your call to groqit() to make sure
            you are passing the inputs argument.
            """
            raise exp.GroqitIntakeError(msg)

    if not isinstance(inputs, dict):
        msg = f"""
        The "inputs" argument to groqit() is required to be a dictionary, where the
        keys map to the named arguments in the model's forward function. The inputs
        received by groqit() were of type {type(inputs)}, not dict.
        """
        raise exp.GroqitIntakeError(msg)


def identify_model_type(model) -> build.ModelType:
    # Validate that the model's type is supported by groqit()
    # and assign a ModelType tag
    if isinstance(model, (torch.nn.Module, torch.jit.ScriptModule)):
        model_type = build.ModelType.PYTORCH
    elif isinstance(model, str):
        if model.endswith(".onnx"):
            model_type = build.ModelType.ONNX_FILE
    elif isinstance(model, tf.keras.Model):
        model_type = build.ModelType.KERAS
        if not tf.executing_eagerly():
            raise exp.GroqitIntakeError(
                "`groqit()` requires Keras models to be run in eager execution mode. "
                "Enable eager execution to continue."
            )
        if not model.built:
            raise exp.GroqitIntakeError(
                "Keras model has not been built. Please call "
                "model.build(input_shape) before running groqit()"
            )
    elif hummingbird.is_supported_model(model):
        model_type = build.ModelType.HUMMINGBIRD
    else:
        raise exp.GroqitIntakeError(
            "Argument 'model' passed to groqit() is "
            f"of unsupported type {type(model)}"
        )

    return model_type


def model_intake(
    user_model,
    user_inputs,
    user_sequence: Optional[stage.Sequence],
    config: build.Config,
    user_quantization_samples: Optional[Collection] = None,
) -> Tuple[Any, Any, stage.Sequence, build.ModelType, str]:

    # Model intake structure options:
    # user_model
    #    |
    #    |------- string
    #    |           |
    #    |           |---- path to onnx model file
    #    |           |
    #    |           |---- python file (must adhere to model.py template)
    #    |                   |
    #    |                   |------- downloads and returns path to onnx file, inputs
    #    |                   |
    #    |                   |------- returns a pytorch model object, inputs
    #    |
    #    |------- pytorch model object
    #    |
    #    |------- keras model object
    #    |
    #    |------- Hummingbird-supported model object

    if user_sequence is None or user_sequence.enable_model_validation:

        if user_model is None and user_inputs is None:
            msg = """
            You are running groqit() without any model, inputs, or custom Sequence. The purpose
            of non-customized groqit() is to build a model against some inputs, so you need to
            provide both.
            """
            raise exp.GroqitIntakeError(msg)

        # Convert paths to models into models
        if isinstance(user_model, str):
            model, inputs, corpus = _load_model_from_file(user_model, user_inputs)
            model_dot_py_used = user_model.endswith(".py")
        else:
            model, inputs, corpus = user_model, user_inputs, ""
            model_dot_py_used = False

        model_type = identify_model_type(model)

        sequence = user_sequence
        if sequence is None:
            # Assign a sequence based on the ModelType
            if user_quantization_samples:
                if model_type != build.ModelType.PYTORCH:
                    raise exp.GroqitIntakeError(
                        "Currently, post training quantization only supports Pytorch models."
                    )
                sequence = model_type_to_sequence_with_quantization[model_type]
            else:
                sequence = model_type_to_sequence[model_type]

        if "--auto-asm" in config.compiler_flags:
            sequence.stages = [
                stage
                for stage in sequence.stages
                if not isinstance(stage, compile.Assemble)
            ]

        _validate_inputs(inputs, model_dot_py_used)

    else:
        # We turn off a significant amount of automation and validation
        # to provide custom stages and sequences with maximum flexibility
        sequence = user_sequence
        model = user_model
        inputs = user_inputs
        model_type = build.ModelType.UNKNOWN
        corpus = ""

    return (model, inputs, sequence, model_type, corpus)
