from typing import Optional, List, Dict, Any
from collections.abc import Collection
import groqflow.justgroqit.ignition as ignition
import groqflow.groqmodel as groqmodel
import groqflow.justgroqit.stage as stage
import groqflow.common.printing as printing
import groqflow.common.build as build


def groqit(
    model: build.UnionValidModelInstanceTypes = None,
    inputs: Optional[Dict[str, Any]] = None,
    build_name: Optional[str] = None,
    cache_dir: str = build.DEFAULT_CACHE_DIR,
    monitor: bool = True,
    rebuild: Optional[str] = None,
    compiler_flags: Optional[List[str]] = None,
    assembler_flags: Optional[List[str]] = None,
    num_chips: Optional[int] = None,
    groqview: bool = False,
    sequence: Optional[List[stage.GroqitStage]] = None,
    quantization_samples: Collection = None,
) -> groqmodel.GroqModel:

    """Use GroqFlow to build a model instance into a GroqModel
        object that can be executed on GroqChip processors.

    Args:
        model: Model to be mapped to a GroqModel, which can be a PyTorch
            model instance, Keras model instance, a path to an ONNX file, or
            a path to a Python script that follows the GroqFlow model.py template.
        inputs: Example inputs to the user's model. The GroqModel will be
            compiled to handle inputs with the same static shape only. Argument
            is not required if the model input is a GroqFlow model.py file.
        build_name: Unique name for the model that will be
            used to store the GroqModel and build state on disk. Defaults to the
            name of the file that calls groqit().
        cache_dir: Directory to use as the GroqFlow cache for this build. Output files
            from this build will be stored at cache_dir/build_name/
            Defaults to ~/.cache/groqflow
        monitor: Display a monitor on the command line that
            tracks the progress of groqit as it builds the GroqModel.
        rebuild: determines whether to rebuild or load a cached build. Options:
            - "if_needed" (default): overwrite invalid cached builds with a rebuild
            - "always": overwrite valid cached builds with a rebuild
            - "never": load cached builds without checking validity, with no guarantee
                of functionality or correctness
            - None: Falls back to default
        compiler_flags: Override groqit's default compiler flags with a list
            of user-specified flags.
        assembler_flags: Override groqit's default assembler flags with a
            list of user-specified flags.
        num_chips: Override the default number of GroqChip processors to be
            used instead of letting groqit decide automatically. Power users
            only.
        groqview: If set, creates a GroqView file for the model during the
            build process. Defaults to false because this option uses up
            significant time and compute/RAM resources.
        sequence: Override groqit's default sequence of build stages. Power
            users only.
        quantization_samples: If set, performs post-training quantization
            on the ONNX model using the provided samples, then generates
            GroqModel from the quantized model. If the previous build used samples
            that are different to the samples used in current build, the "rebuild"
            argument needs to be manually set to "always" in the current build
            in order to create a new GroqModel.
    """
    # Validate and lock in the groqit() config (user arguments that
    # configure the build) that will be used by the rest of groqit()
    (config, auto_name) = ignition.lock_config(
        build_name=build_name,
        compiler_flags=compiler_flags,
        assembler_flags=assembler_flags,
        groqview=groqview,
        groqcard=build.GROQCARD,
        num_chips=num_chips,
    )

    # Analyze the user's model argument and lock in the model, inputs,
    # and sequence that will be used by the rest of groqit()
    (
        model_locked,
        inputs_locked,
        sequence_locked,
        model_type,
        corpus,
    ) = ignition.model_intake(
        model,
        inputs,
        sequence,
        config,
        user_quantization_samples=quantization_samples,
    )

    # Get the state of the model from the GroqFlow cache if a valid build is available
    state = ignition.load_or_make_state(
        config=config,
        cache_dir=cache_dir,
        rebuild=rebuild or build.DEFAULT_REBUILD_POLICY,
        model_type=model_type,
        monitor=monitor,
        use_sdk=build.USE_SDK,
        corpus=corpus,
        model=model_locked,
        inputs=inputs_locked,
        quantization_samples=quantization_samples,
    )

    # Return a cached build if possible, otherwise prepare the model State for
    # a build
    if state.build_status == build.Status.SUCCESSFUL_BUILD:
        # Successful builds can be loaded from cache and returned with
        # no additional steps
        additional_msg = " (build_name auto-selected)" if auto_name else ""
        printing.log_success(
            f' Build "{config.build_name}"{additional_msg} found in cache. Loading it!',
        )

        return groqmodel.load(config.build_name, state.cache_dir)

    state.quantization_samples = quantization_samples

    sequence_locked.show_monitor(config, state.monitor)
    state = sequence_locked.launch(state)

    if state.build_status == build.Status.SUCCESSFUL_BUILD:
        printing.log_success(
            f"\n    Saved to **{build.output_dir(state.cache_dir, config.build_name)}**"
        )

        return groqmodel.load(config.build_name, state.cache_dir)

    else:
        printing.log_success(
            f"Build Sequence {sequence_locked.unique_name} completed successfully"
        )
        msg = """
        groqit() only returns a GroqModel instance if the Sequence includes a Stage
        that sets state.build_status=groqflow.build.Status.SUCCESSFUL_BUILD.
        """
        printing.log_warning(msg)
