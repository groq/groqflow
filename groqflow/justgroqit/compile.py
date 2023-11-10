import os
import shutil
import subprocess
import pathlib
import onnx
import torch
import onnxflow.justbuildit.stage as stage
import onnxflow.common.exceptions as exp
import onnxflow.common.printing as printing
import onnxflow.common.build as of_build
import onnxflow.common.onnx_helpers as onnx_helpers
import groqflow.common.build as build
import groqflow.common.sdk_helpers as sdk


def analyze_parameters(state: build.GroqState):
    # Automatically define the number of chips if num_chips is not provided
    if state.config.num_chips is None:
        state.num_chips_used = build.calculate_num_chips(state.info.num_parameters)
    else:
        state.num_chips_used = state.config.num_chips

    # Compile model
    max_chips = build.max_chips(state.config.groqcard, state.config.topology)
    if not state.num_chips_used <= max_chips:
        msg = f"""
        groqit() automatically decided that {state.num_chips_used} GroqChip
        processors are needed to build model "{state.config.build_name}.
        chips supported by groqit at this time ({max_chips}).

        Hint: you can ask groqit to build with a specific number of chips
        less than {max_chips} by setting the num_chips argument in groqit().
        """
        raise exp.StageError(msg)


def analyze_onnx(state: build.GroqState):
    # TODO: validate this input
    # https://git.groq.io/code/Groq/-/issues/13947
    input_onnx = state.intermediate_results[0]

    (
        state.info.compiled_model_input_bytes,
        state.info.compiled_model_output_bytes,
    ) = onnx_helpers.io_bytes(input_onnx)

    # Count the number of trained model parameters
    onnx_model = onnx.load(input_onnx)
    state.info.num_parameters = int(onnx_helpers.parameter_count(onnx_model))


def analyze_torch_script(state: build.GroqState):
    model = torch.jit.load(state.torch_script_file)
    state.info.compiled_model_input_bytes = sum(
        t.element_size() for t in state.inputs.values()
    )
    outputs = model(**state.inputs)
    state.info.compiled_model_output_bytes = sum(t.element_size() for t in outputs)

    state.info.num_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


def torch_types_to_str(type: torch.dtype):
    if type == torch.float16:
        return "f16"
    elif type == torch.float32:
        return "f32"
    elif type == torch.float64:
        return "f64"
    elif type == torch.uint8:
        return "ui8"
    elif type == torch.bool:
        return "i1"
    elif type == torch.int8:
        return "i8"
    elif type == torch.int16:
        return "i16"
    elif type == torch.int32:
        return "i32"
    elif type == torch.int64:
        return "i64"
    elif type == torch.chalf:
        return "complex<f16>"
    elif type == torch.cfloat:
        return "complex<f32>"
    elif type == torch.cdouble:
        return "complex<f64>"
    else:
        raise TypeError("Unsupported Torch type", type)


class Compile(stage.Stage):
    """
    Base class for the Compile stage. self.input_file will be set by the
    derived class.
    """

    def __init__(self):
        super().__init__(
            unique_name="compile",
            monitor_message="Compiling model",
        )

    def fire(self, state: build.GroqState):

        sdk.check_dependencies(require_devtools=True, exception_type=exp.StageError)

        analyze_parameters(state)

        input_file = state.intermediate_results[0]

        # Select either bake or SDK
        if state.use_sdk:
            cmd = sdk.find_tool("groq-compiler")
        else:
            cmd = ["bake", "r", "//Groq/Compiler:groq-compiler"]

        # Add multichip flag if needed
        if state.num_chips_used != 1:
            multichip_flag = f"--multichip={state.topology}"
            cmd = cmd + [multichip_flag]
            if not any(
                flag.startswith("--partition-mode=")
                for flag in state.config.compiler_flags
            ):
                partition_mode_flag = "--partition-mode=daisy-chain"
                cmd = cmd + [partition_mode_flag]

        if state.config.groqview:
            cmd = cmd + ["--groqview"]

        # Add effort=standard by default to help with fit-ability
        if not any(
            flag.startswith("--effort=") for flag in state.config.compiler_flags
        ):
            cmd = cmd + ["--effort=standard"]

        # Add flags
        cmd = (
            cmd
            + [input_file]
            + state.config.compiler_flags
            + [
                "--save-stats",
                f"{state.compile_dir}/stats.json",
                "-o",
                f"{state.compile_dir}/output",
            ]
        )

        # Remove duplicated flags
        cmd = sorted(set(cmd), key=cmd.index)

        state.info.compiler_command = " ".join(cmd)

        printing.logn("Running Groq Compiler... ")

        # TODO: Use subprocess.CalledProcessError to handle exceptions as in groqmodel.py
        # https://git.groq.io/code/Groq/-/issues/14065
        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process:
            for line in process.stdout:
                string = line.decode("utf8")
                print(string, end="")

                # Parse the compiler's RAM usage from its log
                # FIXME: replace with a better approach if possible
                if "Max memory usage" in string:
                    mem_usage = int(float(string.split(":")[3].split(" ")[1]))
                    mem_usage_units = string.split(":")[3].split(" ")[2]
                    if mem_usage_units == "KiB\n":
                        state.info.compiler_ram_bytes = mem_usage * 1024
                    elif mem_usage_units == "MiB\n":
                        state.info.compiler_ram_bytes = mem_usage * 1024 * 1024
                    elif mem_usage_units == "GiB\n":
                        state.info.compiler_ram_bytes = mem_usage * 1024 * 1024 * 1024

        printing.logn("Groq Compiler exited")

        find_files_by_extension = lambda ext: [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(
                of_build.output_dir(state.cache_dir, state.config.build_name)
            )
            for f in filenames
            if os.path.splitext(f)[1] == ext
        ]

        auto_asm = "--auto-asm" in state.config.compiler_flags
        if auto_asm:
            output_files = find_files_by_extension(".iop")
        else:
            output_files = find_files_by_extension(".aa")

        num_outs = sum(["output." in f for f in output_files])

        state.info.compiler_success = (
            True
            if (state.num_chips_used == 1 and num_outs == 1)
            or (state.num_chips_used != 1 and state.num_chips_used == num_outs)
            else False
        )

        if state.info.compiler_success:
            state.intermediate_results = output_files
            if auto_asm:
                # Building the IOP files qualifies the build as a success
                # because those IOP files are the requirement for running
                # a GroqModel instance.
                state.build_status = of_build.Status.SUCCESSFUL_BUILD
        else:
            msg = f"""
            Attempted use Groq Compiler to compile your model's ONNX file into Groq Alan Assembly (.aa)
            files. However, this operation did not succeed.
            Please contact GroqFlow support to determine a path forwards.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state


class CompileOnnx(Compile):
    """
    Stage that takes an ONNX file and compiles it into one or more
    Alan Assembly (.aa) files.

    Expected inputs:
     - state.intermediate_results contains a single .onnx file

    Outputs:
     - One or more .aa files
     - state.num_chips_used contains the number of chips used by
        Groq Compiler
    """

    def fire(self, state: build.GroqState):
        analyze_onnx(state)
        return super().fire(state)


class CompileTorchScript(Compile):
    """
    Stage that takes an TorchScript file and compiles it into GTen.

    Expected inputs:
     - state.intermediate_results contains a single .pt file

    Outputs:
     - One .mlir file
     - state.expected_output_names will contain the output names of the model.
    """

    def fire(self, state: build.GroqState):
        analyze_torch_script(state)

        # Select either bake or SDK
        if state.use_sdk:
            sdk.check_dependencies(require_devtools=True, exception_type=exp.StageError)
            cmd = sdk.find_tool("groq-torch-importer")
        else:
            cmd = ["bake", "r", "//Groq/Compiler/Import/Torch:groq-torch-importer"]

        input_types = []
        for data in state.inputs.values():
            shape = "x".join([str(dim) for dim in data.shape])
            dtype = torch_types_to_str(data.dtype)
            input_types.append(f"--input-types={shape}x{dtype}")

        gten_file = os.path.join(
            state.compile_dir,
            f"{state.config.build_name}.gten.mlir",
        )

        cmd = cmd + [state.torch_script_file] + input_types + ["-o", gten_file]

        # Remove duplicated flags
        cmd = sorted(set(cmd), key=cmd.index)
        state.info.torch_importer_command = " ".join(cmd)

        printing.logn("Running Groq Torch Importer...")

        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process:
            for line in process.stdout:
                printing.logn(line.decode("utf8"), end="")
        printing.logn("Groq Torch Importer has exited")

        state.info.torch_importer_success = (
            True if os.path.exists(gten_file) and os.path.isfile(gten_file) else False
        )

        if state.info.torch_importer_success:
            state.intermediate_results = [gten_file]
        else:
            msg = f"""
            Attempted to use Groq Torch Importer to import TorchSript model into
            Groq's Tensor(GTen) dialect format. However, this operation did not
            succeed. Please contact GroqFlow support to determine a path forwards.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        # Compile the GTen file
        return super().fire(state)


class Assemble(stage.Stage):
    """
    Stage that takes a list of Alan Assembly (.aa) files and runs Groq
    Assembler to transform them into a list of input output program (.iop
    files).

    Expected inputs:
     - state.intermediate_results contains a list of .aa files

    Outputs:
     - A list of .iop files
    """

    def __init__(self):
        super().__init__(
            unique_name="assemble",
            monitor_message="Assembling model",
        )

    def fire(self, state: build.GroqState):

        # TODO: validate the input
        # https://git.groq.io/code/Groq/-/issues/13947

        if state.num_chips_used == 1:

            sdk.check_dependencies(require_devtools=True, exception_type=exp.StageError)

            # Select either bake or SDK
            if state.use_sdk:
                cmd = sdk.find_tool("aa-latest")
            else:
                cmd = ["bake", "r", "//Compiler/alan-assembly:aa-latest"]

            # Select other flags
            input_aa = f"{state.compile_dir}/output.aa"
            output_iop = f"{state.compile_dir}/output.iop"

            cmd = (
                cmd
                + [input_aa, "--output-iop", output_iop]
                + state.config.assembler_flags
            )

        # Multi-chip assemble
        else:

            sdk.check_dependencies(
                require_devtools=True,
                require_runtime=True,
                exception_type=exp.StageError,
            )

            groqit_folder = str(
                pathlib.Path(__file__).parent.resolve().parents[0].parent.resolve()
            )
            if state.use_sdk:
                # Only true when using the CI
                if shutil.which("groqit_assemble_multichip_exe"):
                    cmd = ["groqit_assemble_multichip_exe"]
                # Otherwise, use the python version that comes with the SDK
                else:
                    # TODO: Make "python" default as soon as all users are using the refactored sdk
                    # Note: /usr/local/groq/bin/python only exists in the old SDK
                    if shutil.which("/usr/local/groq/bin/python"):
                        python_cmd = "/usr/local/groq/bin/python"
                    else:
                        python_cmd = "python"
                    cmd = [
                        python_cmd,
                        os.path.join(
                            groqit_folder, "groqflow/justgroqit/assemble_multichip.py"
                        ),
                    ]

            else:
                cmd = [
                    "bake",
                    "r",
                    "//sales/groqit:groqit_assemble_multichip_exe",
                ]

            cmd = cmd + [
                "-t",
                state.topology,
                "-d",
                state.compile_dir,
            ]

            if "--large-program" in state.config.compiler_flags:
                is_large_program = "-l=True"
                cmd = cmd + [is_large_program]

        # Remove duplicated flags
        cmd = sorted(set(cmd), key=cmd.index)
        state.info.assembler_command = " ".join(cmd)

        print("Running Groq Assembler", flush=True)

        with subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process:
            for line in process.stdout:
                print(line.decode("utf8"), end="")
        print("Groq Assembler has exited")

        iop_files = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(
                of_build.output_dir(state.cache_dir, state.config.build_name)
            )
            for f in filenames
            if os.path.splitext(f)[1] == ".iop"
        ]

        num_iop = sum(["output." in f for f in iop_files])
        state.info.assembler_success = (
            state.num_chips_used >= 1 and state.num_chips_used == num_iop
        )

        if state.info.assembler_success:
            state.intermediate_results = iop_files
            # Building the IOP files qualifies the build as a success
            # because those IOP files are the requirement for running
            # a GroqModel instance.
            state.build_status = of_build.Status.SUCCESSFUL_BUILD
        else:
            msg = f"""
            Attempted to use Groq Assembler to convert your model's Alan Assembly (.aa files) into
            Groq input/output program (IOP) files. However, this operation did not succeed.
            Please contact GroqFlow support to determine a path forwards.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.StageError(msg)

        return state
