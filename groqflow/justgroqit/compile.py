import os
import shutil
import subprocess
import pathlib
import onnx
import groqflow.justgroqit.stage as stage
import groqflow.common.exceptions as exp
import groqflow.common.printing as printing
import groqflow.common.build as build
import groqflow.common.onnx_helpers as onnx_helpers
import groqflow.common.sdk_helpers as sdk


def get_and_analyze_onnx(state: build.State):
    # TODO: validate this input
    # https://git.groq.io/code/Groq/-/issues/13947
    input_onnx = state.intermediate_results[0]

    (
        state.info.compiled_onnx_input_bytes,
        state.info.compiled_onnx_output_bytes,
    ) = onnx_helpers.io_bytes(input_onnx)

    # Count the number of trained model parameters
    onnx_model = onnx.load(input_onnx)
    state.info.num_parameters = int(onnx_helpers.parameter_count(onnx_model))

    # Automatically define the number of chips if num_chips is not provided
    if state.config.num_chips is None:
        state.num_chips_used = build.calculate_num_chips(state.info.num_parameters)
    else:
        state.num_chips_used = state.config.num_chips

    # Compile model
    max_chips = build.max_chips(state.config.groqcard)
    if not state.num_chips_used <= max_chips:
        msg = f"""
        groqit() automatically decided that {state.num_chips_used} GroqChip
        processors are needed to build model "{state.config.build_name}.
        chips supported by groqit at this time ({max_chips}).

        Hint: you can ask groqit to build with a specific number of chips
        less than {max_chips} by setting the num_chips argument in groqit().
        """
        raise exp.GroqitStageError(msg)

    return input_onnx


class CompileOnnx(stage.GroqitStage):
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

    def __init__(self):
        super().__init__(
            unique_name="compile",
            monitor_message="Compiling model",
        )

    def fire(self, state: build.State):

        sdk.check_dependencies(
            require_devtools=True, exception_type=exp.GroqitStageError
        )

        input_onnx = get_and_analyze_onnx(state)

        # Select either bake or SDK
        if state.use_sdk:
            cmd = sdk.find_tool("groq-compiler")
        else:
            cmd = ["bake", "r", "//Groq/Compiler:groq-compiler"]

        # Add multichip flag if needed
        if int(state.num_chips_used) != 1:
            multichip_flag = f"--multichip={state.topology}"
            cmd = cmd + [multichip_flag]

        if state.config.groqview:
            cmd = cmd + ["--groqview"]

        # Add flags
        cmd = (
            cmd
            + [input_onnx]
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
                build.output_dir(state.cache_dir, state.config.build_name)
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
                state.build_status = build.Status.SUCCESSFUL_BUILD
        else:
            msg = f"""
            Attempted use Groq Compiler to compile your model's ONNX file into Groq Alan Assembly (.aa)
            files. However, this operation did not succeed.
            Please contact GroqFlow support to determine a path forwards.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        return state


class Assemble(stage.GroqitStage):
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

    def fire(self, state: build.State):

        # TODO: validate the input
        # https://git.groq.io/code/Groq/-/issues/13947

        if int(state.num_chips_used) == 1:

            sdk.check_dependencies(
                require_devtools=True, exception_type=exp.GroqitStageError
            )

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
                exception_type=exp.GroqitStageError,
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
                build.output_dir(state.cache_dir, state.config.build_name)
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
            state.build_status = build.Status.SUCCESSFUL_BUILD
        else:
            msg = f"""
            Attempted to use Groq Assembler to convert your model's Alan Assembly (.aa files) into
            Groq input/output program (IOP) files. However, this operation did not succeed.
            Please contact GroqFlow support to determine a path forwards.
            More information may be available in the log file at **{self.logfile_path}**
            """
            raise exp.GroqitStageError(msg)

        return state
