import os
import sys
import shutil
import subprocess
import pathlib
from typing import Optional, Dict, List, Tuple, Any
from collections.abc import Collection
from dataclasses import dataclass
import numpy as np
import torch

# TODO: Remove try block once GroqFlow "remote" and "cloud" become part of the release
try:
    import groqflow.groqmodel.cloud as cloud
    import groqflow.groqmodel.remote as remote
except ModuleNotFoundError:
    # Proper exceptions are raised if we attempt to use this module on a release branch
    pass
import groqflow.common.exceptions as exp
import groqflow.common.printing as printing
import groqflow.common.build as build
import groqflow.common.sdk_helpers as sdk
import groqflow.common.tensor_helpers as tensor_helpers

try:
    import tensorflow as tf
except ModuleNotFoundError as module_error:
    raise exp.GroqModelEnvError(
        "GroqFlow added a dependence on tensorflow in version 2.1.2. "
        "You must install tensorflow to continue."
    )


@dataclass
class GroqEstimatedPerformance:
    pcie_input_latency: float
    compute_latency: float
    pcie_output_latency: float
    latency_units: str = "seconds"
    throughput_units: str = "inferences per second"

    @property
    def latency(self) -> float:
        return self.pcie_input_latency + self.compute_latency + self.pcie_output_latency

    @property
    def throughput(self) -> float:
        return 1.0 / self.latency


@dataclass
class GroqMeasuredPerformance:
    latency_file: str
    latency_units: str = "seconds"
    throughput_units: str = "inferences per second"

    @property
    def latency(self):
        return np.load(self.latency_file).item()

    @property
    def throughput(self):
        return 1 / self.latency


# An object of this class creates a state
# that may be shared among all calls to GroqModel
class TopologyState:
    def __init__(self):
        self.last_topology = None

    def topology_initialized(self, topology):
        if self.last_topology == topology:
            return 1
        else:
            self.last_topology = topology
            return 0


class GroqModel:
    def __init__(self, state: build.State, tensor_type=np.array, input_dtypes=None):

        self.input_dtypes = input_dtypes
        self.tensor_type = tensor_type
        self.state = state
        self.remote_client = None
        self.log_execute_path = os.path.join(
            build.output_dir(state.cache_dir, self.state.config.build_name),
            "log_execute.txt",
        )

        # checked_dependencies is a persistent state across executions
        # so that we only need to check the dependencies once
        self.checked_dependencies = False

    def run(self, inputs: Dict) -> Any:
        """
        Run a single set of inputs through the model on Groq hardware and return the result

        Args:
            inputs: the same inputs you would provide to the forward function of your model
                for example,
                    given a forward function with signature "forward(x: List[int])"
                    you would call "run(inputs = {"x":[1,2,3]})"
        """

        self._validate_inputs(inputs, "run")

        # Run on GroqChip
        results, _ = self._execute([inputs], repetitions=1)

        return self._unpack_results_file(results)

    def run_abunch(self, input_collection: Collection) -> List:
        """
        Run a bunch of sets of inputs through the model on Groq hardware and
        return a list of results

        Args:
            input_collection: a Collection (e.g., a list), where each member corresponds
            to a set of inputs you would provide to the forward function of your model
                for example,
                    given a forward function with signature "forward(x: List[int])"
                    you would call "run_abunch(input_collection=[{"x":[1,2,3]},{"x":[4,5,6]}])
        """

        self._validate_input_collection(input_collection, "run_abunch")

        # Run on GroqChip
        results, _ = self._execute(input_collection, repetitions=1)

        # Return unpacked results
        return self._unpack_results_file(results)

    def estimate_performance(self) -> GroqEstimatedPerformance:
        """
        Get estimated performance based on cycles to compute and PCIe latency.
        The estimations account for on-chip compute and PCIe transfers only.
        This function returns the estimated latency in seconds and the throughput
        in inferences per second (IPS).
        """

        # Get the number of cycles needed to execute the model
        on_chip_compute_cycles = build.load_yaml(self.state.stats_file)["total_cycles"]

        # TODO: Read the frequency from a central location once the issue below is solved
        # https://git.groq.io/code/Groq/-/issues/14155
        frequency = 900000000  # 900 MHz
        pcie_latency = 1e-5  # 10 microseconds
        pcie_bandwidth = 24e9  # 24 GB/s

        # Calculate compute latency and estimate PCIe latency
        self.state.info.estimated_pcie_input_latency = (
            self.state.info.compiled_onnx_input_bytes / pcie_bandwidth
        ) + pcie_latency
        self.state.info.deterministic_compute_latency = on_chip_compute_cycles / (
            frequency
        )
        self.state.info.estimated_pcie_output_latency = (
            self.state.info.compiled_onnx_output_bytes / pcie_bandwidth
        ) + pcie_latency

        # When pipelined, the reported cycle is the duration of a single pipelining stage
        # Note: Models are pipelined by default
        if not "--no-multichip-pipelining" in self.state.config.compiler_flags:
            self.state.info.deterministic_compute_latency *= self.state.num_chips_used

        # Save estimated perm
        estimated_perf = GroqEstimatedPerformance(
            pcie_input_latency=self.state.info.estimated_pcie_input_latency,
            compute_latency=self.state.info.deterministic_compute_latency,
            pcie_output_latency=self.state.info.estimated_pcie_output_latency,
        )
        self.state.info.estimated_latency = estimated_perf.latency
        self.state.info.estimated_throughput = estimated_perf.throughput
        self.state.save()

        return estimated_perf

    def benchmark(
        self, inputs: Optional[Dict] = None, repetitions: int = 100
    ) -> GroqMeasuredPerformance:
        if inputs is None:
            printing.log_info(
                (
                    "No inputs received for benchmark. Using the inputs"
                    " provided during model compilation."
                )
            )

            # Load previously-saved input
            inputs = np.load(self.state.original_inputs_file, allow_pickle=True).item()

        else:
            self._validate_inputs(inputs, "benchmark")

        _, benchmark_results = self._execute(
            input_collection=[inputs], repetitions=repetitions
        )

        self.state.info.measured_latency = benchmark_results.latency
        self.state.info.measured_throughput = benchmark_results.throughput
        self.state.save()

        return benchmark_results

    def benchmark_abunch(
        self, input_collection: Collection, repetitions: int = 1
    ) -> GroqMeasuredPerformance:

        self._validate_input_collection(input_collection, "benchmark_abunch")

        _, benchmark_results = self._execute(
            input_collection=input_collection, repetitions=repetitions
        )

        self.state.info.measured_latency = benchmark_results.latency
        self.state.info.measured_throughput = benchmark_results.throughput
        self.state.save()

        return benchmark_results

    def _validate_input_collection(self, input_collection, function_name) -> None:
        if input_collection is None:
            raise exp.GroqModelArgError(
                (
                    f"GroqModel.{function_name}() received an input_collection with type "
                    f"{type(input_collection)}, however the input_collection arg must be "
                    "a collection of dictionaries."
                )
            )
        else:
            if len(input_collection) == 0:
                raise exp.GroqModelArgError(
                    f"GroqModel.{function_name}() received an empty collection as input."
                )

        # Check whether all elements of input_collection have the shape required by the model
        for inputs in input_collection:
            self._validate_inputs(inputs, function_name, True)

    def _validate_inputs(self, inputs, function_name, from_collection=False) -> None:
        if from_collection:
            collection_msg = "input_collection "
        else:
            collection_msg = ""

        if not isinstance(inputs, dict):
            raise exp.GroqModelArgError(
                (
                    f"GroqModel.{function_name}() {collection_msg}received inputs of type "
                    f"{type(inputs)}, however the inputs must be a dictionary."
                )
            )

        # Check whether the inputs provided have the shapes required by the model's
        # forward function
        tensor_helpers.check_shapes_and_dtypes(
            inputs, self.state.expected_input_shapes, self.state.expected_input_dtypes
        )

    def _select_backend(self):
        # Define backend
        # TODO: Allow backend to be changed using python once Remote backend is publicly released
        if os.environ.get("GROQMODEL_BACKEND"):
            groqmodel_backend_env_var = os.environ.get("GROQMODEL_BACKEND")
            try:
                backend = build.Backend(groqmodel_backend_env_var)
            except ValueError as e:
                raise ValueError(
                    (
                        "GROQMODEL_BACKEND environment variable set to "
                        f'"{groqmodel_backend_env_var}", but only "local", "cloud", '
                        '"remote", and "auto" are valid. '
                        '"remote" backend is not yet recommended.'
                    )
                ) from e
        else:
            backend = build.Backend.LOCAL

        # Switch to cloud backend if no local GroqChips are available
        if backend == build.Backend.AUTO:
            if sdk.get_num_chips_available() == 0:
                backend = build.Backend.CLOUD
                print(
                    "Switching to GroqCloud server, since this machine has no GroqChip processors"
                )
            else:
                backend = build.Backend.LOCAL

        # Check if we are trying to use GroqFlow Remote/Cloud on a public release
        if (
            backend == build.Backend.CLOUD
            and "groqflow.groqmodel.cloud" not in sys.modules
        ) or (
            backend == build.Backend.REMOTE
            and "groqflow.groqmodel.remote" not in sys.modules
        ):
            raise exp.GroqModelEnvError(
                (
                    f"GroqFlow {backend.value} is not publicly available yet. "
                    "Please set the environment variable GROQMODEL_BACKEND to 'local'."
                )
            )

        if backend == build.Backend.REMOTE and self.remote_client is None:
            # Setup remote client if needed
            remote_url = os.environ.get("GROQFLOW_REMOTE_URL")
            self.remote_client = (
                remote.RemoteClient()
                if remote_url is None
                else remote.RemoteClient(remote_url)
            )

        return backend

    # Shared execution function
    def _execute(
        self, input_collection: Collection, repetitions: int
    ) -> Tuple[Any, GroqMeasuredPerformance]:
        """
        Execute model on GroqChip processors and return the results and performance
        """

        # Validate inputs arg
        if not isinstance(input_collection, list):
            raise ValueError(
                f"Expected arg input_collection to be a list but received {type(input_collection)}."
            )

        # GroqFlow currently allows 8+ chips models to be built only when GroqFlow's internal
        # features are enabled. Executing 8+ chips models is not currently supported by GroqFlow.
        if self.state.num_chips_used > 8:
            msg = (
                f"Groqit's num_chips_used was set to {self.state.num_chips_used}, "
                "but the current runtime only allows for up to 8 GroqChips."
            )

            raise exp.GroqFlowError(msg)

        # Save inputs to file
        to_downcast = False if self.state.quantization_samples else True
        tensor_helpers.save_inputs(
            input_collection,
            self.state.execution_inputs_file,
            self.input_dtypes,
            downcast=to_downcast,
        )

        # Remove previously stored latency/outputs
        if os.path.isfile(self.state.outputs_file):
            os.remove(self.state.outputs_file)
        if os.path.isfile(self.state.latency_file):
            os.remove(self.state.latency_file)

        bringup_topology = shared_state.topology_initialized(self.state.topology)

        # Select execution script according to backend
        backend = self._select_backend()
        if backend == build.Backend.CLOUD:
            cloud.execute_groqchip_remotely(
                bringup_topology, repetitions, self.state, self.log_execute_path
            )
        elif backend == build.Backend.REMOTE:
            try:
                self.remote_client.execute(self.state, repetitions)
            except Exception as e:
                raise exp.GroqModelRemoteError(
                    (
                        "There was an issue when running your model using GroqFlow Remote. "
                    )
                ) from e
        else:
            self._execute_locally(bringup_topology, repetitions)

        return self.state.outputs_file, GroqMeasuredPerformance(self.state.latency_file)

    # Models with a single output are returned as either a torch.tensor,
    # tf.Tensor, or an np.array (see tensor_type)
    # Models with multiple outputs are returned as either a tuple of
    # torch.tensors, tf.Tensors, or np.arrays
    def _unpack_results(self, results: List[Dict], output_nodes, num_outputs):
        if self.tensor_type is tf.Tensor:
            unpacked_results = [tf.convert_to_tensor(results[x]) for x in output_nodes]
        else:
            unpacked_results = [self.tensor_type(results[x]) for x in output_nodes]
        return unpacked_results[0] if num_outputs == 1 else tuple(unpacked_results)

    def _unpack_results_file(self, packed_results: str) -> Any:
        """
        Unpack execution results from a file
        """

        np_result = np.load(packed_results, allow_pickle=True)

        # Ensure that the output nodes generated are the same as the expected output nodes
        output_nodes = self.state.expected_output_names
        num_outputs = len(output_nodes)
        output_nodes_received = list(np_result[0].keys())
        if not all(node in output_nodes for node in output_nodes_received):
            raise exp.GroqModelRuntimeError(
                (
                    f"GroqFlow expected outputs {str(self.state.expected_output_names)} "
                    f"but got {str(output_nodes_received)}"
                )
            )

        # Unpack all results from the collection and pack them in a list
        unpacked_result_list = [
            self._unpack_results(output_sample, output_nodes, num_outputs)
            for output_sample in np_result
        ]

        # If a collection of inputs was received, return a list of results
        # If a single set of inputs was received, return a single result
        if len(np_result) > 1:
            return unpacked_result_list
        else:
            return unpacked_result_list[0]

    def _execute_locally(self, bringup_topology: bool, repetitions: int) -> None:
        """
        Execute GroqModel on the local machine, rather than remotely
        """

        if not self.checked_dependencies:
            self.checked_dependencies = sdk.check_dependencies(
                require_runtime=True, require_devtools=True
            )

        # Check local number of GroqChips available
        src_folder = pathlib.Path(__file__).parent.resolve()
        chips_available = sdk.get_num_chips_available()
        if self.state.num_chips_used > chips_available:
            raise exp.GroqModelRuntimeError(
                f"Trying to execute a model compiled for {self.state.num_chips_used}"
                f" GroqChip processors but this machine only has {chips_available}"
                " GroqChip processors available."
            )

        # Configure execution command
        if shutil.which("groqmodel_execute_exe"):
            execution_script = ["groqmodel_execute_exe"]
        else:
            # TODO: Make "python" default as soon as all users are using the refactored sdk
            # Note: /usr/local/groq/bin/python only exists in the old SDK
            if shutil.which("/usr/local/groq/bin/python"):
                python_cmd = "/usr/local/groq/bin/python"
            else:
                python_cmd = "python"
            execution_script = [
                python_cmd,
                f"{src_folder}/execute.py",
            ]
        cmd = execution_script + [
            str(self.state.num_chips_used),
            build.output_dir(self.state.cache_dir, self.state.config.build_name),
            self.state.outputs_file,
            self.state.latency_file,
            self.state.topology,
            str(repetitions),
        ]
        if bringup_topology:
            cmd = cmd + ["--bringup_topology"]

        # Call and wait for subprocess
        try:
            # FIXME: STDOUT is only saved when the process succeeds (STDERR always saved)
            # https://git.groq.io/code/Groq/-/issues/13876
            sys.stderr = build.Logger(self.log_execute_path)
            output = subprocess.check_output(cmd)
            print(output.decode("utf-8"), file=sys.stderr)
            sys.stderr = sys.stderr.terminal
        except subprocess.CalledProcessError as e:
            # This exception will show the Traceback on the subprocess followed the message
            # "The above exception was the direct cause of the following exception" and then
            # the Traceback outside the subprocess.
            with open(self.log_execute_path, "r", encoding="utf-8") as f:
                subprocess_stderr = f.read()
            if "in bringup_topology" in subprocess_stderr:
                msg = "Failed to bringup GroqChip topology. "
            else:
                msg = "Failed while trying to run GroqChips locally. "
            msg = msg + f"Refer to **{self.log_execute_path}** for details."
            raise exp.GroqModelRuntimeError(msg) from e

    # Launch groqview
    def groqview(self) -> None:

        # Select either bake or SDK
        if self.state.use_sdk:
            groqview_path = sdk.find_tool("groqview")
        else:
            groqview_path = [
                "bake",
                "r",
                "//Groq/View/Server:groqview",
            ]

        # Check if the groqview file exists
        if not os.path.isdir(self.state.groqview_file):
            raise exp.GroqFlowError(
                "GroqView directory not found. Please recompile your model with groqview=True"
            )

        # Close any existing groqview-servers to avoid port collision and open GroqView
        # FIXME: Ensure that multiple Groqview instances can stay open simultaneously
        # https://git.groq.io/code/Groq/-/issues/14066
        subprocess.Popen(["pkill", "groqview-server"]).wait()
        subprocess.Popen(groqview_path + [self.state.groqview_file]).wait()

    # Launch Netron
    def netron(self) -> None:
        # Check if Netron is installed
        if not shutil.which("netron"):
            raise exp.GroqitEnvError("Netron installation not found.")

        # Launch netron
        subprocess.Popen(["netron", self.state.opt_onnx_file]).wait()


class PytorchModelWrapper(GroqModel):
    def __init__(self, state):
        tensor_type = torch.tensor
        super(PytorchModelWrapper, self).__init__(state, tensor_type)

    # Pytorch models are callable
    def __call__(self, **kwargs):
        return self.run(kwargs)


class KerasModelWrapper(GroqModel):
    def __init__(self, state):
        tensor_type = tf.Tensor
        super(KerasModelWrapper, self).__init__(state, tensor_type)

    # Keras models are callable
    def __call__(self, **kwargs):
        return self.run(kwargs)


class HummingbirdWrapper(GroqModel):
    def __init__(self, state):
        super(HummingbirdWrapper, self).__init__(
            state, input_dtypes={"input_0": "float32"}
        )

    def predict(self, input):
        return self.run({"input_0": input})[0]

    def predict_proba(self, input):
        return self.run({"input_0": input})[1]

    def _unpack_results(self, results: List[Dict], output_nodes, num_outputs):
        unpacked_results = [
            self.tensor_type(results[k]) for k in output_nodes if k != "variable"
        ]
        unpacked_results.insert(0, self.tensor_type(results["variable"]))
        return unpacked_results[0] if num_outputs == 1 else tuple(unpacked_results)


def load(build_name: str, cache_dir=build.DEFAULT_CACHE_DIR) -> GroqModel:
    state = build.load_state(cache_dir=cache_dir, build_name=build_name)

    if state.model_type == build.ModelType.PYTORCH:
        return PytorchModelWrapper(state)
    elif state.model_type == build.ModelType.KERAS:
        return KerasModelWrapper(state)
    elif state.model_type == build.ModelType.HUMMINGBIRD:
        return HummingbirdWrapper(state)
    else:
        return GroqModel(state)


# Instantiate shared state object
shared_state = TopologyState()
