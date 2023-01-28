import os
import sys
import pathlib
import copy
import enum
import math
from typing import Optional, Any, List, Dict, Union
from collections.abc import Collection
import dataclasses
import hashlib
import yaml
import torch
import numpy as np
import sklearn.base
import groqflow.common.exceptions as exp

try:
    import tensorflow as tf
except ModuleNotFoundError as module_error:
    raise exp.GroqitEnvError(
        "GroqFlow added a dependence on tensorflow in version 2.1.2. "
        "You must install tensorflow to continue."
    )


UnionValidModelInstanceTypes = Union[
    None,
    str,
    torch.nn.Module,
    torch.jit.ScriptModule,
    tf.keras.Model,
    sklearn.base.BaseEstimator,
]


class Groqcard(enum.Enum):
    A14 = "A1.4"
    A11 = "A1.1"


# WARNING: The "internal" env var may cause unexpected behavior if enabled
# outside of the internal Groq dev environment.
environment_variables = {
    "cache_dir": "GROQFLOW_CACHE_DIR",
    "rebuild": "GROQIT_REBUILD_POLICY",
    "dont_use_sdk": "GROQFLOW_BAKE_SDK",
    "target_a11": "GROQFLOW_LEGACY_A11",
    "debug": "GROQFLOW_DEBUG",
    "internal": "GROQFLOW_INTERNAL_FEATURES",
}

DEFAULT_ONNX_OPSET = 14
MINIMUM_ONNX_OPSET = 11

# Allow an environment variable to override the default
# location for the GroqFlow build cache
if os.environ.get(environment_variables["cache_dir"]):
    DEFAULT_CACHE_DIR = os.environ.get(environment_variables["cache_dir"])
else:
    DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/groqflow")

# Allow an environment variable to override the default
# rebuild policy
if os.environ.get(environment_variables["rebuild"]):
    DEFAULT_REBUILD_POLICY = os.environ.get(environment_variables["rebuild"])
    rebuild_allowed_values = ["if_needed", "always", "never"]
    if DEFAULT_REBUILD_POLICY not in rebuild_allowed_values:
        raise ValueError(
            f'Environment variable set for {environment_variables["rebuild"]} has '
            f"value {DEFAULT_REBUILD_POLICY}, which is not one of the following allowed "
            f"values: {rebuild_allowed_values} "
        )
else:
    DEFAULT_REBUILD_POLICY = "if_needed"

# Allow an environment variable to tell groqit to build an SDK
# with bake instead of using an installed copy of the SDK (only
# useful for internal Groq developers)
if os.environ.get(environment_variables["dont_use_sdk"]) == "True":
    USE_SDK = False
else:
    USE_SDK = True

# Direct builds to target legacy GroqCard A1.1 accelerators instead
# of the default A1.4 accelerators
if os.environ.get(environment_variables["target_a11"]) == "True":
    GROQCARD = Groqcard.A11
else:
    GROQCARD = Groqcard.A14


class Backend(enum.Enum):
    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"
    REMOTE = "remote"


class ModelType(enum.Enum):
    PYTORCH = "pytorch"
    KERAS = "keras"
    ONNX_FILE = "onnx_file"
    HUMMINGBIRD = "hummingbird"
    UNKNOWN = "unknown"


def supported_topology(groqcard: Groqcard):
    if os.environ.get(environment_variables["internal"]) == "True":
        return [1, 2, 4] if groqcard == Groqcard.A11 else [1, 2, 4, 8, 16, 32, 64]
    else:
        return [1, 2, 4] if groqcard == Groqcard.A11 else [1, 2, 4, 8]


def max_chips(groqcard: Groqcard):
    return supported_topology(groqcard)[-1]


# Each chip can hold approximately 50M parameters
# Number of chips need to be either 1, 2, 4, 8, 16, 32 or 64
def calculate_num_chips(num_parameters, estimate=False):
    if num_parameters is not None:
        if num_parameters == 0:
            return 1
        else:
            x = math.ceil(num_parameters / 50000000)
            if estimate:
                return x
            else:
                return 2 ** (x - 1).bit_length()
    else:
        return None


def load_yaml(file_path):
    with open(file_path, "r", encoding="utf8") as stream:
        try:
            return yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            raise exp.GroqFlowIOError(
                f"Failed while trying to open {file_path}."
                f"The exception that triggered this was:\n{e}"
            )


def output_dir(cache_dir, build_name):
    path = os.path.join(cache_dir, build_name)
    return path


def state_file(cache_dir, build_name):
    state_file_name = f"{build_name}_state.yaml"
    path = os.path.join(output_dir(cache_dir, build_name), state_file_name)
    return path


def hash_model(model, model_type: ModelType, hash_params: bool = True):

    # If the model is a path to a file, hash the file
    if model_type == ModelType.ONNX_FILE:
        # TODO: Implement a way of hashing the models but not the parameters
        # of ONNX inputs.
        if not hash_params:
            msg = "hash_params must be True for model_type ONNX_FILE"
            raise ValueError(msg)
        if os.path.isfile(model):
            with open(model, "rb") as f:
                file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
        else:
            raise ValueError(
                "hash_model received str model that doesn't correspond to a file"
            )

    elif model_type == ModelType.PYTORCH:
        # Convert model parameters and topology to string
        hashable_params = {}
        for name, param in model.named_parameters():
            hashable_params[name] = param.data
        if hash_params:
            hashable_model = (str(model) + str(hashable_params)).encode()
        else:
            hashable_model = str(model).encode()

        # Return hash of topology and parameters
        return hashlib.sha256(hashable_model).hexdigest()

    elif model_type == ModelType.KERAS:
        # Convert model parameters and topology to string
        summary_list = []  # type: List[str]

        # pylint: disable=unnecessary-lambda
        model.summary(print_fn=lambda x: summary_list.append(x))

        summary_str = " ".join(summary_list)
        hashable_params = {}
        for layer in model.layers:
            hashable_params[layer.name] = layer.weights
        if hash_params:
            hashable_model = (summary_str + str(hashable_params)).encode()
        else:
            hashable_model = summary_str.encode()

        # Return hash of topology and parameters
        return hashlib.sha256(hashable_model).hexdigest()

    elif model_type == ModelType.HUMMINGBIRD:
        import pickle

        return hashlib.sha256(pickle.dumps(model)).hexdigest()

    else:
        msg = f"""
        model_type "{model_type}" unsupported by groqit's hash_model function
        """
        raise ValueError(msg)


class Status(enum.Enum):
    NOT_STARTED = "not_started"
    PARTIAL_BUILD = "partial_build"
    BUILD_RUNNING = "build_running"
    SUCCESSFUL_BUILD = "successful_build"
    FAILED_BUILD = "failed_build"


# Create a unique ID from this run by hashing pid + process start time
def unique_id():
    pid = os.getpid()
    start_time = os.path.getctime(f"/proc/{pid}/stat")
    return hashlib.sha256(f"{pid}{start_time}".encode()).hexdigest()


def get_shapes_and_dtypes(inputs: dict):
    """
    Return the shape and data type of each value in the inputs dict
    """
    shapes = {}
    dtypes = {}
    for key in sorted(inputs):
        value = inputs[key]
        if (
            isinstance(
                value,
                (list, tuple),
            )
            or torch.is_tensor(value)
            or tf.is_tensor(value)
        ):
            shapes[key] = np.array(value).shape
            dtypes[key] = np.array(value).dtype.name
        elif isinstance(value, np.ndarray):
            shapes[key] = value.shape
            dtypes[key] = value.dtype.name
        elif isinstance(value, (bool, int, float)):
            shapes[key] = (1,)
            dtypes[key] = type(value).__name__
        elif value is None:
            pass
        else:
            raise exp.GroqFlowError(
                "One of the provided inputs contains the unsupported "
                f' type {type(value)} at key "{key}".'
            )

    return shapes, dtypes


@dataclasses.dataclass(frozen=True)
class Config:
    """
    User-provided build configuration. GroqFlow is not allowed
    to change instances of Config once they have been
    instantiated (frozen=True enforces this).

    Note: modifying this struct can create a breaking change that
    requires users to rebuild their models. Increment the minor
    version number of the groqflow package if you do make a build-
    breaking change.
    """

    build_name: str
    compiler_flags: List[str]
    assembler_flags: List[str]
    groqview: bool
    groqcard: Groqcard
    sequence: List[str]
    num_chips: Optional[int] = None


@dataclasses.dataclass
class Info:
    """
    Information about a build that may be useful for analysis
    or debugging purposes.

    Note: GroqFlow does not guarantee that members of this class will
    have non-None values at the end of a build. GroqFlow code must
    not take a dependence on any member of this class.
    """

    backend: Backend = Backend.AUTO
    num_parameters: Optional[int] = None
    base_onnx_exported: Optional[bool] = None
    opt_onnx_exported: Optional[bool] = None
    opt_onnx_ops: Optional[List[str]] = None
    opt_onnx_unsupported_ops: Optional[List[str]] = None
    opt_onnx_all_ops_supported: Optional[bool] = None
    converted_onnx_exported: Optional[bool] = None
    quantized_onnx_exported: Optional[bool] = None
    compiler_success: Optional[bool] = None
    compiler_command: Optional[str] = None
    assembler_success: Optional[bool] = None
    assembler_command: Optional[str] = None
    measured_latency: Optional[float] = None
    measured_throughput: Optional[float] = None
    estimated_pcie_input_latency: Optional[float] = None
    deterministic_compute_latency: Optional[float] = None
    estimated_pcie_output_latency: Optional[float] = None
    estimated_throughput: Optional[float] = None
    estimated_latency: Optional[float] = None
    skipped_stages: int = 0
    opset: Optional[int] = DEFAULT_ONNX_OPSET
    compiled_onnx_input_bytes: int = None
    compiled_onnx_output_bytes: int = None
    all_build_stages: List[str] = dataclasses.field(default_factory=list)
    current_build_stage: str = None
    completed_build_stages: List[str] = dataclasses.field(default_factory=list)
    build_stage_execution_times: Dict[str, float] = dataclasses.field(
        default_factory=dict
    )
    compiler_ram_bytes: float = None


@dataclasses.dataclass
class State:
    # User-provided args that influence the generated model
    config: Config

    # User-provided args that do not influence the generated model
    monitor: bool
    rebuild: str
    use_sdk: bool
    cache_dir: str

    # User-provided args that will not be saved as part of state.yaml
    model: UnionValidModelInstanceTypes = None
    inputs: Optional[Dict[str, Any]] = None

    # Optional information about the build
    info: Info = Info()

    # Member variable that helps the code know if State has called
    # __post_init__ yet
    after_post_init: bool = False

    # All of the following are critical aspects of the build,
    # including properties of GroqFlow and choices made by GroqFlow
    # while building the model, which determine the outcome of the build.
    # NOTE: adding or changing a member name in this struct can create
    # a breaking change that requires users to rebuild their models.
    # Increment the minor version number of the groqflow package if you
    # do make a build-breaking change.

    groqflow_version: str = ""
    model_type: ModelType = ModelType.UNKNOWN
    uid: Optional[int] = None
    num_chips_used: Optional[int] = None
    model_hash: Optional[int] = None
    build_status: Status = Status.NOT_STARTED
    expected_input_shapes: Optional[Dict[str, list]] = None
    expected_input_dtypes: Optional[Dict[str, list]] = None
    expected_output_names: Optional[List] = None
    # The results of the most recent stage that was executed
    intermediate_results: Any = None
    # Folder a model file was found in. Useful for processing
    # large quantities of models.
    corpus: str = ""

    quantization_samples: Optional[Collection] = None

    def __post_init__(self):
        if self.uid is None:
            self.uid = unique_id()
        if self.inputs is not None:
            (
                self.expected_input_shapes,
                self.expected_input_dtypes,
            ) = get_shapes_and_dtypes(self.inputs)
        if self.model is not None and self.model_type != ModelType.UNKNOWN:
            self.model_hash = hash_model(self.model, self.model_type)

        self.after_post_init = True

    def __setattr__(self, name, val):
        super().__setattr__(name, val)

        # Always automatically save the state.yaml whenever State is modified
        # But don't bother saving until after __post_init__ is done (indicated
        # by the after_post_init flag)
        # Note: This only works when elements of the state are set directly.
        # When an element of state.info gets set, for example, state needs
        # to be explicitly saved by calling state.save().
        if self.after_post_init and name != "after_post_init":
            self.save()

    @property
    def original_inputs_file(self):
        return os.path.join(
            output_dir(self.cache_dir, self.config.build_name), "inputs_original.npy"
        )

    @property
    def execution_inputs_file(self):
        return os.path.join(
            output_dir(self.cache_dir, self.config.build_name), "inputs.npy"
        )

    @property
    def outputs_file(self):
        return os.path.join(
            output_dir(self.cache_dir, self.config.build_name), "outputs.npy"
        )

    @property
    def latency_file(self):
        return os.path.join(
            output_dir(self.cache_dir, self.config.build_name), "latency.npy"
        )

    @property
    def onnx_dir(self):
        return os.path.join(output_dir(self.cache_dir, self.config.build_name), "onnx")

    @property
    def base_onnx_file(self):
        return os.path.join(
            self.onnx_dir,
            f"{self.config.build_name}-op{self.info.opset}-base.onnx",
        )

    @property
    def opt_onnx_file(self):
        return os.path.join(
            self.onnx_dir,
            f"{self.config.build_name}-op{self.info.opset}-opt.onnx",
        )

    @property
    def converted_onnx_file(self):
        return os.path.join(
            self.onnx_dir,
            f"{self.config.build_name}-op{self.info.opset}-opt-f16.onnx",
        )

    @property
    def quantized_onnx_file(self):
        return os.path.join(
            self.onnx_dir,
            f"{self.config.build_name}-op{self.info.opset}-opt-quantized_int8.onnx",
        )

    @property
    def compile_dir(self):
        return os.path.join(
            output_dir(self.cache_dir, self.config.build_name), "compile"
        )

    @property
    def stats_file(self):
        return os.path.join(self.compile_dir, "stats.json")

    @property
    def groqview_file(self):
        return os.path.join(self.compile_dir, "output_bind")

    @property
    def topology(self):
        topo_a14 = {
            1: "n/a",
            2: "DF_A14_2_CHIP",
            4: "DF_A14_4_CHIP",
            8: "DF_A14_8_CHIP",
            16: "DF_A14_16_CHIP",
            32: "DF_A14_32_CHIP",
            64: "DF_A14_64_CHIP",
        }
        topo_a11 = {
            1: "n/a",
            2: "FC2_A11_2_CHIP",
            4: "FC2_A11_4_CHIP",
        }

        # Select topology based on the groqcard gen
        if self.config.groqcard == Groqcard.A11:
            return topo_a11[self.num_chips_used]
        elif self.config.groqcard == Groqcard.A14:
            return topo_a14[self.num_chips_used]
        else:
            return "Unknown"

    def save(self):
        # Create output folder if it doesn't exist
        os.makedirs(output_dir(self.cache_dir, self.config.build_name), exist_ok=True)
        os.makedirs(self.onnx_dir, exist_ok=True)
        os.makedirs(self.compile_dir, exist_ok=True)

        state_dict = {
            key: value
            for key, value in vars(self).items()
            if not key == "inputs"
            and not key == "model"
            and not key == "after_post_init"
        }

        # Special case for saving objects
        state_dict["config"] = copy.deepcopy(vars(self.config))
        state_dict["info"] = copy.deepcopy(vars(self.info))
        state_dict["config"]["groqcard"] = self.config.groqcard.value

        state_dict["model_type"] = self.model_type.value
        state_dict["build_status"] = self.build_status.value

        state_dict["info"]["backend"] = self.info.backend.value

        # During actual execution, quantization_samples in the state
        # stores the actual quantization samples.
        # However, we do not save quantization samples
        # Instead, we save a boolean to indicate whether the model
        # stored has been quantized by some samples.
        for key, value in vars(self).items():
            if key == "quantization_samples" and value is not None:
                state_dict["quantization_samples"] = True
            else:
                state_dict["quantization_samples"] = False

        with open(
            state_file(self.cache_dir, self.config.build_name), "w", encoding="utf8"
        ) as outfile:
            yaml.dump(state_dict, outfile)


def load_state(cache_dir=DEFAULT_CACHE_DIR, build_name=None, state_path=None) -> State:
    if state_path is not None:
        file_path = state_path
    elif build_name is not None:
        file_path = state_file(cache_dir, build_name)
    else:
        raise ValueError(
            "Only build_name or state_path should be set, not both or neither"
        )

    state_dict = load_yaml(file_path)

    try:
        # Special case for loading enums
        state_dict["model_type"] = ModelType(state_dict["model_type"])
        state_dict["build_status"] = Status(state_dict["build_status"])
        state_dict["config"]["groqcard"] = Groqcard(state_dict["config"]["groqcard"])
        state_dict["config"] = Config(**state_dict["config"])

        # The info section is meant to be forwards compatible with future
        # version of groqflow. Fields available in the state.yaml are copied
        # in to the new State instance, and all other fields are left
        # to their default value. Fields that existed in a previous version
        # of groqflow, but have since been removed, are ignored.

        info_tmp = {}
        for key, value in state_dict["info"].items():
            info_keys = [field.name for field in dataclasses.fields(Info)]
            if key in info_keys:
                if key == "backend":
                    info_tmp["backend"] = Backend(value)
                else:
                    info_tmp[key] = value

        state_dict["info"] = Info(**info_tmp)

        state = State(**state_dict)

    except (KeyError, TypeError) as e:
        if state_path is not None:
            path_suggestion = pathlib.Path(state_path).parent
        else:
            path_suggestion = output_dir(cache_dir, build_name)
        msg = f"""
        The cached build of this model was built with an
        incompatible older version of GroqFlow.

        Suggested solution: delete the build with
        rm -rf {path_suggestion}

        The underlying code raised this exception:
        {e}
        """
        raise exp.GroqitStateError(msg)

    return state


class Logger:
    """
    Redirects stdout to to file (and console if needed)
    """

    def __init__(self, log_path=None):
        self.debug = os.environ.get(environment_variables["debug"]) == "True"
        self.terminal = sys.stdout
        self.log_file = (
            None if log_path is None else open(log_path, "w", encoding="utf8")
        )

    def write(self, message):
        if self.log_file is not None:
            self.log_file.write(message)
        if self.debug or self.log_file is None:
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        # needed for python 3 compatibility.
        pass
