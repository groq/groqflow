import os
import enum
import math
from typing import Optional, List
import dataclasses
import onnxflow.common.build as of_build
from groqflow.version import __version__ as groqflow_version


DEFAULT_ONNX_OPSET = 14
MINIMUM_ONNX_OPSET = 11


# Identifiers for specific GroqCard Accelerators
GROQCARD_A14 = "A1.4"
GROQCARD_A11 = "A1.1"


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
    GROQCARD = GROQCARD_A11
else:
    GROQCARD = GROQCARD_A14


class Backend(enum.Enum):
    AUTO = "auto"
    LOCAL = "local"
    CLOUD = "cloud"
    REMOTE = "remote"


def supported_topology(groqcard: str):
    if os.environ.get(environment_variables["internal"]) == "True":
        return [1, 2, 4] if groqcard == GROQCARD_A11 else [1, 2, 4, 8, 16, 32, 64]
    else:
        return [1, 2, 4] if groqcard == GROQCARD_A11 else [1, 2, 4, 8]


def max_chips(groqcard: str):
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


@dataclasses.dataclass(frozen=True)
class GroqConfig(of_build.Config):
    """
    User-provided build configuration. GroqFlow is not allowed
    to change instances of Config once they have been
    instantiated (frozen=True enforces this).

    Inherits `build_name`, `auto_name`, `onnx_opset`, and `sequence` from onnxflow.

    Note: modifying this struct can create a breaking change that
    requires users to rebuild their models. Increment the minor
    version number of the groqflow package if you do make a build-
    breaking change.
    """

    compiler_flags: List[str] = None
    assembler_flags: List[str] = None
    groqview: bool = False
    groqcard: str = GROQCARD
    num_chips: Optional[int] = None


@dataclasses.dataclass
class GroqInfo(of_build.Info):
    """
    Information about a build that may be useful for analysis
    or debugging purposes.

    Note: GroqFlow does not guarantee that members of this class will
    have non-None values at the end of a build. GroqFlow code must
    not take a dependence on any member of this class.
    """

    num_parameters: Optional[int] = None
    opt_onnx_unsupported_ops: Optional[List[str]] = None
    opt_onnx_all_ops_supported: Optional[bool] = None
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
    compiled_onnx_input_bytes: int = None
    compiled_onnx_output_bytes: int = None
    compiler_ram_bytes: float = None


@dataclasses.dataclass
class GroqState(of_build.State):
    # User-provided args that influence the generated model
    config: GroqConfig = None

    # User-provided args that do not influence the generated model
    use_sdk: bool = False

    # Optional information about the build
    info: GroqInfo = GroqInfo()

    # All of the following are critical aspects of the build,
    # including properties of GroqFlow and choices made by GroqFlow
    # while building the model, which determine the outcome of the build.
    # NOTE: adding or changing a member name in this struct can create
    # a breaking change that requires users to rebuild their models.
    # Increment the minor version number of the groqflow package if you
    # do make a build-breaking change.

    groqflow_version: str = groqflow_version
    num_chips_used: Optional[int] = None

    @property
    def original_inputs_file(self):
        return os.path.join(
            of_build.output_dir(self.cache_dir, self.config.build_name),
            "inputs_original.npy",
        )

    @property
    def execution_inputs_file(self):
        return os.path.join(
            of_build.output_dir(self.cache_dir, self.config.build_name), "inputs.npy"
        )

    @property
    def outputs_file(self):
        return os.path.join(
            of_build.output_dir(self.cache_dir, self.config.build_name), "outputs.npy"
        )

    @property
    def latency_file(self):
        return os.path.join(
            of_build.output_dir(self.cache_dir, self.config.build_name), "latency.npy"
        )

    @property
    def compile_dir(self):
        return os.path.join(
            of_build.output_dir(self.cache_dir, self.config.build_name), "compile"
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
        if self.config.groqcard == GROQCARD_A11:
            return topo_a11[self.num_chips_used]
        elif self.config.groqcard == GROQCARD_A14:
            return topo_a14[self.num_chips_used]
        else:
            return "Unknown"

    def prepare_file_system(self):
        super().prepare_file_system()
        os.makedirs(self.compile_dir, exist_ok=True)


def load_state(
    cache_dir=DEFAULT_CACHE_DIR, build_name=None, state_path=None
) -> GroqState:

    return of_build.load_state(
        cache_dir=cache_dir,
        build_name=build_name,
        state_path=state_path,
        state_type=GroqState,
    )
