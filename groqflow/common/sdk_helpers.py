"""
Helper functions for interfacing with the GroqWare SDK
"""

import os
import enum
import subprocess
import shutil
from typing import Type, Union
from pkg_resources import parse_version
import groqflow.common.build as build
import groqflow.common.exceptions as exp


MIN_RELEASE_VERSION = "0.9.2.1"


class OS(enum.Enum):
    UBUNTU = "Ubuntu"
    ROCKY = "Rocky Linux"


def get_num_chips_available(pci_devices=None):

    # The location of lspci may vary according to the OS used
    if shutil.which("lspci"):
        lspci = shutil.which("lspci")
    # This is important to ensure that CI works
    elif os.path.isfile("/usr/bin/lspci"):
        lspci = "/usr/bin/lspci"
    else:
        raise exp.GroqModelEnvError("lspci not found")

    # Capture the list of pci devices on the system using the linux lspci utility
    if pci_devices is None:
        pci_devices = subprocess.check_output([lspci, "-n"]).decode("utf-8").split("\n")

    # Unique registered vendor id: 1de0, and device id: "0000"
    groq_card_id = "1de0:0000"

    # number of chips per device: "1de0:0000":1
    chips_per_card = 1

    # Sum the number of GroqCards in the list of devices
    num_cards = 0
    for device in pci_devices:
        if groq_card_id in device:
            num_cards += 1

    # Calculate total number of chips
    num_chips_available = num_cards * chips_per_card

    return num_chips_available


def find_tool(tool, soft_fail=False):
    if shutil.which(tool):
        return [tool]
    elif os.path.isfile(f"/usr/local/groq/bin/{tool}"):
        return [f"/usr/local/groq/bin/{tool}"]
    elif soft_fail:
        return False
    else:
        raise exp.GroqitStageError(f"{tool} not found")


def _installed_package_version(package: str, os_version: OS) -> Union[bool, str]:
    """
    This function is a simple wrapper around "apt-cache policy" that
    avoids a dependency on python-apt. It returns the installed version
    of the package when installed or "False" when not installed.
    """
    if os_version == OS.UBUNTU:
        # Get package info
        try:
            cmd = ["apt-cache", "policy", package]
            package_info = subprocess.check_output(cmd).decode("utf-8").split("\n")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise exp.GroqFlowError("apt-cache policy command failed") from e

        # Return False if package was not found
        if len(package_info) == 1:
            return False

        # Return version number
        # package_info[1] has the format "Installed: <VERSION_NUMBER>"
        return package_info[1].split(":")[1].replace(" ", "")
    elif os_version == OS.ROCKY:
        # Get package info
        cmd = ["dnf", "info", package]
        try:
            package_info = subprocess.check_output(cmd).decode("utf-8").split("\n")
        except FileNotFoundError as e:
            raise exp.GroqFlowError("dnf info command failed") from e
        except subprocess.CalledProcessError as e:
            # Return False if package was not found
            return False

        # Return version number
        # package_info[3] has the format "Version : <VERSION_NUMBER>"
        return package_info[3].split(":")[1].replace(" ", "")
    else:
        # The following exception will only be raised if a GroqFlow dev forgets to update
        # _installed_package_version() when adding support for a new OS
        raise exp.GroqitEnvError(
            f"_installed_package_version not implemented for {os_version}"
        )


def version_a_less_than_b(version_a, version_b: str):
    """
    Return true if version_a >= version_b, following the scheme:
        major.minor.patch.patchpatch~release_candidate_number

    The release_candidate_number should be ignored.
    """

    # Strip the release candidate number, if any
    clean_version_a = version_a.split("~")[0]
    clean_version_b = version_b.split("~")[0]

    return parse_version(clean_version_a) < parse_version(clean_version_b)


def version_is_valid(
    sdkv: Union[str, bool],
    required: bool,
    requirement_name: str,
    exception_type: Type[Exception] = exp.GroqitEnvError,
    hint: str = "",
):
    """
    Raise an exception if the required version number is not installed
    """

    msg = (
        f"{requirement_name}>={MIN_RELEASE_VERSION} is a required dependency "
        "for this part of GroqFlow"
    )

    # Package not found
    if not sdkv and required:
        msg = msg + f". However, {requirement_name} was not found. "
        raise exception_type(msg + hint)

    # Package found, but version is not acceptable
    elif version_a_less_than_b(sdkv, MIN_RELEASE_VERSION) and required:
        msg = msg + f" ({sdkv} is installed). "
        raise exception_type(msg + hint)


def validate_os_version() -> OS:

    supported_os_names = [x.value for x in OS]
    unsupported_os_msg = (
        "Your OS must be one of the following Linux distributions: "
        f"{', '.join(supported_os_names)}. Please refer to our installation "
        "guide for more details on supported versions."
    )

    # Check if this is a linux-based OS
    if not os.path.isfile("/etc/os-release"):
        raise exp.GroqitEnvError(unsupported_os_msg)

    # Parse OS-release data
    with open("/etc/os-release", encoding="utf-8") as f:
        os_release = {}
        for line in f:
            k, v = line.rstrip().split("=")
            os_release[k] = v.replace('"', "")

    # Check if OS is supported
    if os_release["NAME"] not in supported_os_names:
        raise exp.GroqitEnvError(unsupported_os_msg)

    return OS(os_release["NAME"])


def validate_devtools(
    os_version: OS,
    required=False,
    exception_type: Type[Exception] = exp.GroqitEnvError,
) -> Union[bool, str]:
    version = _installed_package_version("groq-devtools", os_version)
    hint = "Please contact sales@groq.com to get access to groq-devtools."
    version_is_valid(version, required, "groq-devtools", exception_type, hint)


def validate_runtime(
    os_version: OS,
    required=False,
    exception_type: Type[Exception] = exp.GroqitEnvError,
) -> Union[bool, str]:
    version = _installed_package_version("groq-runtime", os_version)
    hint = "Please contact sales@groq.com to get access to groq-runtime."
    version_is_valid(version, required, "groq-runtime", exception_type, hint)


# Returns the root directory of the current git repo and any associated
# error from running the git command
def get_repo_root():
    p = subprocess.Popen(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = p.communicate()
    repo = out.decode("utf-8")
    repo = repo.rstrip("\n")
    err = err.decode("utf-8")
    return repo, err


def validate_bake():
    if not shutil.which("bake"):
        raise exp.GroqitEnvError(
            (
                "Bake must be available when the env var "
                f'{build.environment_variables["dont_use_sdk"]} is set to True'
            )
        )

    # bake commands require Groq to be current git repo
    repo, err = get_repo_root()
    groq_root = repo.split("/")[-1] == "Groq"

    if err:
        raise exp.GroqitEnvError(
            (
                "You must be inside the Groq repo when the env var "
                f'{build.environment_variables["dont_use_sdk"]} is set to True. '
                f"groqit() returned with error {err}"
            )
        )

    elif not groq_root:
        raise exp.GroqitEnvError(
            (
                "You must be inside the Groq repo when the env var "
                f'{build.environment_variables["dont_use_sdk"]} is set to True. '
                f"groqit() detected you are inside repo {repo}"
            )
        )


def check_dependencies(
    require_devtools: bool = False,
    require_runtime: bool = False,
    exception_type: Type[Exception] = exp.GroqitEnvError,
):

    # Skip dependency check if necessary
    if os.environ.get("GROQFLOW_SKIP_SDK_CHECK") == "True":
        return True

    # Check for bake if SDK is not being used
    if not build.USE_SDK:
        validate_bake()
    # Check for the different SDK components when using the SDK
    # Skip all checks if using CI
    else:
        os_version = validate_os_version()
        validate_devtools(
            os_version=os_version,
            required=require_devtools,
            exception_type=exception_type,
        )
        validate_runtime(
            os_version=os_version,
            required=require_runtime,
            exception_type=exception_type,
        )
