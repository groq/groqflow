"""
Helper functions for interfacing with the GroqWare SDK
"""

import os
import re
import subprocess
import shutil
from typing import Type, Union
import pkg_resources
import groqflow.common.exceptions as exp
import groqflow.common.build as build


def get_num_chips_available(pci_devices=None):

    # Check if we have access to lspci
    if not shutil.which("/usr/bin/lspci"):
        raise exp.GroqModelEnvError("lspci not found")

    # Capture the list of pci devices on the system using the linux lspci utility
    if pci_devices is None:
        pci_devices = (
            subprocess.check_output(["/usr/bin/lspci", "-n"])
            .decode("utf-8")
            .split("\n")
        )

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


def _installed_package_version(package: str) -> Union[bool, str]:
    """
    This function is a simple wrapper around "apt-cache policy" that
    avoids a dependency on python-apt. It returns the installed version
    of the package when installed or "False" when not installed.
    """
    cmd = ["apt-cache", "policy", package]
    try:
        apt_cache_policy = subprocess.check_output(cmd).decode("utf-8").split("\n")
        if len(apt_cache_policy) == 1:
            # Package not found
            return False
        else:
            # Package found
            return re.search(r"(?<=Installed: ).*", apt_cache_policy[1]).group()
    except subprocess.CalledProcessError:
        msg = "apt-cache policy command failed"
        raise exp.GroqFlowError(msg)
    except FileNotFoundError:
        # apt-cache policy command failed or not found
        # TODO: apt-cache must be found by CI once CI switches to the new SDK
        return False


def version_is_valid(
    sdkv: Union[str, bool],
    required: bool,
    requirement_name: str,
    exception_type: Type[Exception] = exp.GroqitEnvError,
    hint: str = "",
) -> bool:

    # TODO: update this logic once more versions are valid
    valid_release_version = "0.9.0"
    valid_test_version = "test"

    msg = (
        f"{requirement_name} {valid_release_version} is a required dependence "
        "for this part of GroqFlow"
    )

    # Package not found
    if not sdkv:
        if required:
            msg = msg + f". However, {requirement_name} was not found. "
            raise exception_type(msg + hint)
        else:
            return False
    # Package found, but version is not acceptable
    elif sdkv != valid_release_version and sdkv != valid_test_version:
        if required:
            msg = msg + f" ({sdkv} is installed). "
            raise exception_type(msg + hint)
        else:
            return False
    # Package found and has the right version
    else:
        return True


def validate_devtools(
    required=False, exception_type: Type[Exception] = exp.GroqitEnvError
) -> Union[bool, str]:
    version = _installed_package_version("groq-devtools")
    hint = "Please contact sales@groq.com to get access to groq-devtools."
    return version_is_valid(version, required, "groq-devtools", exception_type, hint)


def validate_runtime(
    required=False, exception_type: Type[Exception] = exp.GroqitEnvError
) -> Union[bool, str]:
    version = _installed_package_version("groq-runtime")
    hint = "Please contact sales@groq.com to get access to groq-runtime."
    return version_is_valid(version, required, "groq-runtime", exception_type, hint)


def validate_groqapi(
    required=False, exception_type: Type[Exception] = exp.GroqitEnvError
) -> Union[bool, str]:
    try:
        version = pkg_resources.get_distribution("groq").version
    except pkg_resources.DistributionNotFound:
        version = False
    hint = (
        "Make sure to install groq-devtools and "
        "add /opt/groq/runtime/site-packages to your PYTHONPATH."
    )
    return version_is_valid(version, required, "Groq API", exception_type, hint)


# Return the result of bake groot
def _bake_groot():
    p = subprocess.Popen(
        ["bake", "groot"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    repo, err = p.communicate()
    repo = repo.decode("utf-8")
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

    repo, err = _bake_groot()

    if err and repo:
        raise exp.GroqitEnvError(
            (
                "You must be inside the Groq repo when the env var "
                f'{build.environment_variables["dont_use_sdk"]} is set to True. '
                f"groqit() detected you are inside repo {repo}"
            )
        )

    if err:
        raise exp.GroqitEnvError(
            (
                "You must be inside the Groq repo when the env var "
                f'{build.environment_variables["dont_use_sdk"]} is set to True'
            )
        )


def check_dependencies(
    require_devtools: bool = False,
    require_runtime: bool = False,
    require_groqapi: bool = False,
    exception_type: Type[Exception] = exp.GroqitEnvError,
) -> bool:

    # Check for bake if SDK is not being used
    if not build.USE_SDK:
        validate_bake()
    # Check for the different SDK components when using the SDK
    # Skip all checks if using CI
    elif not os.environ.get("GROQFLOW_SKIP_SDK_CHECK") == "True":
        validate_devtools(required=require_devtools, exception_type=exception_type)
        validate_runtime(required=require_runtime, exception_type=exception_type)
        validate_groqapi(required=require_groqapi, exception_type=exception_type)

    return True
