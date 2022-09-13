import os
import sys
from typing import Tuple, Union
from stat import S_ISDIR
import yaml
import paramiko
import groqflow.common.exceptions as exp
import groqflow.common.build as build
import groqflow.common.sdk_helpers as sdk


class MySFTPClient(paramiko.SFTPClient):
    def put_dir(self, source, target) -> None:
        # Removes previous directory before transferring
        for item in os.listdir(source):
            if ".aa" in item or ".onnx" in item or ".json" in item or ".npy" in item:
                continue
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), "%s/%s" % (target, item))
            else:
                self.mkdir("%s/%s" % (target, item))
                self.put_dir(os.path.join(source, item), "%s/%s" % (target, item))

    def is_dir(self, path) -> bool:
        try:
            return S_ISDIR(self.stat(path).st_mode)
        except IOError:
            return False

    def rm_dir(self, path) -> None:
        files = self.listdir(path)
        for f in files:
            filepath = os.path.join(path, f)
            if self.is_dir(filepath):
                self.rm_dir(filepath)
            else:
                self.remove(filepath)

    def mkdir(self, path, mode=511) -> None:
        try:
            super(MySFTPClient, self).mkdir(path, mode)
        except IOError:
            self.rm_dir(path)


def load_remote_config() -> Union[Tuple[str, str], Tuple[None, None]]:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file_path = f"{dir_path}/config.yaml"

    # Create a configuration file if one doesn't exist already
    if not os.path.exists(config_file_path):
        conf = {"remote_machine": {"ip": None, "username": None}}
        with open(config_file_path, "w", encoding="utf8") as outfile:
            yaml.dump(conf, outfile)

    # Return the contents of the configuration file
    config_file = open(config_file_path, encoding="utf8")
    conf = yaml.load(config_file, Loader=yaml.FullLoader)
    return (
        conf["remote_machine"]["ip"],
        conf["remote_machine"]["username"],
    )


def save_remote_config(ip, username) -> None:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_file = open(f"{dir_path}/config.yaml", encoding="utf8")
    conf = yaml.load(config_file, Loader=yaml.FullLoader)
    conf["remote_machine"]["ip"] = ip
    conf["remote_machine"]["username"] = username
    with open(f"{dir_path}/config.yaml", "w", encoding="utf8") as outfile:
        yaml.dump(conf, outfile)


def connect_to_host(ip, username) -> paramiko.SSHClient:
    print(f"Connecting to {username}@{ip}")

    class AllowAllKeys(paramiko.MissingHostKeyPolicy):
        def missing_host_key(self, client, hostname, key):
            return

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
    client.set_missing_host_key_policy(AllowAllKeys())
    client.connect(ip, username=username)
    return client


def exec_command(client, command, ignore_error=False) -> Tuple[str, str]:
    _, stdout, stderr = client.exec_command(command)
    exit_code = stdout.channel.recv_exit_status()
    stdout = stdout.read().decode("ascii").strip("\n")
    stderr = str(stderr.read(), "utf-8")
    if not ignore_error:
        print(stderr)

    return stdout, exit_code


def configure_remote() -> Tuple[str, str]:
    # Load stored values
    ip, username = load_remote_config()

    if ip is None or username is None:
        # Print message
        print("\n*** First Time Groq Cloud Setup ***")
        print("Step 1 - Create a VM following this guide: go/cloudquickstart")
        print("Step 2 - SSH into your vm, cd into the SDK and install it (./install)")
        print("Step 3 - Provide your instance IP and hostname below")

        # Get IP
        while ip is None or ip == "":
            ip = input("Groq Cloud IP: ")

        # Get username
        if username is None:
            username_input = input(f"Username for {ip} (default: ubuntu): ")
            if username_input == "":
                username = "ubuntu"
            else:
                username = username_input

        # Store information on yaml file
        save_remote_config(ip, username)

    return ip, username


def setup_host(client) -> None:
    # Make sure at least one GroqChip is available remotely
    stdout, exit_code = exec_command(client, "/usr/bin/lspci")
    if stdout == "" or exit_code == 1:
        msg = "Failed to run lspci to get GroqChips available"
        raise exp.GroqModelRuntimeError(msg)
    num_chips_available = sdk.get_num_chips_available(stdout.split("\n"))
    if num_chips_available < 1:
        raise exp.GroqModelRuntimeError("No GroqChips found")
    print(f"{num_chips_available} GroqChips found")

    # Transfer common files to host
    exec_command(client, "mkdir groqflow_remote_cache", ignore_error=True)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.put(f"{dir_path}/execute.py", "groqflow_remote_cache/execute.py")


def setup_connection() -> paramiko.SSHClient:
    # Setup authentication scheme if needed
    ip, username = configure_remote()

    # Connect to host
    client = connect_to_host(ip, username)

    # Check for GroqChips and transfer common files
    setup_host(client)

    return client


def execute_remotely(
    bringup_topology: bool,
    repetitions: int,
    state: build.State,
    log_execute_path: str,
) -> None:
    """
    Execute GroqModel on the remote machine rather than locally
    """

    # Ask the user for credentials if needed
    configure_remote()

    # Redirect all stdout to log_file
    sys.stdout = build.Logger(log_execute_path)

    # Connect to remote machine and transfer common files
    client = setup_connection()

    # Transfer iop and inputs file
    print("Transferring model and inputs...")
    if not os.path.exists(state.execution_inputs_file):
        msg = "Model input file not found"
        raise exp.GroqModelRuntimeError(msg)

    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.mkdir("groqflow_remote_cache/compile")
        s.put_dir(state.compile_dir, "groqflow_remote_cache/compile")
        s.put(state.execution_inputs_file, "groqflow_remote_cache/inputs.npy")

    # Run benchmarking script
    output_dir = "groqflow_remote_cache"
    remote_outputs_file = "groqflow_remote_cache/outputs.npy"
    remote_latency_file = "groqflow_remote_cache/latency.npy"
    print("Running benchmarking script...")
    bringup_topology_arg = "" if bringup_topology else "--bringup_topology"
    _, exit_code = exec_command(
        client,
        (
            f"/usr/local/groq/bin/python groqflow_remote_cache/execute.py "
            f"{state.num_chips_used} {output_dir} {remote_outputs_file} "
            f"{remote_latency_file} {state.topology} {repetitions} "
            f"{bringup_topology_arg}"
        ),
    )
    if exit_code == 1:
        msg = f"""
        Failed to execute GroqChip(s) remotely.
        Look at **{log_execute_path}** for details.
        """
        raise exp.GroqModelRuntimeError(msg)

    # Get output files back
    with MySFTPClient.from_transport(client.get_transport()) as s:
        s.get(remote_outputs_file, state.outputs_file)
        s.get(remote_latency_file, state.latency_file)
        s.remove(remote_outputs_file)
        s.remove(remote_latency_file)

    # Stop redirecting stdout
    sys.stdout = sys.stdout.terminal
