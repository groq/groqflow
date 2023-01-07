"""
The following script is used to get the latency and outputs of a given run on the GroqChip.
This script doesn't depend on GroqFlow to be executed.
"""
# pylint: disable = no-name-in-module
# pylint: disable = import-error
import argparse
from timeit import Timer
from typing import Tuple, List
import numpy as np
import groq.api as g
import groq.runner.tsp as tsp


def get_multi_tsp_runner(
    compile_dir: str, topology: str, bringup_topology: bool = False
) -> tsp.local_runner.MultichipTSPRunner:

    # FIXME: topo_config is defined in two files, both assembler_multichip.py
    #  and execute.py. If you modify this code, make sure to modify it in
    #  both places. We will remove this code replication when we are able to
    #  import the groqit.misc package into execute.py.

    # Declare different topologies
    topo_config = {
        "DF_A14_2_CHIP": g.TopologyConfig.DF_A14_2_CHIP,
        "DF_A14_4_CHIP": g.TopologyConfig.DF_A14_4_CHIP,
        "DF_A14_8_CHIP": g.TopologyConfig.DF_A14_8_CHIP,
        "FC2_A11_2_CHIP": g.TopologyConfig.FC2_A11_2_CHIP,
        "FC2_A11_4_CHIP": g.TopologyConfig.FC2_A11_4_CHIP,
    }
    speed_config = {
        "DF_A14_2_CHIP": 25,
        "DF_A14_4_CHIP": 25,
        "DF_A14_8_CHIP": 25,
        "FC2_A11_2_CHIP": 30,
        "FC2_A11_4_CHIP": 30,
    }

    if bringup_topology:
        print("Bringup C2C topology...")
        tsp.bringup_topology(
            user_config=topo_config[topology], speed=speed_config[topology]
        )

    program_name = "output"
    tsp_runner = tsp.create_multi_tsp_runner(
        program_name,
        compile_dir,
        program_name,
        user_config=topo_config[topology],
        speed=speed_config[topology],
    )
    return tsp_runner


def rtime(func, num_times: int, *args, **kwargs) -> Tuple[float, List]:
    """
    Measure time of a given function multiple times and return
    the average time in seconds
    """
    output_container = []

    def wrapper():
        output_container.append(func(*args, **kwargs))

    timer = Timer(wrapper)
    delta = timer.timeit(num_times)
    return delta, output_container.pop()


def run(
    input_batch: np.ndarray,
    num_chips: int,
    output_dir: str,
    topology: str,
    bringup_topology: bool,
    repetitions=1,
) -> Tuple[float, List]:

    # Get tsp_runner
    if num_chips == 1:
        iop_file = f"{output_dir}/compile/output.iop"
        tsp_runner = tsp.create_tsp_runner(iop_file)
    else:
        compile_dir = f"{output_dir}/compile"
        tsp_runner = get_multi_tsp_runner(compile_dir, topology, bringup_topology)

    # Multi-TSP Runner will run a pipeline of inputs
    # through the entire topology of the program 1-chip at a time
    # to get the actual output from the entire graph we need to invoke `num_chip` times
    def forward_multichip(example):
        for _ in range(num_chips):
            output = tsp_runner(**example)
        return output

    # Forward function for models compiled for a single chip
    def forward_singlechip(example):
        return tsp_runner(**example)

    forward = forward_singlechip if num_chips == 1 else forward_multichip
    batch_size = len(input_batch)
    output_batch = []
    total_latency = 0
    for idx in range(batch_size):
        example = input_batch[idx]
        latency, output = rtime(forward, repetitions, example)
        total_latency += latency
        output_batch.append(output)

    total_latency = total_latency / repetitions / batch_size

    return total_latency, output_batch


if __name__ == "__main__":

    # Disabling lint warning for using pickle
    # pylint: disable = unexpected-keyword-arg

    # Terminology:
    # This function receives a batch of inputs (input_batch)
    # Each element of this batch is called an "example"
    # Each example may contain one or more arguments

    # Parse Inputs
    parser = argparse.ArgumentParser(description="Execute models built by GroqFlow")
    parser.add_argument(
        "num_chips",
        type=int,
        help="Number of chips used to build the model",
    )
    parser.add_argument("output_dir", help="Path where the build files are stored")
    parser.add_argument("outputs_file", help="File in which the outputs will be saved")
    parser.add_argument("latency_file", help="File in which the latency will be saved")
    parser.set_defaults(bringup_topology=False)
    parser.add_argument("topology", help="GroqChip topology used when building model")
    parser.add_argument(
        "repetitions",
        type=int,
        help="Number of times to execute the received inputs",
    )
    parser.add_argument(
        "--bringup_topology",
        help="Describes whether or not the topology should be initialized",
        action="store_true",
    )
    args = vars(parser.parse_args())

    # Read inputs
    input_file = f"{args['output_dir']}/inputs.npy"
    input_batch = np.load(input_file, allow_pickle=True)

    # Get latency/output_data
    latency, output_data = run(
        input_batch,
        args["num_chips"],
        args["output_dir"],
        args["topology"],
        args["bringup_topology"],
        repetitions=args["repetitions"],
    )

    # Save results to file
    with open(args["outputs_file"], "wb") as f:
        np.save(args["outputs_file"], output_data)
    with open(args["latency_file"], "wb") as f:
        np.save(args["latency_file"], latency)
