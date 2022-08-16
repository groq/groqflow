import sys
import groq.api as g


def assembler_multichip(topology, compile_dir):

    # FIXME: topo_config is defined in two files, both assembler_multichip.py
    #  and benchmark.py. If you modify this code, make sure to modify it in
    #  both places. We will remove this code replication when we are able to
    #  import groqflow packages into these files.

    # Identify the topology. The topology specified with
    # groq-compiler should match the one configured here.
    topo_config = {
        "DF_A14_2_CHIP": g.TopologyConfig.DF_A14_2_CHIP,
        "DF_A14_4_CHIP": g.TopologyConfig.DF_A14_4_CHIP,
        "DF_A14_8_CHIP": g.TopologyConfig.DF_A14_8_CHIP,
        "FC2_A11_2_CHIP": g.TopologyConfig.FC2_A11_2_CHIP,
        "FC2_A11_4_CHIP": g.TopologyConfig.FC2_A11_4_CHIP,
    }

    # Select topology
    topo = g.configure_topology(config=topo_config[topology])

    # Initiate the program package object with package name and output directory
    md_pgm_pkg = g.ProgramPackage(name="output", output_dir=compile_dir)

    # assign the name and topology to the create_program_context
    pgm_ctx = md_pgm_pkg.create_program_context("output", topo)

    # add the .aa files created by the groq-compiler and add them to the program
    md_pgm_pkg.add_precompiled_program(pgm_ctx, compile_dir, "output")

    # if any extra instruction memory slices were defined
    # during groq-compiler add them here.
    extra_slices = []

    # The .assemble method takes all the files and topologies and
    # assembles the multi chip program package.
    print("Starting multi-chip assembling process", flush=True)
    md_pgm_pkg.assemble(extra_ifetch_slices=extra_slices, ifetch_from_self=True)


if __name__ == "__main__":

    # Checking if the right number of inputs has been received
    args = sys.argv
    expected_num_args = 3
    if len(args) != expected_num_args:
        print(len)
        raise ValueError(
            f"Wrong number of input arguments to assemble_multichip.py's main"
            f" function: got {len(args)} but expected {expected_num_args}"
        )

    # Run script
    topology = args[1]
    compile_dir = args[2]
    assembler_multichip(topology, compile_dir)
