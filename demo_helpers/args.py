import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        dest="should_build",
        default=False,
        help="If specified, will build the model to be executed on GroqChip™ processor.",
    )
    parser.add_argument(
        "-e",
        "--execute",
        action="store_true",
        dest="should_execute",
        default=False,
        help="If specified, will execute a pre-built model on GroqChip™ processor "
        "and print accuracy statistics.",
    )
    args = parser.parse_args()

    should_build = args.should_build
    should_execute = args.should_execute

    # If neither set, perform both operations
    if not (should_build or should_execute):
        should_build = True
        should_execute = True

    return {
        "rebuild_policy": "if_needed" if should_build else "never",
        "should_execute": should_execute,
    }
