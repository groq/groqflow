from contextlib import contextmanager
import os
import sys
import subprocess

import pkg_resources


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def check_deps(script_filepath):
    dir_path = os.path.dirname(os.path.realpath(script_filepath))
    reqs_filepath = os.path.join(dir_path, "requirements.txt")
    with open(reqs_filepath, "r", encoding="utf-8") as f:
        reqs = pkg_resources.parse_requirements(f)
        str_reqs = [str(req) for req in reqs]
        try:
            with suppress_stdout():
                for req in str_reqs:
                    pkg_resources.require(str(req))
        except pkg_resources.DistributionNotFound as e:
            print("Some required packages below are missing:\n")
            reqs = pkg_resources.parse_requirements(f)
            for req in str_reqs:
                print(str(req))
            print()
            reply = None
            question = "Install missing pacakges (y/n): "
            while reply not in ["y", "n"]:
                reply = str(input(question)).lower().strip()
            if reply == "n":
                raise e
            subprocess.check_call(["pip", "install", "-r", reqs_filepath])
