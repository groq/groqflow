"""
Helper functions for managing the GroqFlow cache
"""

import os
import shutil


def rmdir(folder):
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)

            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        shutil.rmtree(folder)

        return True

    else:
        return False


def get_all(path, exclude_path=False, file_type="state.yaml", recursive=True):
    if recursive:
        files = [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(path)
            for f in filenames
            if file_type in f
        ]
    else:
        files = []
        dp, _, filenames = os.walk(path)
        for f in filenames:
            if file_type in f:
                files.append(os.path.join(dp, f))

    if exclude_path:
        files = [os.path.basename(f) for f in files]

    return files
