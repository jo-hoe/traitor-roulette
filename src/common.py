
from datetime import datetime
import os


def get_output_dir_path() -> str:
    """
    Returns the path to the output folder.
    """
    src_path = os.path.dirname(os.path.realpath(__file__))
    root_path = os.path.dirname(src_path)
    return os.path.join(root_path, "output")


def generate_filepath(base_filename: str) -> str:
    """
    Creates a file path to an output folder.
    """
    parts = base_filename.split(".")
    file_extension = parts[len(parts) - 1]
    file_name = "".join(parts[0:len(parts) - 1])

    date = datetime.now().strftime("%Y%m%d_%H%M")

    output_path = os.path.join(
        get_output_dir_path(), f"{file_name}_{date}.{file_extension}")

    return output_path
