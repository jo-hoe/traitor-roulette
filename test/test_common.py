
from datetime import datetime
from src.common import generate_filepath


def test_generate_filepath():
    extension = "txt"
    filename = "test_filename"

    base_filename = f"{filename}.{extension}"

    filepath = generate_filepath(base_filename)

    assert filepath.endswith(extension)
    assert filename in filepath
    assert datetime.now().strftime("%Y") in filepath
    assert "output" in filepath
