import requests
from requests import Response
from tempfile import NamedTemporaryFile
from contextlib import contextmanager


@contextmanager
def response_to_temp_file(response) -> NamedTemporaryFile:
    with NamedTemporaryFile(delete=True) as tmp_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive new chunks
                tmp_file.write(chunk)

        tmp_file.seek(0)
        yield tmp_file
