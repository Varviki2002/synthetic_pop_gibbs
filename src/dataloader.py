import os

import requests
import pandas as pd
import gdown

from src import PROJECT_PATH


class Downloader:
    """
    This class downloads the data.
    """
    def __init__(self, gdrive_id: str = None, file_name: str = None) -> None:
        """
        Constructor for downloading and loading the data file.
        :param str gdrive_id: Google Drive id
        :param str file_name: file name for saving
        """
        if gdrive_id is not None and file_name is not None:
            gdrive_link = "https://drive.google.com/uc?export=download&id="
            data_folder = os.path.join(PROJECT_PATH, "data")
            data_generated = os.path.join(PROJECT_PATH, "generated")
            os.makedirs(data_generated, exist_ok=True)
            os.path.join(data_folder, "generated")
            os.makedirs(data_folder, exist_ok=True)
            file = os.path.join(data_folder, file_name)
            url = gdrive_link + gdrive_id
            output = file_name
            gdown.download(url, file)

            self.data = pd.read_csv(file)