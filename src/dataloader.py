import os

import gdown
import pandas as pd
import pyreadstat

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
        # if (os.path.exists(PROJECT_PATH + "data/" + file_name) or
        #         os.path.exists(PROJECT_PATH + "generated/" + file_name)):
        if gdrive_id is not None and file_name is not None:
            gdrive_link = "https://drive.google.com/uc?export=download&id="
            data_folder = os.path.join(PROJECT_PATH, "data")
            data_generated = os.path.join(PROJECT_PATH, "generated")
            os.makedirs(data_generated, exist_ok=True)
            os.makedirs(data_folder, exist_ok=True)
            file = os.path.join(data_folder, file_name)
            url = gdrive_link + gdrive_id
            gdown.download(url, file)

    @staticmethod
    def read_data(file, province=None):
        df_indiv, meta_indiv = pyreadstat.read_dta(
            file,
            usecols=["ppsort", "weight", "agegrp", "Sex",
                     "hdgree", "lfact", "TotInc",
                     "hhsize", "cfstat",
                     "cma", "pr"])
        df_indiv["pr"] = df_indiv["pr"].astype(str)
        if province is not None:
            df_indiv = df_indiv.loc[df_indiv["pr"].str.strip() == province]
        else:
            pass

        df_indiv = df_indiv.query("agegrp != 88")
        df_indiv = df_indiv.query("TotInc != 88888888")

        df_indiv["agegrp"] = pd.cut(
            x=df_indiv["agegrp"],
            bins=[0, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
        df_indiv["Sex"] = pd.cut(
            x=df_indiv["Sex"], bins=[0, 1, 2],
            labels=[0, 1])
        df_indiv["hdgree"] = pd.cut(
            x=df_indiv["hdgree"], bins=[0, 1, 2, 13, 99],
            labels=[0, 1, 2, 0], ordered=False)
        df_indiv["lfact"] = pd.cut(
            x=df_indiv["lfact"], bins=[0, 2, 10, 99],
            labels=[0, 1, 2])
        df_indiv["hhsize"] = pd.cut(
            x=df_indiv["hhsize"], bins=[0, 1, 2, 3, 4, 7, 8],
            labels=[0, 1, 2, 3, 4, 0], ordered=False)
        df_indiv["TotInc"] = pd.cut(
            x=df_indiv["TotInc"],
            bins=[-60000, 20000, 59999, 99999, 99000000, 99999999],
            labels=[0, 1, 2, 3, 0], ordered=False)

        return df_indiv
