##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################

import pandas as pd
from Utils.Log import logger, writer


class FileUtils:

    @staticmethod
    def read_data_frame_from_path(file_path: str, headers=None) -> pd.DataFrame:
        """
        :param file_path: absolute or relative path name
        :param headers: list of columns to filter, default is None (all)
        :return: DataFrame structure
        """
        import os
        file_name, file_extension = os.path.splitext(file_path)
        if file_extension == '.csv':
            writer.debug(f"Reading file '{file_name}{file_extension}' with pandas | pd.read_csv")
            return pd.read_csv(file_path, index_col=False, usecols=headers)
        elif file_extension == '.xlsx':
            writer.debug(f"Reading file '{file_name}{file_extension}' with pandas | pd.read_excel")
            return pd.read_excel(file_path, index_col=False, usecols=headers)
        elif file_extension == '.pkl':
            writer.debug(f"Reading file '{file_name}{file_extension}' with pandas | pd.read_pickle")
            return pd.read_pickle(file_path)

    @staticmethod
    def read_models_from_text() -> [dict]:
        """
        reads the models from file and returns it as list of dictionaries
        """
        from Algorithm.Model import return_model_by_name
        import json
        with open("Algorithm/models.txt") as file:
            data = json.loads(file.read())
        for item in data:
            for key in item:
                if FileUtils.is_boolean(item[key]):
                    item[key] = item[key].lower() == "true"
                if key.lower() == "model":
                    item["model"] = return_model_by_name(item["model"])
        return data

    @staticmethod
    def is_boolean(value: str) -> bool:
        """
        :param value: check if value is a boolean string
        :return: bool (true / false)
        """
        return type(value) is str and value.lower() in ["true", "false"]
