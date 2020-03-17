##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################

import pandas as pd


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
            return pd.read_csv(file_path, index_col=False, usecols=headers)
        elif file_extension == '.xlsx':
            return pd.read_excel(file_path, index_col=False, usecols=headers)
        elif file_extension == '.pkl':
            return pd.read_pickle(file_path)


