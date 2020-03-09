import pandas as pd
from typing import Union, Iterable


class Preprocess:

    @staticmethod
    def delete_column(df: pd.DataFrame, columns: Union[str, Iterable]) -> pd.DataFrame:
        """
        :param df: Data frame
        :param columns: a column or columns to drop
        :return: a new Data frame
        """
        return df.drop(columns, axis=1)

    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: list = None, sparse_matrix: bool = True) -> pd.DataFrame:
        """
        :param df: Dataframe
        :param columns: columns to one hot (0/1)
        :param sparse_matrix: to sparse the matrix
        :return: a new Data frame
        """
        return pd.get_dummies(df, columns=columns, sparse=sparse_matrix)

    @staticmethod
    def split_train_test_by_pandas(df: pd.DataFrame, label: str, test_size: float = 0.2) -> [pd.DataFrame]:
        """
        :param df: Data frame
        :param label: y-label
        :param test_size: 0 to 1 how big is your test size will be
        :return: X_train, X_test, y_train, y_test
        """
        train = df.sample(frac=1 - test_size, random_state=42)
        test = df.drop(train.index)
        X_train = Preprocess.delete_column(train, label)
        y_train = train[label]
        X_test = Preprocess.delete_column(test, label)
        y_test = test[label]
        return X_train, X_test, y_train, y_test
