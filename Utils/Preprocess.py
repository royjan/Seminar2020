import pandas as pd
from typing import Union, Iterable
import numpy as np
from pandas import Series


class Preprocess:

    def __init__(self, df: pd.DataFrame, label: str):
        self.df = df
        self.label = label

    @property
    def y(self) -> pd.Series:
        return self.df[self.label]

    @property
    def X(self) -> pd.DataFrame:
        return self.delete_column(self.df, self.label)

    @staticmethod
    def delete_column(df: pd.DataFrame, columns: Union[str, Iterable]) -> pd.DataFrame:
        """
        :param df: Data frame
        :param columns: a column or columns to drop
        :return: a new Data frame
        """
        return df.drop(columns, axis=1)

    def one_hot_encode(self, columns: list = None, sparse_matrix: bool = True) -> pd.DataFrame:
        """
        :param columns: columns to one hot (0/1)
        :param sparse_matrix: to sparse the matrix
        :return: a new Data frame
        """
        return pd.get_dummies(self.df, columns=columns, sparse=sparse_matrix)

    def split_train_test_by_pandas(self, test_size: float = 0.2) -> [pd.DataFrame]:
        """
        :param test_size: 0 to 1 how big is your test size will be
        :return: X_train, X_test, y_train, y_test
        """
        train = self.df.sample(frac=1 - test_size, random_state=42)
        test = self.df.drop(train.index)
        X_train = Preprocess.delete_column(train, self.label)
        y_train = train[self.label]
        X_test = Preprocess.delete_column(test, self.label)
        y_test = test[self.label]
        return X_train, X_test, y_train, y_test

    def calc_and_fill_mean(self, column: str) -> [float]:
        """
        :param column: which column to calculate
        :return: mean of the column
        """
        mean = np.nanmean(self.X[column])
        self.X[column] = self.X[column].fillna(mean)
        return mean

    def replace_nan(self) -> pd.DataFrame:
        """
        This function replace nan with column's mean
        """
        means = [self.calc_and_fill_mean(column) for column in self.X]
        means_df = pd.DataFrame([means], columns=self.X.columns)
        return means_df

##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################