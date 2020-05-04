##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################

import pandas as pd
from typing import Union, Iterable
import numpy as np


class Preprocess:

    def __init__(self, df: pd.DataFrame, label: str):
        self.df = self.delete_column(df, 'Id')
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

    def one_hot_encode(self, columns: list = None, sparse_matrix: bool = False):
        """
        :param columns: columns to one hot (0/1)
        :param sparse_matrix: to sparse the matrix
        :return: a new Data frame
        """
        for column in columns:
            temp_df = pd.get_dummies(self.df[column], sparse=sparse_matrix, prefix=column)
            self.df = pd.concat([self.df, temp_df], axis=1)
            self.df = self.delete_column(self.df, column)

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
        """
        mean = np.nanmean(self.df[column])
        self.df[column] = self.df[column].fillna(mean)

    def replace_nan(self):
        """
        replaces nan with column's mean
        """
        for column in self.X.columns:
            self.calc_and_fill_mean(column)
