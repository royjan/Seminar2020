from typing import Tuple

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from Utils.Log import writer
from Utils.FileUtils import FileUtils


def running_threads_args(X_train, y_train, X_test, y_test, params: [dict]) -> Tuple[int, dict]:
    """
    :param X_train: X train data
    :param y_train: y train Data
    :param X_test: X test Data
    :param y_test: y test Data
    :param params: Hyper parameters to check which set of parameters is the best set for our model
    :return: Tuple of best result and which set of parameters
    """
    from threading import Thread
    import numpy as np
    threads, results = [], []
    best_result = np.inf
    best_params = params[0]
    for param in params:
        thread = Thread(name=str(param), target=learning_model, args=(X_train, y_train, X_test, y_test), kwargs=param)
        param['result'] = results
        writer.debug(f'Thread [{thread.getName()}] is running')
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
        writer.debug(f'Thread [{thread.getName()}] is finished')
    for param, result in zip(params, results):
        param.pop('result')
        if result < best_result:
            best_result = result
            best_params = param
    return best_result, best_params


def learning_model(X_train, y_train, X_test, y_test, **kwargs):
    """
    :param X_train: X train data
    :param y_train: y train Data
    :param X_test: X test Data
    :param y_test: y test Data
    :param kwargs: Hyper parameters
    """
    results = kwargs.pop('result')
    clf = LinearRegression(**kwargs).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    writer.debug(f"set of params: {kwargs} scored {score}")
    results.append(score)


if __name__ == '__main__':
    headers = ['Id', 'OverallQual', 'YearBuilt', 'OverallCond', 'OpenPorchSF']
    df1 = FileUtils.read_data_frame_from_path('Data/train_1.csv', headers)
    df2 = FileUtils.read_data_frame_from_path('Data/train_2.xlsx', headers)
    df3 = FileUtils.read_data_frame_from_path('Data/train_targets.csv')
    df_total = df1.append(df2).merge(df3, on='Id', how='left').fillna("None")
    df_X = df_total.drop(['Id', 'SalePrice'], axis=1)
    df_y = df_total['SalePrice']
    # df_X_ohe = pd.get_dummies(df_X)
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y)

    ### Linear VS SVC

    params = [{"fit_intercept": True}, {"fit_intercept": False}, {'normalize': False}]
    best_result, best_params = running_threads_args(X_train, y_train, X_test, y_test, params)
    writer.info(f'The best score is {best_result} with these params: {best_params}')
    # Plot outputs
