from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Utils.Log import writer


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
