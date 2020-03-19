##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from Utils.Log import writer


def learning_model(X_train, y_train, X_test, y_test, model=LinearRegression, **kwargs):
    """
    :param model: which model to use
    :param X_train: X train data
    :param y_train: y train Data
    :param X_test: X test Data
    :param y_test: y test Data
    :param kwargs: Hyper parameters
    """
    from Algorithm.ThreadManager import ThreadManager
    results = kwargs.pop('result')
    clf = model(**kwargs).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = mean_squared_error(y_test, y_pred)
    writer.debug(f"set of params: {clf} scored {score}")
    results.append(ThreadManager.result_struct(params=clf, score=score))


def get_model_name(clf) -> str:
    """
    :param clf: classifier object
    :return: a string with classifier name
    """
    return str(clf.params).split("(")[0]
