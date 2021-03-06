##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################

from Algorithm import Model
from Utils.Preprocess import Preprocess
from Algorithm.ThreadManager import ThreadManager
from Utils.Log import writer
from Utils.FileUtils import FileUtils
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    # create our data set

    headers = ['Id', 'OverallQual', 'YearBuilt', 'OverallCond', 'OpenPorchSF']
    df1 = FileUtils.read_data_frame_from_path('Data/train_1.csv', headers)
    df2 = FileUtils.read_data_frame_from_path('Data/train_2.xlsx', headers)
    df3 = FileUtils.read_data_frame_from_path('Data/train_targets.csv')
    df_total = df1.append(df2).merge(df3, on='Id', how='left')
    pp = Preprocess(df_total, 'SalePrice')
    pp.replace_nan()
    X_train, X_test, y_train, y_test = pp.split_train_test_by_pandas()

    # send to train
    params = [{'model': SVC, "C": 0.4}, {'model': LinearRegression,'normalize': False, "fit_intercept": False},
              {'model': LinearRegression,'normalize': True, "fit_intercept": False}, {'model': SVC, "degree": 4},
              {'model': DecisionTreeRegressor}]
    ThreadManager.running_threads_args(X_train, y_train, X_test, y_test, params)
    ThreadManager.wait_for_all_threads()
    best_params, best_result = ThreadManager.return_best_score()
    writer.info(f'The best score is {best_result} with these params: {best_params}')

    # show some graphs
    y_label = [item.score for item in ThreadManager.results.values()]
    x_label = np.arange(len(y_label))
    models_labels = [Model.get_model_name(clf) for clf in ThreadManager.results.values()]
    plt.bar(x_label, y_label)
    plt.xticks(range(len(models_labels)), models_labels)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.show()


