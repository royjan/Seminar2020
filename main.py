from Utils.Preprocess import Preprocess
from Algorithm.ThreadManager import ThreadManager
from Utils.Log import writer
from Utils.FileUtils import FileUtils
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

if __name__ == '__main__':
    # create our data set
    headers = ['Id', 'OverallQual', 'YearBuilt', 'OverallCond', 'OpenPorchSF']
    df1 = FileUtils.read_data_frame_from_path('Data/train_1.csv', headers)
    df2 = FileUtils.read_data_frame_from_path('Data/train_2.xlsx', headers)
    df3 = FileUtils.read_data_frame_from_path('Data/train_targets.csv')
    df_total = df1.append(df2).merge(df3, on='Id', how='left').fillna("None")
    X_train, X_test, y_train, y_test = Preprocess.split_train_test_by_pandas(df_total, 'SalePrice')

    # send to train
    params = [{'model': SVC, "C": 0.4}, {'normalize': False, "fit_intercept": False},
              {'normalize': True, "fit_intercept": False}, {'model': SVC, "degree": 4},
              {'model': LogisticRegression, 'C': 0.6}]
    ThreadManager.running_threads_args(X_train, y_train, X_test, y_test, params)
    ThreadManager.wait_for_all_threads()
    best_result, best_params = ThreadManager.return_best_score()
    writer.info(f'The best score is {best_result} with these params: {best_params}')

    # show some graphs
    y_label = [item.score for item in ThreadManager.results]
    x_label = np.arange(len(y_label))
    models_labels = [str(item.params).split("(")[0] for item in ThreadManager.results]
    plt.bar(x_label, y_label)
    plt.xticks(range(len(models_labels)), models_labels)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.show()
