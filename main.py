from Utils.Preprocess import Preprocess
from Algorithm.ThreadManager import ThreadManager
from Utils.Log import writer
from Utils.FileUtils import FileUtils

if __name__ == '__main__':
    headers = ['Id', 'OverallQual', 'YearBuilt', 'OverallCond', 'OpenPorchSF']
    df1 = FileUtils.read_data_frame_from_path('Data/train_1.csv', headers)
    df2 = FileUtils.read_data_frame_from_path('Data/train_2.xlsx', headers)
    df3 = FileUtils.read_data_frame_from_path('Data/train_targets.csv')
    df_total = df1.append(df2).merge(df3, on='Id', how='left').fillna("None")
    X_train, X_test, y_train, y_test = Preprocess.split_train_test_by_pandas(df_total, 'SalePrice')

    ### Linear VS SVC

    params = [{"fit_intercept": True}, {'normalize': False, "fit_intercept": False}]
    ThreadManager.running_threads_args(X_train, y_train, X_test, y_test, params)
    ThreadManager.wait_for_all_threads()
    best_result, best_params = ThreadManager.return_best_score(params)

    writer.info(f'The best score is {best_result} with these params: {best_params}')
    # Plot outputs
