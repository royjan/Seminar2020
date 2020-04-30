##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################

from collections import namedtuple
from threading import Thread
from typing import Tuple

from Algorithm.Model import learning_model
from Utils.Log import writer

TIMEOUT_THREAD = 10  # seconds


class ThreadManager:
    _threads = {}
    result_struct = namedtuple('Results', ['params', 'score'])
    results = {}

    @classmethod
    def reset_values(cls):
        cls._threads = {}
        cls.results = {}

    @classmethod
    def thread_task(cls, index, X_train, y_train, X_test, y_test, params: [dict]):
        cls.results[index] = learning_model(X_train, y_train, X_test, y_test, **params)

    @classmethod
    def running_threads_args(cls, X_train, y_train, X_test, y_test, params: [dict]):
        """
        :param X_train: X train data
        :param y_train: y train Data
        :param X_test: X test Data
        :param y_test: y test Data
        :param params: Hyper parameters to check which set of parameters is the best set for our model
        :return: list of the threads
        """
        for index, param in enumerate(params):
            thread = Thread(name=str(param), target=ThreadManager.thread_task,
                            args=(index, X_train, y_train, X_test, y_test, param))

            thread.start()
            writer.debug(f'Thread [{thread.getName()}] is running')
            cls._threads[index] = thread
        return cls.results

    @classmethod
    def wait_for_all_threads(cls, **kwargs):
        for thread in cls._threads.values():
            thread.join(TIMEOUT_THREAD)
            writer.debug(f'Thread [{thread.getName()}] is finished')
        best_params, best_result = cls.return_best_model()
        writer.info(f'The best score is {best_result:.2f} with these params: {best_params}')

    @classmethod
    def return_best_model(cls) -> Tuple[int, dict]:
        """
        :return: Tuple of best result and which set of parameters
        """
        return sorted(cls.results.values(), key=lambda res: res.score, reverse=False)[0]


class ThreadManagerGUI(ThreadManager):
    sorted_results = []

    @classmethod
    def wait_for_all_threads(cls, **kwargs):
        """
        A manual thread.join(), needed for components that will need to be triggered when a thread ends his task
        @param kwargs: components needed to react to the thread's training process
        """
        check_checkbox = kwargs['check_checkbox']
        lst_finished = []
        num_of_threads = len(super()._threads)
        while len(lst_finished) < num_of_threads:
            for index in range(num_of_threads):
                if cls.is_finished_by_index(index) and index not in lst_finished:
                    check_checkbox.emit(index)
                    lst_finished.append(index)
        cls.sorted_results = sorted(cls.results.values(), key=lambda res: res.score, reverse=False)
        best_params, best_result = cls.return_best_model()
        writer.info(f'The best score is {best_result:.2f} with these params: {best_params}')

    @classmethod
    def is_finished_by_index(cls, index: int):
        """
        :param index: number of index
        :return: boolean, if thread is finished
        """
        if not cls._threads[index].is_alive():
            writer.debug(f'Thread [{cls._threads[index].getName()}] is running')
            return True
        return False
