from contextlib import contextmanager
from threading import Thread, Lock
from typing import Tuple

import numpy as np

from Algorithm.Model import learning_model
from Utils.Log import writer


@contextmanager
def lock_action(lock: Lock):
    """
    :param lock: Lock threading object to acquire and release after an action
    """
    lock.acquire()
    yield
    lock.release()


class ThreadManager:
    list_of_threads = []
    list_of_results = []

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
        lock = Lock()
        for param in params:
            thread = Thread(name=str(param), target=learning_model, args=(X_train, y_train, X_test, y_test),
                            kwargs=param)
            param['result'] = cls.list_of_results
            writer.debug(f'Thread [{thread.getName()}] is running')
            with lock_action(lock):
                thread.start()
                cls.list_of_threads.append(thread)
        return cls.list_of_threads

    @classmethod
    def wait_for_all_threads(cls):
        """
        :param threads: join for every thread in the list
        """
        threads: [Thread]
        for thread in cls.list_of_threads:
            thread.join()
            writer.debug(f'Thread [{thread.getName()}] is finished')

    @classmethod
    def return_best_score(cls, params: list) -> Tuple[int, dict]:
        """
        :param params: list of parameter sets
        :param results: list of scores
        :return: Tuple of best result and which set of parameters
        """
        best_result = np.inf
        best_params = params[0]
        for param, result in zip(params, cls.list_of_results):
            param.pop('result')
            if result < best_result:
                best_result = result
                best_params = param
        return best_result, best_params
