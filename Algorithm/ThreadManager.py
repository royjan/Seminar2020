##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################

from contextlib import contextmanager
from threading import Thread, Lock
from typing import Tuple
from collections import namedtuple

from Algorithm.Model import learning_model
from Utils.Log import writer

TIMEOUT_THREAD = 10  # seconds


@contextmanager
def lock_action(lock: Lock):
    """
    :param lock: Lock threading object to acquire and release after an action
    """
    lock.acquire()
    yield
    lock.release()


class ThreadManager:
    _threads = {}
    result_struct = namedtuple('Results', ['params', 'score'])
    results: [result_struct] = []

    @classmethod
    def reset_values(cls):
        cls._threads = {}
        cls.results = []

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
        for index, param in enumerate(params):
            thread = Thread(name=str(param), target=learning_model, args=(X_train, y_train, X_test, y_test),
                            kwargs=param)
            param['result'] = cls.results
            writer.debug(f'Thread [{thread.getName()}] is running')
            with lock_action(lock):
                thread.start()
                cls._threads[index] = thread
        return cls._threads

    @classmethod
    def wait_for_all_threads(cls):
        for thread in cls._threads.values():
            thread.join(TIMEOUT_THREAD)
            writer.debug(f'Thread [{thread.getName()}] is finished')

    @classmethod
    def return_best_model(cls) -> Tuple[int, dict]:
        """
        :return: Tuple of best result and which set of parameters
        """
        return sorted(cls.results, key=lambda item: item.score, reverse=False)[0]

    @classmethod
    def is_finished_by_index(cls, index: int):
        """
        :param index: number of index
        :return: boolean, if thread is finished
        """
        return not cls._threads[index].is_alive()

