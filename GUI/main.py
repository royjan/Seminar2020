#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial

This program creates a submenu.

Author: Jan Bodnar
Website: zetcode.com
Last edited: August 2017
"""

import sys

from PyQt5 import QtCore
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication, QPushButton, QGridLayout, QCheckBox, QLabel, \
    QListView
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor


def hello_wold():
    print("asd")


class PrimaryWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.lst_boxes = {}
        self.init_menu_bar()
        self.init_window()
        self.show_models()
        self.show()

    def init_menu_bar(self):
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        about = QAction('About', self)
        bye = QAction('Exit', self)
        fileMenu.addAction(about)
        fileMenu.addAction(bye)
        self.setGeometry(400, 400, 1000, 800)
        self.setWindowTitle('Seminar2020 - Roy Ronen Yuval Ricky')

    def init_window(self):
        buttonWindow1 = QPushButton('Run Models', self)
        buttonWindow1.setGeometry(250, 100, 400, 30)
        buttonWindow1.clicked.connect(hello_wold)

    def show_models(self):
        params = [{'model': SVC, "C": 0.4}, {'model': LinearRegression, 'normalize': False, "fit_intercept": False},
                  {'model': LinearRegression, 'normalize': True, "fit_intercept": False}, {'model': SVC, "degree": 4},
                  {'model': DecisionTreeRegressor}]
        for offset, param in enumerate(params, start=1):
            box = QCheckBox(self.get_model_name_by_clf(param['model']), self)
            self.lst_boxes[offset] = box
            box.resize(500, 40)
            box.move(200, 200 + offset * 50)
            box.setEnabled(False)
            # box.toggle()

    def toggle_check_box(self):
        threads_list = ThreadManager.return_copy_of_threads_list()
        while threads_list:
            for index in range(len(threads_list)):
                if ThreadManager.is_thread_alive(index):
                    self.lst_boxes[index].toggle()
                    threads_list.pop(index)


    @staticmethod
    def get_model_name_by_clf(clf_name):
        return str(clf_name).split(".")[-1].split("'")[0]

    @staticmethod
    def prevent_toggle(check_box: QCheckBox):
        check_box.setChecked(QtCore.Qt.Checked)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PrimaryWindow()
    sys.exit(app.exec_())
