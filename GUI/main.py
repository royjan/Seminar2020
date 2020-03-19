##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################
import sys

from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, QPushButton, QCheckBox, QLabel
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from Algorithm.ThreadManager import ThreadManager
from Utils.FileUtils import FileUtils
from Utils.Log import logger
from Utils.Preprocess import Preprocess

logger.set_logger_severity('debug')
headers = ['Id', 'OverallQual', 'YearBuilt', 'OverallCond', 'OpenPorchSF']
df1 = FileUtils.read_data_frame_from_path('../Data/train_1.csv', headers)
df2 = FileUtils.read_data_frame_from_path('../Data/train_2.xlsx', headers)
df3 = FileUtils.read_data_frame_from_path('../Data/train_targets.csv')
df_total = df1.append(df2).merge(df3, on='Id', how='left')
pp = Preprocess(df_total, 'SalePrice')
mean_df = Preprocess.replace_nan(pp)
X_train, X_test, y_train, y_test = pp.split_train_test_by_pandas()
params = [{'model': SVC, "C": 0.4}, {'model': LinearRegression, 'normalize': False, "fit_intercept": False},
          {'model': LinearRegression, 'normalize': True, "fit_intercept": False}, {'model': SVC, "degree": 4},
          {'model': DecisionTreeRegressor}]


class PrimaryWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.dict_boxes = {}
        self.init_menu_bar()
        self.init_window()
        self.show_models()
        self.show()

    def init_menu_bar(self):
        """
        creating the menu bar
        """
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('File')
        about_button = QAction('About', self)
        about_button.triggered.connect(self.about_button)
        exit_button = QAction('Exit', self)
        exit_button.triggered.connect(self.exit_button)
        fileMenu.addAction(about_button)
        fileMenu.addAction(exit_button)
        self.setGeometry(400, 400, 1000, 800)
        self.setWindowTitle('Seminar2020 - Roy Ronen Yuval Ricky')

    def about_button(self):
        """
        shows About window
        """
        self.w = AboutWindow()
        self.w.show()

    def exit_button(self):
        """
        closes project
        """
        self.close()

    def init_window(self):
        """
        button init
        """
        buttonWindow1 = QPushButton('Run Models', self)
        buttonWindow1.setGeometry(250, 100, 400, 30)
        buttonWindow1.clicked.connect(self.toggle_check_box)

    def show_models(self):
        """
        show checkbox for each model
        """
        for offset, param in enumerate(params):
            box = QCheckBox(self.get_model_name_by_clf(param['model']), self)
            self.dict_boxes[offset] = box
            box.resize(500, 40)
            box.move(200, 200 + (offset + 1) * 50)
            box.setEnabled(False)

    def toggle_check_box(self):
        """
        toggle to True if the thread finished
        """
        lst_finished = []
        self.reset_check_boxes()
        result = ThreadManager.running_threads_args(X_train, y_train, X_test, y_test, params)
        num_of_threads = len(result)
        while len(lst_finished) < num_of_threads:
            for index in range(num_of_threads):
                if ThreadManager.is_finished_by_index(index) and index not in lst_finished:
                    self.dict_boxes[index].setChecked(True)
                    self.update()
                    lst_finished.append(index)
        ThreadManager.wait_for_all_threads()
        self.show_graph_after_training()

    def reset_check_boxes(self):
        """
        clear every checkbox
        """
        if self.dict_boxes:
            for box in self.dict_boxes.values():
                box.setChecked(False)

    @staticmethod
    def update():
        """
        This function is updating the GUI
        """
        QtGui.QGuiApplication.processEvents()

    @staticmethod
    def get_model_name_by_clf(param) -> str:
        """
        :param param: param as object
        :return: clear classifier name as string
        """
        return str(param).split(".")[-1].split("'")[0]

    def show_graph_after_training(self):
        from Utils.GraphUtils import GraphUtils
        GraphUtils.create_grpah(ThreadManager)


class AboutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(500, 500, 900, 200)
        self.setWindowTitle('About')
        self.init_text()

    def init_text(self):
        """
        text init
        """
        label = QLabel('Names: Roy Jan, Ronen Rozen, Yuval Bar-Nahor, Ricky Danipog\nLaProffesior: Itzhak', self)
        label.resize(900, 100)
        label.move(50, 50)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PrimaryWindow()
    sys.exit(app.exec_())
