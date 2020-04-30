##############################
# Yuval Barnahor - 308465350
# Roy Jan - 204271209
# Ricky Danipog - 327072310
# Ronen Rozen - 203024542
##############################

import sys

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import QMovie, QPainter
from PyQt5.QtWidgets import QMainWindow, QAction, QApplication, QPushButton, QCheckBox, QLabel, QLineEdit

from Algorithm import Model
from Algorithm.ThreadManager import ThreadManagerGUI
from Utils.FileUtils import FileUtils
from Utils.Log import logger, writer
from Utils.Preprocess import Preprocess

logger.set_logger_severity('info')
headers = ['Id', 'OverallQual', 'YearBuilt', 'OverallCond', 'OpenPorchSF']
df1 = FileUtils.read_data_frame_from_path('Data/train_1.csv', headers)
df2 = FileUtils.read_data_frame_from_path('Data/train_2.xlsx', headers)
df3 = FileUtils.read_data_frame_from_path('Data/train_targets.csv')
df_total = df1.append(df2).merge(df3, on='Id', how='left')
pp = Preprocess(df_total, 'SalePrice')
pp.replace_nan()
X_train, X_test, y_train, y_test = pp.split_train_test_by_pandas()
params = FileUtils.read_models_from_text()


class Worker(QtCore.QObject):
    """
    Enable simultaneity work with GUI and other actions
    """
    reset_check_boxes = QtCore.pyqtSignal()
    check_checkbox = QtCore.pyqtSignal(int)
    show_graph = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super(Worker, self).__init__(parent)
        self.window = parent

    @QtCore.pyqtSlot()
    def start_worker(self):
        """
        Worker starts in a separated thread, doing its work and shows the results
        """
        global best_model
        try:
            self.reset_check_boxes.emit()
            ThreadManagerGUI.running_threads_args(X_train, y_train, X_test, y_test, params)
            ThreadManagerGUI.wait_for_all_threads(check_checkbox=self.check_checkbox)
            best_model, _ = ThreadManagerGUI.return_best_model()
            self.show_graph.emit()
        except Exception as e:  # if learning is fail for some-reason..
            writer.error(f"Learning Failed: {repr(e)}")

    @QtCore.pyqtSlot()
    def _predict_worker(self):
        """
        predict with GUI worker
        """
        self._predict.emit()

class PrimaryWindow(QMainWindow):
    """
    Primary Window - the main menu window
    """

    class AboutWindow(QMainWindow):
        """
        About window - names
        """

        def __init__(self):
            super().__init__()
            self.setGeometry(1000, 800, 1000, 800)
            self.setWindowTitle('About')
            self.movie = QMovie("giphy.gif")
            self.movie.frameChanged.connect(self.repaint)
            self.movie.start()
            self.init_text()

        def paintEvent(self, event):
            """
            This function repaint the event over and over (animation)
            """
            currentFrame = self.movie.currentPixmap()
            frameRect = currentFrame.rect()
            frameRect.moveCenter(self.rect().center())
            if frameRect.intersects(event.rect()):
                painter = QPainter(self)
                painter.drawPixmap(frameRect.left(), frameRect.top(), currentFrame)

        def init_text(self):
            """
            text init
            """
            text_to_show = 'Names: Roy Jan, Ronen Rozen, Yuval Bar-Nahor, Ricky Danipog\nLaProfessor: Itzhak Aviv'
            label = QLabel(text_to_show, self)
            label.setGeometry(50, 50, 900, 100)


    def __init__(self):
        super().__init__()
        self.name_label = QLabel(self)
        self.run_model_explain = QLabel(self)
        self.line = QLineEdit(self)
        self.result_clf = QLabel(self)
        self.about_window = self.AboutWindow()
        self.worker = Worker()
        self.checkboxes = {}
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
        self.about_window.show()

    def exit_button(self):
        """
        closes project
        """
        app = QApplication.instance()
        app.closeAllWindows()
        self.close()

    def init_window(self):
        """
        button init
        """
        buttonWindow1 = QPushButton('Run Models', self)
        buttonWindow1.setGeometry(250, 100, 400, 30)
        self.init_worker()
        buttonWindow1.clicked.connect(self.worker.start_worker)
        self.run_model_explain.setText('Click "Run Models" to start learning!')
        self.run_model_explain.setGeometry(250, 150, 500, 30)

    def init_worker(self):
        """
        setup functions to worker and starts the main thread
        """
        thread = QtCore.QThread(self)
        thread.start()
        self.worker.reset_check_boxes.connect(self.reset_check_boxes)
        self.worker.check_checkbox.connect(self.check_checkbox)
        self.worker.show_graph.connect(self.show_graph_after_training)
        self.worker.moveToThread(thread)

    def show_models(self):
        """
        show checkbox for each model
        """
        for index, param in enumerate(params):
            model_clf = Model.get_model_name_by_clf(param['model'])
            model_params = f"\t({','.join(f'{k}:{v}' for k, v in param.items() if k != 'model')})"  # param1: value1, ..
            box = QCheckBox(f"{model_clf} {model_params}", self)
            self.checkboxes[index] = box
            box.resize(700, 40)
            box.move(200, 200 + (index + 1) * 50)
            box.setEnabled(False)

    @QtCore.pyqtSlot()
    def reset_check_boxes(self):
        """
        clear every checkbox
        """
        if self.checkboxes:
            for box in self.checkboxes.values():
                box.setChecked(False)

    @QtCore.pyqtSlot(int)
    def check_checkbox(self, index):
        """
        :param index: the index in dict boxes
        checkbox set to True and update GUI
        """
        self.checkboxes[index].setChecked(True)
        self.update()

    @staticmethod
    def update():
        """
        updating the GUI
        """
        QtGui.QGuiApplication.processEvents()

    @staticmethod
    def show_graph_after_training():
        """
        open new window with graph
        """
        from Utils.GraphUtils import GraphUtils
        GraphUtils.create_graph(ThreadManagerGUI.sorted_results)
        # ThreadManagerGUI.reset_values()  # reset results for the next running

    def create_input_subwindow(self):
        """
        shows new inputs to predict with the best model
        """
        self.name_label.setText('SQBT:')
        self.line.setGeometry(300, 600, 200, 32)
        self.name_label.move(200, 600)
        buttonWindow2 = QPushButton('Predict', self)
        buttonWindow2.setGeometry(250, 700, 400, 30)
        self.result_clf.setText("")
        self.result_clf.setGeometry(200, 520, 400, 30)
        buttonWindow2.clicked.connect(self.worker._predict_worker)

    def show_err_msg(self, err_msg: str, user_input: str):
        """
        :param user_input: user input
        :param err_msg: pop-up an error message with err_msg text
        """
        writer.error(f'Error: {err_msg} | Input: {user_input}')
        emsg = QtWidgets.QErrorMessage(self)
        emsg.setWindowModality(QtCore.Qt.WindowModal)
        emsg.showMessage(err_msg)

    @QtCore.pyqtSlot()
    def _predict(self):
        """
        receive an input from user and use it to prediction
        """
        user_input = self.line.text()
        if not best_model:
            self.show_err_msg('You have to run train first!', user_input)
            return
        if not user_input:
            self.show_err_msg("Input can't be empty!", user_input)
            return
        # clf_result = best_model.predict([[123, 24]]) #TODO: let's talk about how we create a predict row
        clf_result = 124124124124
        writer.info(f"Prediction for {user_input} => {clf_result}")
        self.result_clf.setText(f"Prediction: {clf_result}")
        self.result_clf.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PrimaryWindow()
    sys.exit(app.exec_())
