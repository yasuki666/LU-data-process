# -*- coding: utf-8 -*-

#pyuic5 -o original_UI.py LU信号处理.ui
#pyuic5 -o new_ui.py new_ui.ui

import sys

from PyQt5.QtWidgets import QApplication , QMainWindow
from PyQt5 import QtCore
import Myui
import new_ui
import function_UI
import original_UI as pro_ui

if __name__ == '__main__':
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = function_UI.Ui_MainWindow()
    ui.new_set_UI(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
