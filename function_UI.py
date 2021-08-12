# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LU信号处理.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QAction,QMainWindow,QFileDialog,QTextEdit,QMessageBox
from PyQt5.QtWidgets import QApplication

import pyqtgraph as pg

import pandas as pd
import numpy as np

import original_UI

class Ui_MainWindow(original_UI.Ui_MainWindow):

    def new_set_UI(self,mainWindow):
        self.setupUi(mainWindow)

        #菜单栏 File
        #open new file
        self.actionopen_new_file.triggered.connect(self.open_file)

        #控制区模块 初始化
        self.column_number_value.setText('1') #默认列数为1
        self.sampling_rate_value.setText('2500')  #默认采样率为2500MHz
        self.start_sampling_time_value.setText('1') #默认开始采样时间为1us
        self.end_sampling_time_value.setText('5') #默认结束采样时间为5us

        # 按钮 导入数据
        self.import_data_button.clicked.connect(self.import_data)

        # 按钮 数据列数改变 ← →
        self.last_column_button.clicked.connect(self.column_number_minus)
        self.next_column_button.clicked.connect(self.column_number_plus)

        # 必须先将四个画布建立好
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.original_time_canvas = pg.PlotWidget(self.Original_time)  # 创建一个绘图控件
        self.original_time_canvas.resize(300,160)
        self.original_time_canvas_plot = self.original_time_canvas.plot()
        self.original_frequency_canvas = pg.PlotWidget(self.Original_frequency)
        self.original_frequency_canvas.resize(300, 160)
        self.original_frequency_canvas_plot = self.original_frequency_canvas.plot()
        self.processed_time_canvas = pg.PlotWidget(self.processed_time)
        self.processed_time_canvas.resize(300,160)
        self.processed_time_canvas_plot = self.processed_time_canvas.plot()
        self.processed_frequency_canvas = pg.PlotWidget(self.processed_frequency)
        self.processed_frequency_canvas.resize(300,160)
        self.processed_frequency_canvas_plot = self.processed_frequency_canvas.plot()

        # 按钮 作图
        self.drawing_button.clicked.connect(self.original_data_time_frequency_drawing)






    #函数 打开文件对话窗 将路径保存至文件路径栏
    def open_file(self):
        #注意，这里参数必须传入self.centralwidget，因为QFileDialog只接受widget参数
        file_name = QFileDialog.getOpenFileName(self.centralwidget,'选择文件','','Excel files(*.xlsx , *.xls , *.csv)')
        self.file_path_value.setText(file_name[0])

    #函数 按文件路径栏已储存的路径将文件数据导入
    def import_data(self):
        try:
            self.file_data = pd.read_excel(self.file_path_value.text())
            self.file_data = np.array(self.file_data)
            QMessageBox.information(self.centralwidget, '提示', '数据导入成功')
        except:
            QMessageBox.information(self.centralwidget, '提示', '文件路径有误')

    #函数 数据列数减1
    def column_number_minus(self):
        num = int(self.column_number_value.text())
        # 判断是否越栈
        if num > 1:
            num -= 1
        self.column_number_value.setText(str(num))

    #函数 数据列数加1
    def column_number_plus(self):
        num = int(self.column_number_value.text())
        self.column_total_number = self.file_data.shape[1]
        num += 1
        #判断是否越栈
        if num >= self.column_total_number:
            num -= 1
            QMessageBox.information(self.centralwidget, '提示', '已到达最后一列')
        self.column_number_value.setText(str(num))

    #函数 画图 将原始数据时域图与频域图画出
    def original_data_time_frequency_drawing(self):
        try:
            # 原始数据时域图
            self.column_num = int(self.column_number_value.text())
            self.time_aix = self.file_data[:,0]
            self.time_aix = self.time_aix * 1000000
            self.original_time_amplitude_aix = self.file_data[:, self.column_num]
            try:
                i,j = 0,0
                while(self.time_aix[i]<int(self.start_sampling_time_value.text())):
                    i += 1
                while(self.time_aix[j]<int(self.end_sampling_time_value.text())):
                    j += 1
                self.time_aix = self.time_aix[i:j]

                self.original_time_amplitude_aix = self.original_time_amplitude_aix[i:j]
                self.original_time_canvas_plot.setData(self.time_aix, self.original_time_amplitude_aix,pen='b')  # pen参数改变线条颜色，symbol改变点形状，symbolColor改变点颜色
            except:
                QMessageBox.information(self.centralwidget, '提示', '采样时间设置有误')

            # 原始数据频域图
            temprory_time_amplitude_aix = self.original_time_amplitude_aix - np.mean(self.original_time_amplitude_aix) #消去直流分量，更能体现频谱信息
            self.original_frequency_amplitude_aix = np.fft.fft(temprory_time_amplitude_aix) # 快速傅里叶变换
            self.original_frequency_amplitude_aix = abs(self.original_frequency_amplitude_aix)
            n = len(self.original_time_amplitude_aix)
            self.frequency_aix = []
            self.sampling_rate = int(self.sampling_rate_value.text())
            for i in range(n):
                self.frequency_aix.append(i * self.sampling_rate/n)
            self.frequency_aix = self.frequency_aix[:int(n/2)]
            self.original_frequency_amplitude_aix = self.original_frequency_amplitude_aix[:int(n/2)]
            #为了方便观看只展示前边0-100MHz部分
            self.showing_frequency_aix = self.frequency_aix[:int(n*100/(self.sampling_rate))]
            self.showing_original_frequency_amplitude_aix = self.original_frequency_amplitude_aix[:int(n*100/(self.sampling_rate))]
            self.original_frequency_canvas_plot.setData(self.showing_frequency_aix, self.showing_original_frequency_amplitude_aix,pen = 'b')


        except:
            QMessageBox.information(self.centralwidget, '提示', '未导入数据或数据有误')

