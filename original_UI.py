# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LU信号处理.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(891, 615)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.drawing_box = QtWidgets.QGroupBox(self.centralwidget)
        self.drawing_box.setGeometry(QtCore.QRect(220, 70, 661, 371))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.drawing_box.setFont(font)
        self.drawing_box.setObjectName("drawing_box")
        self.Original_time = QtWidgets.QGraphicsView(self.drawing_box)
        self.Original_time.setGeometry(QtCore.QRect(20, 20, 301, 161))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.Original_time.setFont(font)
        self.Original_time.setObjectName("Original_time")
        self.Original_frequency = QtWidgets.QGraphicsView(self.drawing_box)
        self.Original_frequency.setGeometry(QtCore.QRect(340, 20, 301, 161))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.Original_frequency.setFont(font)
        self.Original_frequency.setObjectName("Original_frequency")
        self.processed_time = QtWidgets.QGraphicsView(self.drawing_box)
        self.processed_time.setGeometry(QtCore.QRect(20, 200, 301, 161))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.processed_time.setFont(font)
        self.processed_time.setObjectName("processed_time")
        self.processed_frequency = QtWidgets.QGraphicsView(self.drawing_box)
        self.processed_frequency.setGeometry(QtCore.QRect(340, 200, 301, 161))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.processed_frequency.setFont(font)
        self.processed_frequency.setObjectName("processed_frequency")
        self.bandpass_filter_box = QtWidgets.QGroupBox(self.centralwidget)
        self.bandpass_filter_box.setGeometry(QtCore.QRect(10, 190, 201, 171))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.bandpass_filter_box.setFont(font)
        self.bandpass_filter_box.setObjectName("bandpass_filter_box")
        self.passbond_lowerlimit_frequency_value = QtWidgets.QLineEdit(self.bandpass_filter_box)
        self.passbond_lowerlimit_frequency_value.setGeometry(QtCore.QRect(100, 20, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.passbond_lowerlimit_frequency_value.setFont(font)
        self.passbond_lowerlimit_frequency_value.setText("")
        self.passbond_lowerlimit_frequency_value.setObjectName("passbond_lowerlimit_frequency_value")
        self.passbond_upperlimit_frequency_value = QtWidgets.QLineEdit(self.bandpass_filter_box)
        self.passbond_upperlimit_frequency_value.setGeometry(QtCore.QRect(100, 50, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.passbond_upperlimit_frequency_value.setFont(font)
        self.passbond_upperlimit_frequency_value.setText("")
        self.passbond_upperlimit_frequency_value.setObjectName("passbond_upperlimit_frequency_value")
        self.stopbond_lowerlimit_frequency_value = QtWidgets.QLineEdit(self.bandpass_filter_box)
        self.stopbond_lowerlimit_frequency_value.setGeometry(QtCore.QRect(100, 80, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.stopbond_lowerlimit_frequency_value.setFont(font)
        self.stopbond_lowerlimit_frequency_value.setText("")
        self.stopbond_lowerlimit_frequency_value.setObjectName("stopbond_lowerlimit_frequency_value")
        self.stopbond_upperlimit_frequency_value = QtWidgets.QLineEdit(self.bandpass_filter_box)
        self.stopbond_upperlimit_frequency_value.setGeometry(QtCore.QRect(100, 110, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.stopbond_upperlimit_frequency_value.setFont(font)
        self.stopbond_upperlimit_frequency_value.setText("")
        self.stopbond_upperlimit_frequency_value.setObjectName("stopbond_upperlimit_frequency_value")
        self.bandpass_filter_button = QtWidgets.QPushButton(self.bandpass_filter_box)
        self.bandpass_filter_button.setGeometry(QtCore.QRect(40, 140, 111, 23))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.bandpass_filter_button.setFont(font)
        self.bandpass_filter_button.setObjectName("bandpass_filter_button")
        self.passbond_lowerlimit_frequency_text = QtWidgets.QLabel(self.bandpass_filter_box)
        self.passbond_lowerlimit_frequency_text.setGeometry(QtCore.QRect(20, 20, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.passbond_lowerlimit_frequency_text.setFont(font)
        self.passbond_lowerlimit_frequency_text.setObjectName("passbond_lowerlimit_frequency_text")
        self.passbond_upperlimit_frequency_text = QtWidgets.QLabel(self.bandpass_filter_box)
        self.passbond_upperlimit_frequency_text.setGeometry(QtCore.QRect(20, 50, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.passbond_upperlimit_frequency_text.setFont(font)
        self.passbond_upperlimit_frequency_text.setObjectName("passbond_upperlimit_frequency_text")
        self.stopbond_lowerlimit_frequency_text = QtWidgets.QLabel(self.bandpass_filter_box)
        self.stopbond_lowerlimit_frequency_text.setGeometry(QtCore.QRect(20, 80, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.stopbond_lowerlimit_frequency_text.setFont(font)
        self.stopbond_lowerlimit_frequency_text.setObjectName("stopbond_lowerlimit_frequency_text")
        self.stopbond_upperlimit_frequency_text = QtWidgets.QLabel(self.bandpass_filter_box)
        self.stopbond_upperlimit_frequency_text.setGeometry(QtCore.QRect(20, 110, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.stopbond_upperlimit_frequency_text.setFont(font)
        self.stopbond_upperlimit_frequency_text.setObjectName("stopbond_upperlimit_frequency_text")
        self.import_data_button = QtWidgets.QPushButton(self.centralwidget)
        self.import_data_button.setGeometry(QtCore.QRect(30, 10, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.import_data_button.setFont(font)
        self.import_data_button.setObjectName("import_data_button")
        self.wavelet_filter_box = QtWidgets.QGroupBox(self.centralwidget)
        self.wavelet_filter_box.setGeometry(QtCore.QRect(10, 370, 201, 201))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.wavelet_filter_box.setFont(font)
        self.wavelet_filter_box.setObjectName("wavelet_filter_box")
        self.threshold_select_criteria_value = QtWidgets.QComboBox(self.wavelet_filter_box)
        self.threshold_select_criteria_value.setGeometry(QtCore.QRect(100, 20, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.threshold_select_criteria_value.setFont(font)
        self.threshold_select_criteria_value.setObjectName("threshold_select_criteria_value")
        self.threshold_select_criteria_value.addItem("")
        self.threshold_select_criteria_value.addItem("")
        self.threshold_select_criteria_value.addItem("")
        self.threshold_function_select_method_value = QtWidgets.QComboBox(self.wavelet_filter_box)
        self.threshold_function_select_method_value.setGeometry(QtCore.QRect(120, 50, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.threshold_function_select_method_value.setFont(font)
        self.threshold_function_select_method_value.setObjectName("threshold_function_select_method_value")
        self.threshold_process_method_value = QtWidgets.QComboBox(self.wavelet_filter_box)
        self.threshold_process_method_value.setGeometry(QtCore.QRect(100, 80, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.threshold_process_method_value.setFont(font)
        self.threshold_process_method_value.setObjectName("threshold_process_method_value")
        self.decomposition_layer_value = QtWidgets.QLineEdit(self.wavelet_filter_box)
        self.decomposition_layer_value.setGeometry(QtCore.QRect(100, 110, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.decomposition_layer_value.setFont(font)
        self.decomposition_layer_value.setText("")
        self.decomposition_layer_value.setObjectName("decomposition_layer_value")
        self.wavelet_layer_number_value = QtWidgets.QLineEdit(self.wavelet_filter_box)
        self.wavelet_layer_number_value.setGeometry(QtCore.QRect(100, 140, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.wavelet_layer_number_value.setFont(font)
        self.wavelet_layer_number_value.setText("")
        self.wavelet_layer_number_value.setObjectName("wavelet_layer_number_value")
        self.wavelet_filter_button = QtWidgets.QPushButton(self.wavelet_filter_box)
        self.wavelet_filter_button.setGeometry(QtCore.QRect(40, 170, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.wavelet_filter_button.setFont(font)
        self.wavelet_filter_button.setObjectName("wavelet_filter_button")
        self.threshold_select_criteria_text = QtWidgets.QLabel(self.wavelet_filter_box)
        self.threshold_select_criteria_text.setGeometry(QtCore.QRect(20, 20, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.threshold_select_criteria_text.setFont(font)
        self.threshold_select_criteria_text.setObjectName("threshold_select_criteria_text")
        self.threshold_function_select_method_text = QtWidgets.QLabel(self.wavelet_filter_box)
        self.threshold_function_select_method_text.setGeometry(QtCore.QRect(20, 50, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.threshold_function_select_method_text.setFont(font)
        self.threshold_function_select_method_text.setObjectName("threshold_function_select_method_text")
        self.threshold_process_method_text_3 = QtWidgets.QLabel(self.wavelet_filter_box)
        self.threshold_process_method_text_3.setGeometry(QtCore.QRect(20, 80, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.threshold_process_method_text_3.setFont(font)
        self.threshold_process_method_text_3.setObjectName("threshold_process_method_text_3")
        self.decomposition_layer_text = QtWidgets.QLabel(self.wavelet_filter_box)
        self.decomposition_layer_text.setGeometry(QtCore.QRect(20, 110, 51, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.decomposition_layer_text.setFont(font)
        self.decomposition_layer_text.setObjectName("decomposition_layer_text")
        self.wavelet_layer_number_text = QtWidgets.QLabel(self.wavelet_filter_box)
        self.wavelet_layer_number_text.setGeometry(QtCore.QRect(20, 140, 51, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.wavelet_layer_number_text.setFont(font)
        self.wavelet_layer_number_text.setObjectName("wavelet_layer_number_text")
        self.drawing_button = QtWidgets.QPushButton(self.centralwidget)
        self.drawing_button.setGeometry(QtCore.QRect(30, 40, 75, 23))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.drawing_button.setFont(font)
        self.drawing_button.setObjectName("drawing_button")
        self.lowpass_filter_box = QtWidgets.QGroupBox(self.centralwidget)
        self.lowpass_filter_box.setGeometry(QtCore.QRect(10, 70, 201, 111))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.lowpass_filter_box.setFont(font)
        self.lowpass_filter_box.setObjectName("lowpass_filter_box")
        self.stopband_boundary_frequency_value = QtWidgets.QLineEdit(self.lowpass_filter_box)
        self.stopband_boundary_frequency_value.setGeometry(QtCore.QRect(100, 50, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.stopband_boundary_frequency_value.setFont(font)
        self.stopband_boundary_frequency_value.setText("")
        self.stopband_boundary_frequency_value.setObjectName("stopband_boundary_frequency_value")
        self.passbond_boundary_frequency_value = QtWidgets.QLineEdit(self.lowpass_filter_box)
        self.passbond_boundary_frequency_value.setGeometry(QtCore.QRect(100, 20, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.passbond_boundary_frequency_value.setFont(font)
        self.passbond_boundary_frequency_value.setText("")
        self.passbond_boundary_frequency_value.setObjectName("passbond_boundary_frequency_value")
        self.lowpass_filter_button = QtWidgets.QPushButton(self.lowpass_filter_box)
        self.lowpass_filter_button.setGeometry(QtCore.QRect(40, 80, 111, 23))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.lowpass_filter_button.setFont(font)
        self.lowpass_filter_button.setObjectName("lowpass_filter_button")
        self.passbond_boundary_frequency_text = QtWidgets.QLabel(self.lowpass_filter_box)
        self.passbond_boundary_frequency_text.setGeometry(QtCore.QRect(20, 20, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.passbond_boundary_frequency_text.setFont(font)
        self.passbond_boundary_frequency_text.setObjectName("passbond_boundary_frequency_text")
        self.stopband_boundary_frequency_text = QtWidgets.QLabel(self.lowpass_filter_box)
        self.stopband_boundary_frequency_text.setGeometry(QtCore.QRect(20, 50, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.stopband_boundary_frequency_text.setFont(font)
        self.stopband_boundary_frequency_text.setObjectName("stopband_boundary_frequency_text")
        self.sampling_rate_value = QtWidgets.QLineEdit(self.centralwidget)
        self.sampling_rate_value.setGeometry(QtCore.QRect(170, 10, 91, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.sampling_rate_value.setFont(font)
        self.sampling_rate_value.setText("")
        self.sampling_rate_value.setObjectName("sampling_rate_value")
        self.column_number_value = QtWidgets.QLineEdit(self.centralwidget)
        self.column_number_value.setGeometry(QtCore.QRect(200, 40, 31, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.column_number_value.setFont(font)
        self.column_number_value.setText("")
        self.column_number_value.setObjectName("column_number_value")
        self.sampling_rate_text = QtWidgets.QLabel(self.centralwidget)
        self.sampling_rate_text.setGeometry(QtCore.QRect(110, 10, 54, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.sampling_rate_text.setFont(font)
        self.sampling_rate_text.setObjectName("sampling_rate_text")
        self.column_number_text = QtWidgets.QLabel(self.centralwidget)
        self.column_number_text.setGeometry(QtCore.QRect(110, 40, 54, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.column_number_text.setFont(font)
        self.column_number_text.setObjectName("column_number_text")
        self.sampling_time_text = QtWidgets.QLabel(self.centralwidget)
        self.sampling_time_text.setGeometry(QtCore.QRect(290, 10, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.sampling_time_text.setFont(font)
        self.sampling_time_text.setObjectName("sampling_time_text")
        self.start_sampling_time_value = QtWidgets.QLineEdit(self.centralwidget)
        self.start_sampling_time_value.setGeometry(QtCore.QRect(300, 30, 21, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.start_sampling_time_value.setFont(font)
        self.start_sampling_time_value.setText("")
        self.start_sampling_time_value.setObjectName("start_sampling_time_value")
        self.start_end_text = QtWidgets.QLabel(self.centralwidget)
        self.start_end_text.setGeometry(QtCore.QRect(330, 30, 16, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.start_end_text.setFont(font)
        self.start_end_text.setObjectName("start_end_text")
        self.end_sampling_time_value = QtWidgets.QLineEdit(self.centralwidget)
        self.end_sampling_time_value.setGeometry(QtCore.QRect(360, 30, 21, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.end_sampling_time_value.setFont(font)
        self.end_sampling_time_value.setText("")
        self.end_sampling_time_value.setObjectName("end_sampling_time_value")
        self.file_path_value = QtWidgets.QLineEdit(self.centralwidget)
        self.file_path_value.setGeometry(QtCore.QRect(410, 30, 401, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.file_path_value.setFont(font)
        self.file_path_value.setText("")
        self.file_path_value.setObjectName("file_path_value")
        self.file_path_text = QtWidgets.QLabel(self.centralwidget)
        self.file_path_text.setGeometry(QtCore.QRect(410, 10, 101, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.file_path_text.setFont(font)
        self.file_path_text.setObjectName("file_path_text")
        self.last_column_button = QtWidgets.QPushButton(self.centralwidget)
        self.last_column_button.setGeometry(QtCore.QRect(170, 40, 21, 20))
        self.last_column_button.setObjectName("last_column_button")
        self.next_column_button = QtWidgets.QPushButton(self.centralwidget)
        self.next_column_button.setGeometry(QtCore.QRect(240, 40, 21, 20))
        self.next_column_button.setObjectName("next_column_button")
        self.data_analyse_box = QtWidgets.QGroupBox(self.centralwidget)
        self.data_analyse_box.setGeometry(QtCore.QRect(220, 450, 661, 121))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.data_analyse_box.setFont(font)
        self.data_analyse_box.setObjectName("data_analyse_box")
        self.original_data_maximum_text = QtWidgets.QLabel(self.data_analyse_box)
        self.original_data_maximum_text.setGeometry(QtCore.QRect(10, 30, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.original_data_maximum_text.setFont(font)
        self.original_data_maximum_text.setObjectName("original_data_maximum_text")
        self.original_data_minimum_text = QtWidgets.QLabel(self.data_analyse_box)
        self.original_data_minimum_text.setGeometry(QtCore.QRect(10, 60, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.original_data_minimum_text.setFont(font)
        self.original_data_minimum_text.setObjectName("original_data_minimum_text")
        self.original_data_maximum_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.original_data_maximum_value.setGeometry(QtCore.QRect(100, 30, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.original_data_maximum_value.setFont(font)
        self.original_data_maximum_value.setText("")
        self.original_data_maximum_value.setObjectName("original_data_maximum_value")
        self.original_data_minimum_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.original_data_minimum_value.setGeometry(QtCore.QRect(100, 60, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.original_data_minimum_value.setFont(font)
        self.original_data_minimum_value.setText("")
        self.original_data_minimum_value.setObjectName("original_data_minimum_value")
        self.original_data_peaktopeak_text = QtWidgets.QLabel(self.data_analyse_box)
        self.original_data_peaktopeak_text.setGeometry(QtCore.QRect(10, 90, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.original_data_peaktopeak_text.setFont(font)
        self.original_data_peaktopeak_text.setObjectName("original_data_peaktopeak_text")
        self.original_data_peaktopeak_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.original_data_peaktopeak_value.setGeometry(QtCore.QRect(100, 90, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.original_data_peaktopeak_value.setFont(font)
        self.original_data_peaktopeak_value.setText("")
        self.original_data_peaktopeak_value.setObjectName("original_data_peaktopeak_value")
        self.processed_data_peaktopeak_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.processed_data_peaktopeak_value.setGeometry(QtCore.QRect(280, 90, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.processed_data_peaktopeak_value.setFont(font)
        self.processed_data_peaktopeak_value.setText("")
        self.processed_data_peaktopeak_value.setObjectName("processed_data_peaktopeak_value")
        self.processed_data_maximum_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.processed_data_maximum_value.setGeometry(QtCore.QRect(280, 30, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.processed_data_maximum_value.setFont(font)
        self.processed_data_maximum_value.setText("")
        self.processed_data_maximum_value.setObjectName("processed_data_maximum_value")
        self.processed_data_maximum_text = QtWidgets.QLabel(self.data_analyse_box)
        self.processed_data_maximum_text.setGeometry(QtCore.QRect(190, 30, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.processed_data_maximum_text.setFont(font)
        self.processed_data_maximum_text.setObjectName("processed_data_maximum_text")
        self.processed_data_minimum_text = QtWidgets.QLabel(self.data_analyse_box)
        self.processed_data_minimum_text.setGeometry(QtCore.QRect(190, 60, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.processed_data_minimum_text.setFont(font)
        self.processed_data_minimum_text.setObjectName("processed_data_minimum_text")
        self.processed_data_peaktopeak_text = QtWidgets.QLabel(self.data_analyse_box)
        self.processed_data_peaktopeak_text.setGeometry(QtCore.QRect(190, 90, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.processed_data_peaktopeak_text.setFont(font)
        self.processed_data_peaktopeak_text.setObjectName("processed_data_peaktopeak_text")
        self.processed_data_minimum_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.processed_data_minimum_value.setGeometry(QtCore.QRect(280, 60, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.processed_data_minimum_value.setFont(font)
        self.processed_data_minimum_value.setText("")
        self.processed_data_minimum_value.setObjectName("processed_data_minimum_value")
        self.spread_distance_text = QtWidgets.QLabel(self.data_analyse_box)
        self.spread_distance_text.setGeometry(QtCore.QRect(360, 90, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.spread_distance_text.setFont(font)
        self.spread_distance_text.setObjectName("spread_distance_text")
        self.headwave_moment_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.headwave_moment_value.setGeometry(QtCore.QRect(430, 30, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.headwave_moment_value.setFont(font)
        self.headwave_moment_value.setText("")
        self.headwave_moment_value.setObjectName("headwave_moment_value")
        self.echo_moment_text = QtWidgets.QLabel(self.data_analyse_box)
        self.echo_moment_text.setGeometry(QtCore.QRect(360, 60, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.echo_moment_text.setFont(font)
        self.echo_moment_text.setObjectName("echo_moment_text")
        self.spread_distance_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.spread_distance_value.setGeometry(QtCore.QRect(440, 90, 61, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.spread_distance_value.setFont(font)
        self.spread_distance_value.setText("")
        self.spread_distance_value.setObjectName("spread_distance_value")
        self.headwave_moment_text = QtWidgets.QLabel(self.data_analyse_box)
        self.headwave_moment_text.setGeometry(QtCore.QRect(360, 30, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.headwave_moment_text.setFont(font)
        self.headwave_moment_text.setObjectName("headwave_moment_text")
        self.echo_moment_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.echo_moment_value.setGeometry(QtCore.QRect(430, 60, 71, 20))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.echo_moment_value.setFont(font)
        self.echo_moment_value.setText("")
        self.echo_moment_value.setObjectName("echo_moment_value")
        self.spread_time_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.spread_time_value.setGeometry(QtCore.QRect(580, 30, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.spread_time_value.setFont(font)
        self.spread_time_value.setText("")
        self.spread_time_value.setObjectName("spread_time_value")
        self.spread_valocity_value = QtWidgets.QLineEdit(self.data_analyse_box)
        self.spread_valocity_value.setGeometry(QtCore.QRect(590, 60, 61, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.spread_valocity_value.setFont(font)
        self.spread_valocity_value.setText("")
        self.spread_valocity_value.setObjectName("spread_valocity_value")
        self.spread_velocity_text = QtWidgets.QLabel(self.data_analyse_box)
        self.spread_velocity_text.setGeometry(QtCore.QRect(510, 60, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.spread_velocity_text.setFont(font)
        self.spread_velocity_text.setObjectName("spread_velocity_text")
        self.spread_time_text = QtWidgets.QLabel(self.data_analyse_box)
        self.spread_time_text.setGeometry(QtCore.QRect(510, 30, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.spread_time_text.setFont(font)
        self.spread_time_text.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.spread_time_text.setObjectName("spread_time_text")
        self.data_processed_caculate_button = QtWidgets.QPushButton(self.data_analyse_box)
        self.data_processed_caculate_button.setGeometry(QtCore.QRect(540, 90, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(9)
        self.data_processed_caculate_button.setFont(font)
        self.data_processed_caculate_button.setObjectName("data_processed_caculate_button")
        self.bandpass_filter_box.raise_()
        self.import_data_button.raise_()
        self.wavelet_filter_box.raise_()
        self.drawing_button.raise_()
        self.lowpass_filter_box.raise_()
        self.sampling_rate_value.raise_()
        self.column_number_value.raise_()
        self.sampling_rate_text.raise_()
        self.column_number_text.raise_()
        self.sampling_time_text.raise_()
        self.start_sampling_time_value.raise_()
        self.start_end_text.raise_()
        self.end_sampling_time_value.raise_()
        self.file_path_value.raise_()
        self.file_path_text.raise_()
        self.drawing_box.raise_()
        self.last_column_button.raise_()
        self.next_column_button.raise_()
        self.data_analyse_box.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 891, 22))
        self.menubar.setObjectName("menubar")
        self.menufile = QtWidgets.QMenu(self.menubar)
        self.menufile.setObjectName("menufile")
        self.menuprocess = QtWidgets.QMenu(self.menubar)
        self.menuprocess.setObjectName("menuprocess")
        self.menuanalyse = QtWidgets.QMenu(self.menubar)
        self.menuanalyse.setObjectName("menuanalyse")
        self.menusave = QtWidgets.QMenu(self.menubar)
        self.menusave.setObjectName("menusave")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action1 = QtWidgets.QAction(MainWindow)
        self.action1.setObjectName("action1")
        self.actionsave_single_column = QtWidgets.QAction(MainWindow)
        self.actionsave_single_column.setObjectName("actionsave_single_column")
        self.actionsave_all_column = QtWidgets.QAction(MainWindow)
        self.actionsave_all_column.setObjectName("actionsave_all_column")
        self.actionsave_reasonable_columns = QtWidgets.QAction(MainWindow)
        self.actionsave_reasonable_columns.setObjectName("actionsave_reasonable_columns")
        self.actionopen_new_file = QtWidgets.QAction(MainWindow)
        self.actionopen_new_file.setObjectName("actionopen_new_file")
        self.actionnew = QtWidgets.QAction(MainWindow)
        self.actionnew.setObjectName("actionnew")
        self.actionnew_2 = QtWidgets.QAction(MainWindow)
        self.actionnew_2.setObjectName("actionnew_2")
        self.actionexit = QtWidgets.QAction(MainWindow)
        self.actionexit.setObjectName("actionexit")
        self.actionlowpass_filter = QtWidgets.QAction(MainWindow)
        self.actionlowpass_filter.setObjectName("actionlowpass_filter")
        self.actionbandpass_filter = QtWidgets.QAction(MainWindow)
        self.actionbandpass_filter.setObjectName("actionbandpass_filter")
        self.actionwavelet_filter = QtWidgets.QAction(MainWindow)
        self.actionwavelet_filter.setObjectName("actionwavelet_filter")
        self.actioncaculate_spread_time_and_velocity = QtWidgets.QAction(MainWindow)
        self.actioncaculate_spread_time_and_velocity.setObjectName("actioncaculate_spread_time_and_velocity")
        self.menufile.addAction(self.actionopen_new_file)
        self.menufile.addAction(self.actionnew)
        self.menufile.addAction(self.actionnew_2)
        self.menufile.addAction(self.actionexit)
        self.menuprocess.addAction(self.actionlowpass_filter)
        self.menuprocess.addAction(self.actionbandpass_filter)
        self.menuprocess.addAction(self.actionwavelet_filter)
        self.menuanalyse.addAction(self.action1)
        self.menuanalyse.addAction(self.actioncaculate_spread_time_and_velocity)
        self.menusave.addAction(self.actionsave_single_column)
        self.menusave.addAction(self.actionsave_all_column)
        self.menusave.addAction(self.actionsave_reasonable_columns)
        self.menubar.addAction(self.menufile.menuAction())
        self.menubar.addAction(self.menuprocess.menuAction())
        self.menubar.addAction(self.menuanalyse.menuAction())
        self.menubar.addAction(self.menusave.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.drawing_box.setTitle(_translate("MainWindow", "作图区"))
        self.bandpass_filter_box.setTitle(_translate("MainWindow", "带通滤波"))
        self.bandpass_filter_button.setText(_translate("MainWindow", "带通滤波"))
        self.passbond_lowerlimit_frequency_text.setText(_translate("MainWindow", "通带下限频率"))
        self.passbond_upperlimit_frequency_text.setText(_translate("MainWindow", "通带上限频率"))
        self.stopbond_lowerlimit_frequency_text.setText(_translate("MainWindow", "阻带下限频率"))
        self.stopbond_upperlimit_frequency_text.setText(_translate("MainWindow", "阻带上限频率"))
        self.import_data_button.setText(_translate("MainWindow", "导入数据"))
        self.wavelet_filter_box.setTitle(_translate("MainWindow", "小波滤波"))
        self.threshold_select_criteria_value.setItemText(0, _translate("MainWindow", "heursure"))
        self.threshold_select_criteria_value.setItemText(1, _translate("MainWindow", "新建项目"))
        self.threshold_select_criteria_value.setItemText(2, _translate("MainWindow", "新建项目"))
        self.wavelet_filter_button.setText(_translate("MainWindow", "小波滤波"))
        self.threshold_select_criteria_text.setText(_translate("MainWindow", "阈值选择标准"))
        self.threshold_function_select_method_text.setText(_translate("MainWindow", "阈值函数选择方式"))
        self.threshold_process_method_text_3.setText(_translate("MainWindow", "阈值处理方式"))
        self.decomposition_layer_text.setText(_translate("MainWindow", "分解层数"))
        self.wavelet_layer_number_text.setText(_translate("MainWindow", "小波类型"))
        self.drawing_button.setText(_translate("MainWindow", "作图"))
        self.lowpass_filter_box.setTitle(_translate("MainWindow", "低通滤波"))
        self.lowpass_filter_button.setText(_translate("MainWindow", "低通滤波"))
        self.passbond_boundary_frequency_text.setText(_translate("MainWindow", "通带边界频率"))
        self.stopband_boundary_frequency_text.setText(_translate("MainWindow", "阻带边界频率"))
        self.sampling_rate_text.setText(_translate("MainWindow", "  采样率"))
        self.column_number_text.setText(_translate("MainWindow", " 数据列数"))
        self.sampling_time_text.setText(_translate("MainWindow", "  采样时间（us）"))
        self.start_end_text.setText(_translate("MainWindow", "——"))
        self.file_path_text.setText(_translate("MainWindow", "文件路径"))
        self.last_column_button.setText(_translate("MainWindow", "←"))
        self.next_column_button.setText(_translate("MainWindow", "→"))
        self.data_analyse_box.setTitle(_translate("MainWindow", "数据分析"))
        self.original_data_maximum_text.setText(_translate("MainWindow", "原始数据最大值"))
        self.original_data_minimum_text.setText(_translate("MainWindow", "原始数据最小值"))
        self.original_data_peaktopeak_text.setText(_translate("MainWindow", "原始数据峰峰值"))
        self.processed_data_maximum_text.setText(_translate("MainWindow", "滤波数据最大值"))
        self.processed_data_minimum_text.setText(_translate("MainWindow", "滤波数据最小值"))
        self.processed_data_peaktopeak_text.setText(_translate("MainWindow", "滤波数据峰峰值"))
        self.spread_distance_text.setText(_translate("MainWindow", "传播路程(mm)"))
        self.echo_moment_text.setText(_translate("MainWindow", "回波时刻(us)"))
        self.headwave_moment_text.setText(_translate("MainWindow", "头波时刻(us)"))
        self.spread_velocity_text.setText(_translate("MainWindow", "传播速度(m/s)"))
        self.spread_time_text.setText(_translate("MainWindow", "传播时间(us)"))
        self.data_processed_caculate_button.setText(_translate("MainWindow", "计算"))
        self.menufile.setTitle(_translate("MainWindow", "file"))
        self.menuprocess.setTitle(_translate("MainWindow", "process"))
        self.menuanalyse.setTitle(_translate("MainWindow", "analyse"))
        self.menusave.setTitle(_translate("MainWindow", "save"))
        self.action1.setText(_translate("MainWindow", "auto-find wave moment"))
        self.actionsave_single_column.setText(_translate("MainWindow", "save single column"))
        self.actionsave_all_column.setText(_translate("MainWindow", "save all columns"))
        self.actionsave_reasonable_columns.setText(_translate("MainWindow", "save reasonable columns"))
        self.actionopen_new_file.setText(_translate("MainWindow", "open new file"))
        self.actionnew.setText(_translate("MainWindow", "open recent file"))
        self.actionnew_2.setText(_translate("MainWindow", "new"))
        self.actionexit.setText(_translate("MainWindow", "exit"))
        self.actionlowpass_filter.setText(_translate("MainWindow", "lowpass filter"))
        self.actionbandpass_filter.setText(_translate("MainWindow", "bandpass filter"))
        self.actionwavelet_filter.setText(_translate("MainWindow", "wavelet filter"))
        self.actioncaculate_spread_time_and_velocity.setText(_translate("MainWindow", "caculate spread time and velocity"))