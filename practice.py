# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
from PyQt5.QtWidgets import QApplication, QWidget,QPushButton,QToolTip,QDesktopWidget
from PyQt5.QtWidgets import QHBoxLayout,QVBoxLayout
from PyQt5.QtWidgets import QAction,QMainWindow,QFileDialog,QTextEdit
from PyQt5.QtGui import QFont,QIcon


class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        #按钮提示字体
        QToolTip.setFont(QFont('Sanserif',10))

        #创建提示
        self.setToolTip("This is a tip")

        #创建按钮

        btn1 = QPushButton('Button',self)
        btn1.setToolTip("This is a button")
        btn1.resize(btn1.sizeHint())
        btn1.move(50,50)

        '''
        Button_ok = QPushButton("ok")
        Button_cancel = QPushButton("cancel")
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(Button_ok)
        hbox.addWidget(Button_cancel)
        vbox = QVBoxLayout()
        vbox.addStretch(2)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        '''
        #创建窗口
        self.resize(1200,800)
        self.center()
        self.setWindowTitle("LU信号处理")
        self.top_menu()
        self.show()

    #获取屏幕中心位置并且将窗口置于中心
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    #顶部菜单栏
    def top_menu(self):

        #File栏
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        #Open new File
        open_file = QAction(QIcon('open.png'),'open',self)
        open_file.setShortcut('Ctrl+O')
        open_file.setStatusTip('Open new File')
        open_file.triggered.connect(self.showDialog)

        #顶部菜单栏设置
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('File')
        file_menu.addAction(open_file)

    # 打开文件目录对话框
    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.read()
                self.textEdit.setText(data)





def open_app():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    open_app()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
