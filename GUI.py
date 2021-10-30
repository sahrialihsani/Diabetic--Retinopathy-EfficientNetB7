
from PyQt5 import QtCore, QtGui, QtWidgets
import source

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(735, 494)
        MainWindow.setAnimated(True)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, -10, 735, 494))
        self.label.setStyleSheet("image: url(C:/Users/Pc/OneDrive/Desktop/DiabeticRetinopathy/Welcome.png);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.inputImg = QtWidgets.QLabel(self.centralwidget)
        self.inputImg.setGeometry(QtCore.QRect(242, 170, 248, 190))
        self.inputImg.setText("")
        self.inputImg.setObjectName("inputImg")
        self.LoadBtn = QtWidgets.QPushButton(self.centralwidget)
        self.LoadBtn.setGeometry(QtCore.QRect(289, 367, 159, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.LoadBtn.setFont(font)
        self.LoadBtn.setStyleSheet("background-color: rgb(255,102,128);")
        self.LoadBtn.setObjectName("LoadBtn")
        self.OutImg = QtWidgets.QLabel(self.centralwidget)
        self.OutImg.setGeometry(QtCore.QRect(468, 170, 248, 190))
        self.OutImg.setText("")
        self.OutImg.setObjectName("OutImg")
        self.LoadBtn_2 = QtWidgets.QPushButton(self.centralwidget)
        self.LoadBtn_2.setGeometry(QtCore.QRect(510, 367, 159, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe Print")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.LoadBtn_2.setFont(font)
        self.LoadBtn_2.setStyleSheet("background-color: rgb(255,102,128);")
        self.LoadBtn_2.setObjectName("LoadBtn_2")

        self.DetTxt = QtWidgets.QTextEdit(self.centralwidget)
        self.DetTxt.setGeometry(QtCore.QRect(375, 410, 190, 30))
        self.DetTxt.setSizeIncrement(QtCore.QSize(0, 0))
        self.DetTxt.setFrameShape(QtWidgets.QFrame.Panel)
        self.DetTxt.setLineWidth(2)
        self.DetTxt.setObjectName("DetTxt")

        self.PropTxt = QtWidgets.QTextEdit(self.centralwidget)
        self.PropTxt.setGeometry(QtCore.QRect(42, 260, 175, 35))
        self.PropTxt.setSizeIncrement(QtCore.QSize(0, 0))
        self.PropTxt.setFrameShape(QtWidgets.QFrame.Panel)
        self.PropTxt.setLineWidth(2)
        self.PropTxt.setObjectName("PropTxt")

        self.ResultTxt = QtWidgets.QTextEdit(self.centralwidget)
        self.ResultTxt.setGeometry(QtCore.QRect(42, 220, 175, 35))
        self.ResultTxt.setSizeIncrement(QtCore.QSize(0, 0))
        self.ResultTxt.setFrameShape(QtWidgets.QFrame.Panel)
        self.ResultTxt.setLineWidth(2)
        self.ResultTxt.setObjectName("ResultTxt")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 735, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.LoadBtn.setText(_translate("MainWindow", "Load Image"))
        self.LoadBtn_2.setText(_translate("MainWindow", "Analyse"))




