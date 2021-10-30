from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap

import cv2
import numpy as np
from efficientnet.tfkeras import EfficientNetB7
from keras.applications.efficientnet import EfficientNetB7
from keras import Model
from keras.models import load_model
from keras.preprocessing import image
from skimage.filters import threshold_otsu

from myModel import myModel, probImg

import sys
from GUI import *
#dictionary to label all traffic signs class.

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent=parent)
        self.ui=Ui_MainWindow()
        self.ui.setupUi(self)

        self.model=load_model("MODELS/efficientnetb7_20.h5")
        self.ui.LoadBtn.clicked.connect(self.load_img)
        self.ui.LoadBtn_2.clicked.connect(self.Predict)

    def load_img(self):
        global fileName
        fileName,_=QtWidgets.QFileDialog.getOpenFileName(None, "Select Image", "", "Image Files(*.png *.jpg *.jpeg *.bmp *.tif)")
        print(fileName)
        self.ui.OutImg.clear()
        self.ui.PropTxt.clear()
        self.ui.ResultTxt.clear()
        print(fileName)
        if fileName:
            pixmap=QtGui.QPixmap(fileName)
            pixmap=pixmap.scaled(self.ui.inputImg.width(),self.ui.inputImg.height(),QtCore.Qt.KeepAspectRatio)
            self.ui.inputImg.setPixmap(pixmap)
            self.ui.inputImg.setAlignment(QtCore.Qt.AlignCenter)
    
    
    def Predict(self):
        img=cv2.imread(fileName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224,224))
        img_t = cv2.addWeighted(img,4, cv2.GaussianBlur(img , (0,0) , 30) ,-4 ,128)
        cv2.imwrite("Data/img/WaterShed.jpg",img_t)
        pixmap=QtGui.QPixmap("Data/img/WaterShed.jpg")
        pixmap=pixmap.scaled(self.ui.OutImg.width(),self.ui.OutImg.height(),QtCore.Qt.KeepAspectRatio)
        self.ui.OutImg.setPixmap(pixmap)
        self.ui.OutImg.setAlignment(QtCore.Qt.AlignCenter)
        Data=image.load_img(fileName,target_size=(224,224))
        Data2=image.load_img(fileName,target_size=(224,224,3))
        Result=myModel(self.model,Data)
        Probability=probImg(self.model,Data2)
        i = Probability.argmax(axis=1)[0]
        if(Result==0):
            print("Tidak Terindikasi DR")
            Result="Tidak Terdeteksi DR" 
            self.ui.ResultTxt.setText("Tidak Terindikasi DR")
            self.ui.DetTxt.setText(Result)
            self.ui.PropTxt.setText("Probabilitas: "+ str(Probability[0][i]))
        elif(Result==1):
            print("Terindikasi DR")
            Result="Mild"
            self.ui.ResultTxt.setText("Terindikasi DR")
            self.ui.DetTxt.setText("Jenis DR: "+ str(Result))
            self.ui.PropTxt.setText("Probabilitas: "+ str(Probability[0][i]))
        elif(Result==2):
            print("Terindikasi DR")
            Result="Moderate" 
            self.ui.ResultTxt.setText("Terindikasi DR")
            self.ui.DetTxt.setText("Jenis DR: "+ str(Result))
            self.ui.PropTxt.setText("Probabilitas: "+ str(Probability[0][i]))
        elif(Result==3):
            print("Terindikasi DR")
            Result="Severe"         
            self.ui.ResultTxt.setText("Terindikasi DR")
            self.ui.DetTxt.setText("Jenis DR: "+ str(Result))
            self.ui.PropTxt.setText("Probabilitas: "+ str(Probability[0][i]))
        elif(Result==4):
            print("Terindikasi DR")
            Result="Proliferative DR"
            self.ui.ResultTxt.setText("Terindikasi DR")
            self.ui.DetTxt.setText("Jenis DR: "+ str(Result))
            self.ui.PropTxt.setText("Probabilitas: "+ str(Probability[0][i]))

        # if(Result==0 or Result==1 or Result==2 or Result==3 or Result==4):
        #     print("Prediksi Kelas: ",Result)
        #     print("Probabilitas: ",Probability[0][i])
        # else:
        #     print("DR Lainnya")
        #     print("Probabilitas: ",Probability[0][i])

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())