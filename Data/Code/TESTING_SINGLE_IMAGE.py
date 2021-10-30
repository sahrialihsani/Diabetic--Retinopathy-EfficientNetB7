import cv2
from keras.applications.efficientnet import EfficientNetB7, preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array,load_img
from PIL import Image
import numpy as np
from skimage import transform
import keras
from keras import Model
from keras.preprocessing import image
from Watershed_seg import Watershed



IMG_SIZE = (224,224)

def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

imgpath='TEST_CROP/YES/29.jpg'
img = image.load_img(imgpath, target_size=IMG_SIZE)
img2=cv2.imread(imgpath)

from keras.models import load_model
vgg16=load_model('MODELS/VGG16_120_epochs_1.h5')
preds = predict(vgg16, img)
result = preds[0]
if result==0:
    print ("NORMAL")
    cv2.imshow("Normal", img2)
elif result==1:
    print("Terindikasi Brain Tumor")
    Hasil=Watershed(img2)
    cv2.imshow("Terindikasi Brain Tumor",Hasil)
     






























