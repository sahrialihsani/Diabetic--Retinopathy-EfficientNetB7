from keras.preprocessing.image import img_to_array,load_img
import numpy as np
from efficientnet.tfkeras import EfficientNetB7
from keras.applications.efficientnet import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image, ImageOps
from tkinter import filedialog
import cv2 as cv

classes = { 0:'Normal',
            1:'Mild', 
            2:'Moderate', 
            3:'Severe', 
            4:'Proliferative DR'
          }
# def myModel(model,img):
#     img = img.resize((224,224))
#     img = np.expand_dims(img, axis=0)
#     img = np.array(img)
#     pred = np.arg(model.predict,axis=1)
#     Result = classes[pred]
#     return Result

# def probImg(model,img):
#     img=img_to_array(img)/255 #add this one
#     img=np.expand_dims(img,axis=0)
#     img=preprocess_input(img)
#     Probability=model.predict_proba(img.reshape(1,224,224,3))
#     return Probability

def myModel(model,img):
    img=img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    Result = np.argmax(model.predict(img), axis=1)
    return Result

def probImg(model,img):
    img=img_to_array(img)/255 #add this one
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    Probability=model.predict(img.reshape(1,224,224,3))
    return Probability