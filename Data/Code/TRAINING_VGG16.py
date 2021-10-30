import glob
import numpy as np
import pandas as pd
import math
import seaborn as sns
import umap
import sys
import random
import warnings
import cv2
import os
import shutil
import itertools
import imutils
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib import pyplot
from numpy import expand_dims
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile,join
from collections import Counter
from random import shuffle
from collections import Counter
from itertools import chain
from tqdm import tqdm


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import tensorflow as tf
import keras
from keras import layers
from keras import backend as K
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.merge import concatenate
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.models import Model, Sequential,load_model
from keras.optimizers import Adam, RMSprop
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array,array_to_img

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array



RANDOM_SEED = 123
files=glob.glob('Data/*')

Tumor_files=[fn for fn in files if 'Y' in fn]
Normal_files=[fn for fn in files if 'n' in fn]

print(len(Tumor_files),len(Normal_files))


Tumor_train=np.random.choice(Tumor_files,size=124, replace=False)
Normal_train=np.random.choice(Normal_files,size=69,replace=False)
Tumor_files=list(set(Tumor_files)-set(Tumor_train))
Norma_files=list(set(Normal_files)-set(Normal_train))

Tumor_val=np.random.choice(Tumor_files,size=26, replace=False)
Normal_val=np.random.choice(Normal_files,size=24, replace=False)
Tumor_files=list(set(Tumor_files)-set(Tumor_val))
Norma_files=list(set(Normal_files)-set(Normal_val))
                            
Tumor_test=np.random.choice(Tumor_files,size=5, replace=False)
Normal_test=np.random.choice(Normal_files,size=5, replace=False)

print('Tumor Dataset:', Tumor_train.shape, Tumor_val.shape, Tumor_test.shape)
print('Normal Dataset:', Normal_train.shape, Normal_val.shape, Normal_test.shape)


os.mkdir('TRAIN') if not os.path.isdir('TRAIN') else None
os.mkdir('TRAIN/YES') if not os.path.isdir('TRAIN/YES') else None
os.mkdir('TRAIN/NO') if not os.path.isdir('TRAIN/NO') else None
os.mkdir('VAL') if not os.path.isdir('VAL') else None
os.mkdir('VAL/YES') if not os.path.isdir('VAL/YES') else None
os.mkdir('VAL/NO') if not os.path.isdir('VAL/NO') else None
os.mkdir('TEST') if not os.path.isdir('TEST') else None
os.mkdir('TEST/YES') if not os.path.isdir('TEST/YES') else None
os.mkdir('TEST/NO') if not os.path.isdir('TEST/NO') else None

#os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
#os.mkdir(test_dir) if not os.path.isdir(test_dir) else None


for fn in Normal_train:
    shutil.copy(fn,'TRAIN/NO')

for fn in Tumor_train:
    shutil.copy(fn,'TRAIN/YES')

for fn in Normal_val:
    shutil.copy(fn,'VAL/NO')

for fn in Tumor_val:
    shutil.copy(fn,'VAL/YES')

for fn in Normal_test:
    shutil.copy(fn,'TEST/NO')

for fn in Tumor_test:
    shutil.copy(fn,'TEST/YES')


def load_data(dir_path, img_size=(100,100)):
  
    #Load resized images as np.arrays to workspace
    
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels


TRAIN_DIR = 'TRAIN/'
TEST_DIR = 'TEST/'
VAL_DIR = 'VAL/'
IMG_SIZE = (224,224)

X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

def plot_samples(X, y, labels_dict, n=50):
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)

        plt.figure(figsize=(15,6))
        c = 1
        for img in imgs:
            plt.subplot(i,j,c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('Tumor: {}'.format(labels_dict[index]))
        plt.show()

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

X_train_crop = crop_imgs(set_name=X_train)
X_val_crop = crop_imgs(set_name=X_val)
X_test_crop = crop_imgs(set_name=X_test)


def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)
        i += 1

os.mkdir('TRAIN_CROP') if not os.path.isdir('TRAIN_CROP') else None
os.mkdir('TRAIN_CROP/YES') if not os.path.isdir('TRAIN_CROP/YES') else None
os.mkdir('TRAIN_CROP/NO') if not os.path.isdir('TRAIN_CROP/NO') else None
os.mkdir('VAL_CROP') if not os.path.isdir('VAL_CROP') else None
os.mkdir('VAL_CROP/YES') if not os.path.isdir('VAL_CROP/YES') else None
os.mkdir('VAL_CROP/NO') if not os.path.isdir('VAL_CROP/NO') else None
os.mkdir('TEST_CROP') if not os.path.isdir('TEST_CROP') else None
os.mkdir('TEST_CROP/YES') if not os.path.isdir('TEST_CROP/YES') else None
os.mkdir('TEST_CROP/NO') if not os.path.isdir('TEST_CROP/NO') else None


save_new_images(X_train_crop, y_train, folder_name='TRAIN_CROP/')
save_new_images(X_val_crop, y_val, folder_name='VAL_CROP/')
save_new_images(X_test_crop, y_test, folder_name='TEST_CROP/')

def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)

X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)

#plot_samples(X_train_prep, y_train, labels, 10)




demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1./255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)

"""
os.mkdir('preview')
x = X_train_crop[0]  
x = x.reshape((1,) + x.shape) 

i = 0
for batch in demo_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='aug_img', save_format='jpg'):
    i += 1
    if i > 20:
        break 
"""

TRAIN_DIR = 'TRAIN_CROP/'
VAL_DIR = 'VAL_CROP/'


train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=RANDOM_SEED
)


validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)


vgg16_weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg = VGG16(
    weights=vgg16_weight_path,
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)

#plot feature map

f=plt.figure(figsize=(16,16))
model=VGG16()
model=Model(input=model.inputs, outputs=model.layers[1].output)
model.summary()

img=img_to_array(X_val_prep[43])
img=expand_dims(img, axis=0)
img=preprocess_input(img)
feature_maps=model.predict(img)
square=8
ix=1
for _ in range (square):
    for _ in range(square):
        ax=pyplot.subplot(square,square,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='viridis')
        ix+1
#pyplot.show()

NUM_CLASSES = 1

vgg16 = Sequential()
vgg16.add(vgg)
vgg16.add(layers.Dropout(0.3))
vgg16.add(layers.Flatten())
vgg16.add(layers.Dropout(0.5))
vgg16.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

vgg16.layers[0].trainable = False

vgg16.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])


adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

vgg16.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
vgg16.summary()

"""
model=VGG16()
ixs=[2,5,9,13,17]
outputs=[model.layers[i].output for i in ixs]
model=Model(inputs=model.inputs, outputs=outputs)
img=img_to_array(X_val_prep[43])
img=expand_dims(img,axis=0)
feature_maps=model.predict(img)
square=8
for fmap in feature_maps:
    ix=1
    for _ in range(square):
        plt.figure(figsize=(64,64))
        for _ in range(square):
            ax=pyplot.subplot(square,square,ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(fmap[0, :, :, ix-1], cmap='viridis')
            ix+=1
    plt.show()        
"""
"""
import time

start = time.time()

vgg16_history = vgg16.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=120,
    validation_data=validation_generator,
    validation_steps=30,
)


end = time.time()
print(end - start)
"""
