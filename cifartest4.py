# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import tensorflow as tf
import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def minmax_normalize(X):
    min = np.min(X)
    max = np.max(X)
    Xn = (X-min)/(max-min)
    return Xn

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#train_img, test_img = train_images / 255.0, test_images / 255.0 #convert values to floating point

#train_img, test_img = minmax_normalize(train_images), minmax_normalize(test_images) #convert values to floating point

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def create_NNinput(images):
    NN_input = np.empty([len(images),32,32,18])
    img = images/255.0  #convert to floating point values
    
    #z-score matrix
    epsilon = 1e-7
    mean = np.mean(img)
    std = np.std(img)
    zscore = (img - mean)/(std + epsilon)
    
    NN_input[:,:,:,0:3] = img
    NN_input[:,:,:,3:6] = zscore
    h_sharp = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    for i in range(len(images)):
        gray_img = cv2.cvtColor(images[i],cv2.COLOR_BGR2GRAY)  
#        gray_edgex_img = cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize =3)
#        gray_edgey_img = cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize =3)
#        gray_edge_img =  gray_edgex_img + gray_edgey_img
        gray_lap = cv2.Laplacian(gray_img,cv2.CV_64F)
        gray_edge_enhanced = gray_img - gray_lap    
        blurred_img = cv2.GaussianBlur(images[i],(5,5),0)
        sharp_img = cv2.filter2D(images[i],-1,h_sharp)
        hsv = cv2.cvtColor(images[i],cv2.COLOR_BGR2HSV)
        
        NN_input[i,:,:,6] = gray_img /255.0
        NN_input[i,:,:,7:10] = hsv/255.0
#        NN_input[i,:,:,7] = gray_edgex_img /255.0
#        NN_input[i,:,:,8] = gray_edgey_img /255.0
#        NN_input[i,:,:,9] = gray_edge_img /255.0
        NN_input[i,:,:,10] = gray_lap /255.0
        NN_input[i,:,:,11] = gray_edge_enhanced /255.0
        NN_input[i,:,:,12:15] = blurred_img /255.0
        NN_input[i,:,:,15:18] = sharp_img /255.0
    return NN_input

train_input = create_NNinput(train_images)
test_input = create_NNinput(test_images)

print('dataset organization complete \n')
# Creating a NNET model------------------------------------------------------------------------- 
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = "D:/CS273A/project"+checkpoint_path

# Create a callback that saves the model's weights every n epochs
n = 1
cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto', period=n)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 18)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 18)),
    tf.keras.layers.BatchNormalization(),
#        tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
#        tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
#        tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
#        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
        ])
#model = tf.keras.models.Sequential([
#        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 15)),
#	tf.keras.layers.BatchNormalization(),
#	tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 15)),
#	tf.keras.layers.BatchNormalization(),
##        tf.keras.layers.MaxPooling2D((2, 2)),
#        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
#	tf.keras.layers.Dropout(0.2),
#        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
#	tf.keras.layers.BatchNormalization(),
##        tf.keras.layers.MaxPooling2D((2, 2)),
#        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
#	tf.keras.layers.Dropout(0.2),
#        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
#	tf.keras.layers.BatchNormalization(),
##        tf.keras.layers.MaxPooling2D((2, 2)),
#        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
#	tf.keras.layers.Dropout(0.2),
#        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
#	tf.keras.layers.BatchNormalization(),
##        tf.keras.layers.MaxPooling2D((2, 2)),
#        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
#	tf.keras.layers.Dropout(0.2),
#        tf.keras.layers.Flatten(),
#        #tf.keras.layers.Dense(128, activation='relu'),
#        #tf.keras.layers.Dense(256, activation='relu'),
#        #tf.keras.layers.Dense(512, activation='relu'),
#        #tf.keras.layers.Dense(1024, activation='relu'),
#        tf.keras.layers.Dense(10, activation='softmax')
#        ])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# training the model ------------------------------------------------------------------------------
history = model.fit(
        train_input, train_labels, epochs=25,
        validation_data=(test_input, test_labels))

# plotting results --------------------------------------------------------------------------------
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
#plt.xlim([0, 11])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_input, test_labels, verbose=2)
model.save('my_model4.h5') 

