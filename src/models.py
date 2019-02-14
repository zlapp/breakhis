import os
import sys
import glob
import shutil
import random
from random import randint
import numpy as np 
from PIL import Image
from PIL import ImageFilter


import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Reshape, Input, Dense, GlobalAveragePooling2D
from keras.layers.core import Activation, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD 

import matplotlib.pyplot as plt



####### Fine tune a model

def ft_pre_model(base_model, data_loc, out_loc, magnification, num_out, epochs_first=10, img_dim=512, layer_train=249, epochs=100):   
    # Params
    epochs_first=int(epochs_first)
    batch_size=16   # make this divisible by len(x_data)
    img_dim=int(img_dim)
    layer_train = int(layer_train)
    
    # Paths to data
    if not os.path.exists(out_loc):
        os.makedirs(out_loc)
    train_loc = os.path.join(str(data_loc), str(magnification), 'train')
    valid_loc = os.path.join(str(data_loc), str(magnification), 'valid')
    num_train = len(glob.glob(train_loc + '/**/*.png', recursive=True))
    num_valid = len(glob.glob(valid_loc + '/**/*.png', recursive=True))
    print('num_train', num_train)
    print('num_valid', num_valid)

    # Set the number of steps per epoch
    steps_per_epoch = np.floor(num_train/batch_size) # num of batches from generator at each epoch. (make it full train set)
    validation_steps = np.floor(num_valid/batch_size)# size of validation dataset divided by batch size
    print('steps_per_epoch', steps_per_epoch)
    print('validation_steps', validation_steps)

    # Image generators
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.1,
        height_shift_range=.2,
        width_shift_range=.2,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True)

    generator = datagen.flow_from_directory(
            train_loc,
            target_size=(img_dim, img_dim),
            batch_size=batch_size,
            class_mode='categorical')

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    valid_generator = valid_datagen.flow_from_directory(
            valid_loc,
            target_size=(img_dim, img_dim),
            batch_size=batch_size,
            class_mode='categorical')

    # train the last layers first. We don't need this to be perfect, just get resonable weights before fine tuning
    # add a global spatial average pooling layer for visualization
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_out, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print('Training the top layers only')
    hist = model.fit_generator(generator,
                                  validation_data=valid_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs_first,
                                  validation_steps=validation_steps)

    print('Make sure nothing is going terribly wrong with training last layers')
    print('Final 2 Epochs Avg Validation loss: ', np.mean(hist.history['val_acc'][-2:]))


    plt.plot(hist.history['loss'])    
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Training top layers')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Train acc', 'val_acc'], loc='upper left')
    plt.figure(figsize=(10,6))
    plt.show()
    
    # at this point, the top layers are well trained and we can start fine-tuning convolutional layers 
    # We will freeze the bottom N layers and train the remaining top layers.
    for layer in model.layers[:layer_train]:
       layer.trainable = False
    for layer in model.layers[layer_train:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    print('Fine tuning layers')
    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=0),
                 ModelCheckpoint(filepath=os.path.join(out_loc, 'fine_tune'+'_.{epoch:02d}-{val_acc:.2f}.hdf5'), 
                                 verbose=1, monitor='val_loss', save_best_only=True)]

    hist = model.fit_generator(generator,
                                  validation_data=valid_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=epochs,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks)

    print('Check fine-tuning')
    print('Final 5 Epochs Avg Validation loss: ', np.mean(hist.history['val_acc'][-5:]))
    plt.plot(hist.history['loss'])    
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Fine tuning model')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Train acc', 'val_acc'], loc='upper left')
    plt.figure(figsize=(10,6))
    plt.show()


############################


