import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, merge
from keras.layers import Activation
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, activity_l2, l1, activity_l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.layers.pooling import GlobalAveragePooling2D

BOUND_SIZE = 32
def get_model():
    model = Sequential([
        Lambda(norm_input, input_shape=(1,BOUND_SIZE,BOUND_SIZE)),
        Convolution2D(64,2,2, activation='relu'), #31
        BatchNormalization(axis=1),
        Convolution2D(64,2,2, activation='relu'), #30
        BatchNormalization(axis=1),
        Convolution2D(64,2,2, subsample=(2, 2), activation='relu'), #15
        BatchNormalization(axis=1),
        Convolution2D(128,2,2, activation='relu'), #14
        BatchNormalization(axis=1),
        Dropout(0.3),
        Convolution2D(128,2,2, activation='relu'), #13
        BatchNormalization(axis=1),
        Dropout(0.3),
        Convolution2D(128,2,2, subsample=(2, 2), activation='relu'), #6
        BatchNormalization(axis=1),
        Dropout(0.3),
        Convolution2D(192,2,2, activation='relu'), #5
        BatchNormalization(axis=1),
        Dropout(0.4),
        Convolution2D(192,2,2, activation='relu'), #4
        BatchNormalization(axis=1),
        Dropout(0.4),
        Convolution2D(192,2,2, subsample=(2, 2), activation='relu'), #2
        BatchNormalization(axis=1),
        Dropout(0.4),
        Convolution2D(256,2,2, activation='relu'), #1
        BatchNormalization(axis=1),
        Dropout(0.5),
        Convolution2D(256,1,1, activation='relu'), #1
        BatchNormalization(axis=1),
        Dropout(0.5),
        Convolution2D(2,1,1, activation='relu'), #1
        GlobalAveragePooling2D(),
        Activation('softmax')
        ])
    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model