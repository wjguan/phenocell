import os
import numpy as np
import clarity.IO as io

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
from keras.utils.data_utils import get_file

BOUND_SIZE = 32
ModelsDirectory = r'D:\\analysis\\models'


#ProcessedDirectory = r'D:\\analysis\\data\\processed'
#mean_px = np.load(ProcessedDirectory+r"\\mean.npy")
#std_px = np.load(ProcessedDirectory+r"\\std.npy")
#def norm_input(x): return (x-mean_px)/std_px

class ConvNet():
    def __init__(self, model_fname, norm_input, num_channels=1):
        self.create(model_fname, norm_input, num_channels)

    def predict(self, imgs):
        print("predicting")
        probs = self.model.predict_proba(imgs)[:,1]
        return probs

    def create(self, model_fname, norm_input, num_channels):
        model = self.model = Sequential([
        Lambda(norm_input, input_shape=(num_channels,BOUND_SIZE,BOUND_SIZE)),
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
        fname = os.path.join(ModelsDirectory, model_fname)
        model.load_weights(fname)

def classifyPoints(img, source, sink, model = 'GFAP_0.6x0.6x3_Leica_iba1', x = all, y = all, z = all, orientation=(1,2,3), threshold=0.8):
    points, intensities = io.readPoints(source)

    if isinstance(z,tuple):
        mask = (z[0] < points[:,2]) * (z[1] > points[:,2])
        points = points[mask]
        points[:,2] -= z[0]
        intensities = intensities[mask]
        
    img = io.readData(img, x=x, y=y, z=z)
    img_padded = np.pad(img, BOUND_SIZE//2, 'constant')
    
    z = list(map(abs,orientation)).index(3)
    
    X = np.zeros((len(points),1,BOUND_SIZE,BOUND_SIZE),dtype='uint16')
    for i in range(len(points)):
        s = points[i].astype('uint16')
        if z == 2:
            X[i,0,:,:] = img_padded[s[0]:s[0]+BOUND_SIZE, s[1]:s[1]+BOUND_SIZE, s[2]+BOUND_SIZE//2]
        elif z == 1:
            X[i,:,:,0] = img_padded[s[0]:s[0]+BOUND_SIZE, s[1]+BOUND_SIZE//2, s[2]:s[2]+BOUND_SIZE]
        elif z == 0:
            X[i,:,:,0] = img_padded[s[0]+BOUND_SIZE//2, s[1]:s[1]+BOUND_SIZE, s[2]:s[2]+BOUND_SIZE]
            
    mean_px = X.mean().astype(np.float32)
    std_px = X.std().astype(np.float32)
    def norm_input(x): return (x-mean_px)/std_px

    model = ConvNet(model+'.hdf5', norm_input)
    print("Model initialized")
    probs = model.predict(X)
    print("Prediction finished")
    preds = probs > threshold
    points = points[preds]
    intensities = intensities[preds]
    return io.writePoints(sink, (points, intensities))