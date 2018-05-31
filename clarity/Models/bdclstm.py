import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Bidirectional, Dropout, Cropping2D, Dense, Flatten, Reshape, Activation, Conv2DTranspose, TimeDistributed, Lambda
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.merge import concatenate 
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as K
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import class_weight 

from matplotlib import pyplot as plt 
from pathlib import Path
from objectives import * 
from metrics import * 
import os.path, bcolz

"""
This module introduces the BDC LSTM class, that we can use to extract 3 dimensional information from images.
"""
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

class DeepBdcLSTM(object):
    def __init__(self, ProcessedDirectory, ModelFile, num_hidden_layers, filters = 64, img_rows = 64, img_cols = 64, num_context_slices=1, batch_size=32):
        # img_rows = the number of rows in each feature map 
        # img_cols = the number of columns in each feature map 
        # num_context_slices = +/- the number of slices to consider for each 2D image i.e. 2*n+1 sequence of slices to classify middle image 
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.time_steps = 2*num_context_slices+1
        self.ProcessedDirectory = ProcessedDirectory
        self.ModelFile = ModelFile 
        self.filters = filters 
        self.num_hidden_layers = num_hidden_layers 
        self.batch_size = batch_size
    
    def loadData(self, categorical=False):
        
        Xn = load_array(os.path.join(self.ProcessedDirectory,r'X_train_1zslices.bc'))
        Xn = (Xn-Xn.mean())/Xn.std()
        X = Xn.astype('float32')
        Y = load_array(self.ProcessedDirectory+r'\\y_train.bc')
        if categorical:
            Y = to_categorical(Y)
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
        return X_train, X_val, y_train, y_val 

    
    def loadTestData(self, categorical=False):
        # Test sets 
        Xt = load_array(self.ProcessedDirectory+r'\\X_test_1zslices.bc')
        Xt = (Xt-Xt.mean())/Xt.std()
        Xt = Xt.astype('float32')
        
        Yt = load_array(self.ProcessedDirectory+r'\\y_test.bc')
        if categorical:
            Yt = to_categorical(Yt)
        return Xt, Yt 
    
    def getModel(self):
        inputs = Input((self.time_steps, self.img_rows, self.img_cols, 1))
        bdc = Bidirectional(ConvLSTM2D(return_sequences=True, kernel_size=(3,3), filters=self.filters, activation='relu', dropout=0.5, recurrent_dropout=0.5), merge_mode='ave')(inputs) 
        bdc = Bidirectional(ConvLSTM2D(return_sequences=True, kernel_size=(3,3), filters=self.filters, activation='relu', dropout=0.5, recurrent_dropout=0.5), merge_mode='ave')(bdc) 
        bdc = Bidirectional(ConvLSTM2D(return_sequences=True, strides=(2,2), kernel_size=(2,2), filters=self.filters, activation='relu'), merge_mode='ave')(bdc)
                
        if self.num_hidden_layers > 1:
            for _ in range(self.num_hidden_layers - 1):
                bdc = Bidirectional(ConvLSTM2D(return_sequences=True, kernel_size=(3,3), filters=self.filters, activation='relu', dropout=0.5, recurrent_dropout=0.5), merge_mode='ave')(bdc) 
                bdc = Bidirectional(ConvLSTM2D(return_sequences=True, kernel_size=(3,3), filters=self.filters, activation='relu', dropout=0.5, recurrent_dropout=0.5), merge_mode='ave')(bdc)
                bdc = Bidirectional(ConvLSTM2D(return_sequences=True, strides=(2,2), kernel_size=(3,3), filters=self.filters, activation='relu'), merge_mode='ave')(bdc)

        dense = Flatten()(bdc) 
        #dense = Dense(400, activation='relu')(dense) 
        output = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=inputs, outputs=output)
        model.compile(Adam(), loss = modifiedDiceLoss, metrics = ['accuracy',dice])
        model.summary()
        return model
        
    def train(self):
        model = self.getModel()
        print("got BDC-LSTM")
        print("loading data")
        
        ## Solve json problem by doing only save_weights - could be caused by lambda layer 
        model_checkpoint = ModelCheckpoint(self.ModelFile, monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False, mode='min')
        early = EarlyStopping(monitor='loss', mode = 'min', patience = 25) 
        X_train, X_val, y_train, y_val = self.loadData()
        print("loading data done")
        gen = image.ImageDataGenerator3D(rotation_range=30,
                                width_shift_range=0.0,
                                height_shift_range=0.0,
                                shear_range=0.3,
                                zoom_range=0.3,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='nearest')
        batches = gen.flow(X_train, y_train, batch_size=self.batch_size)
        test_batches = gen.flow(X_val, y_val, batch_size=self.batch_size)
        history = model.fit_generator(batches, steps_per_epoch=int(np.ceil(len(X_train)/self.batch_size)), epochs=500, verbose=1,
                            validation_data=test_batches, validation_steps=int(np.ceil(len(X_val)/self.batch_size)), callbacks=[model_checkpoint, early])
        plt.plot(history.history['acc']) 
        plt.show()
        plt.plot(history.history['val_acc'])
        plt.show()
        return history 
        
    def predictModel(self, datas):
        # Makes prediction 
        # datas = numpy array that we will predict 
        # imageSink = where we want to save the output of the predictions 
        model = self.getModel()
        model.load_weights(self.ModelFile)
        
        print('predict some data')
        imgs_mask_test = model.predict(datas, verbose=1)

        return imgs_mask_test 
        
    def validateModel(self, categorical=False):
        # categorical = True if we're using softmax multi-cateogircal classification  
        ## Test 
        X_test, y_test = self.loadTestData(categorical)
        model = self.getModel()
        model.load_weights(self.ModelFile)
        y_pred = model.predict(X_test, verbose=1)
        y_pred = y_pred > 0.5 # binary mask the vector 
        # compute the accuracy, dice 
        fone = npdice(y_test, y_pred, categorical) 
        acc = compute_accuracy(y_test, y_pred, categorical) 
        print("Test DICE Coefficient:", fone)
        print("Test accuracy:", acc) 
        
        ## Train 
        X_train, X_val, y_train, y_val = self.loadData(categorical)
        X_train = np.concatenate((X_train, X_val), axis=0); y_train = np.concatenate((y_train, y_val), axis=0)
        y_pred = model.predict(X_train, verbose=1)
        y_pred = y_pred > 0.5 # binary mask the vector 
        # compute the accuracy, dice 
        fone = npdice(y_train, y_pred, categorical) 
        acc = compute_accuracy(y_train, y_pred, categorical) 
        print("Train DICE Coefficient:", fone)
        print("Train accuracy:", acc) 
        
def compute_accuracy(y_true, y_pred, categorical):
    # both of these are binary masks 
    if not categorical:
        y_true = to_categorical(y_true)
        y_pred = to_categorical(y_pred) 
    return np.sum(y_pred * y_true) / y_pred.shape[0]
    
def npdice(y_true, y_pred, categorical):
    # Numpy version of dice coefficient
    if categorical:
        y_true = y_true[:,1]
        y_pred = y_pred[:,1] # because these are one-hot representations, we only look at one of the vectors 
    y_int = y_true.flatten()*y_pred.flatten()
    return (2.*np.sum(y_int))/ (np.sum(y_true) + np.sum(y_pred))
    
if __name__ == '__main__':
    train = False
    predict = True 
    NNParameter = {'ProcessedDirectory': r'D:\analysis\data\processed\TRAP\LSTM',
                   'ModelFile': r'D:\analysis\models\TRAP_LSTM.hdf5',
                   'num_hidden_layers': 1,
                   'filters': 64,
                   'img_rows': 32,
                   'img_cols': 32,
                   'num_context_slices': 1
                   }
    net = DeepBdcLSTM(**NNParameter)
    
    if train:
        start = time.time()
        net.train()
        print("Time elapsed:",time.time()-start)
    if predict:
        net.validateModel(categorical=False)
    
    
    
    
    