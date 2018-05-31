import bcolz, os, time
import numpy as np
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, merge
from keras.layers import Activation
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.merge import concatenate 

from metrics import * 
from objectives import * 


class TriPlanarNet(object):
    def __init__(self, ModelFile, img_dim, num_channels=1, batch_size=32):
        self.ModelFile = ModelFile
        self.img_dim = img_dim 
        self.batch_size = batch_size 
        self.num_channels = num_channels # coudl be more if we use nucleus channel 
    
    def singlePlaneModel(self):
        input = Input((self.img_dim, self.img_dim, self.num_channels))
        x = Conv2D(64, (2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(input)
        x = BatchNormalization()(x)
        x = Conv2D(64, (2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (2,2), strides=(2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(128, (2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Conv2D(128, (2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Conv2D(128, (2,2), strides=(2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Conv2D(192, (2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Conv2D(192, (2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Conv2D(192, (2,2), strides=(2,2), activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(1e-3))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        
        model = Model(inputs=input, outputs=x)
        return model
        
    def getModel(self, class_weights=None):
        # Try parameter sharing between yz and xz networks 
        inputxy = Input((self.img_dim, self.img_dim, self.num_channels))
        inputyz = Input((self.img_dim, self.img_dim, self.num_channels)) 
        inputxz = Input((self.img_dim, self.img_dim, self.num_channels)) 
        
        # Make different models for xz,yz and xy 
        yzxz_model = self.singlePlaneModel()
        yz = yzxz_model(inputyz)
        xz = yzxz_model(inputxz)
        
        xy_model = self.singlePlaneModel()
        xy = xy_model(inputxy) 
        
        x = concatenate([xy,xz,yz])
        
        ## TODO: see if this is the optimal configuration 
        x = Conv2D(256, (2,2), activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(256, (1,1), activation='relu', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Conv2D(2, (1,1), activation='relu', kernel_initializer='he_normal')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax')(x)
        
        model = Model([inputxy, inputxz, inputyz], x)
        if class_weights is not None:
            model.compile(Adam(), loss=weighted_pixelwise_crossentropy(class_weights), metrics=['accuracy',dice])
        else:
            model.compile(Adam(), loss=generalizedDiceLoss, metrics=['accuracy',dice])
        model.summary()
        return model
        
    def train(self, X, y, max_epochs, loadModel=False, class_weights=None, use_val_set=True):
        if not loadModel:
            model = self.getModel(class_weights)
        else:
            model = load_model(self.ModelFile, custom_objects={'dice':dice, 'modifiedDiceLoss':modifiedDiceLoss,
                                                               'generalizedDiceLoss':generalizedDiceLoss})
        if use_val_set:
            model_checkpoint = ModelCheckpoint(self.ModelFile, monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=False, mode='min')
            early = EarlyStopping(monitor='val_loss', mode = 'min', patience = 25) 
            X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.15,random_state=33)
            X_train_xy = X_train[:,0,:,:,:]; X_val_xy = X_val[:,0,:,:,:]
            X_train_xz = X_train[:,1,:,:,:]; X_val_xz = X_val[:,1,:,:,:]
            X_train_yz = X_train[:,2,:,:,:]; X_val_yz = X_val[:,2,:,:,:]
        
        # gen = image.ImageDataGenerator(rotation_range=30,
                                # width_shift_range=0.0,
                                # height_shift_range=0.0,
                                # shear_range=0.3,
                                # zoom_range=0.3,
                                # horizontal_flip=True,
                                # vertical_flip=True,
                                # fill_mode='nearest')
        # batches = gen.flow(X_train, y_train, batch_size=self.batch_size)
        # test_batches = gen.flow(X_val, y_val, batch_size=self.batch_size)
        # history = model.fit_generator(batches, steps_per_epoch=int(np.ceil(len(X_train)/self.batch_size)), epochs=500, verbose=1,
                            # validation_data=test_batches, validation_steps=int(np.ceil(len(X_val)/self.batch_size)), callbacks=[model_checkpoint, early])
        
            history = model.fit([X_train_xy, X_train_xz, X_train_yz], y_train, batch_size = self.batch_size, epochs=max_epochs,shuffle=True, 
                                 validation_data=([X_val_xy,X_val_xz,X_val_yz],y_val),verbose=1, callbacks=[model_checkpoint,early])
        else:
            X_train = X; y_train = y 
            X_train_xy = X_train[:,0,:,:,:]
            X_train_xz = X_train[:,1,:,:,:]
            X_train_yz = X_train[:,2,:,:,:]
            model_checkpoint = ModelCheckpoint(self.ModelFile, monitor='loss',verbose=1, save_best_only=True, save_weights_only=False, mode='min')
            history = model.fit([X_train_xy, X_train_xz, X_train_yz], y_train, batch_size = self.batch_size, epochs=max_epochs,shuffle=True, 
                                 verbose=1, callbacks=[model_checkpoint])
        
        # plt.plot(history.history['acc']) 
        # plt.show()
        # plt.plot(history.history['val_acc'])
        # plt.show()
        return history 
        
    def predictModel(self, datas, class_weights=None):
        # Makes prediction 
        # datas = numpy array that we will predict 
        # imageSink = where we want to save the output of the predictions 
        model = self.getModel(class_weights)
        model.load_weights(self.ModelFile)
        
        print('predict some data')
        imgs_mask_test = model.predict(datas, verbose=1)

        return imgs_mask_test 
    
    def validateModel(self, X, y, categorical):
        # categorical = True if we're using softmax multi-cateogircal classification  
        ## Test 
        #model = load_model(self.ModelFile, custom_objects = {'dice':dice})
        model = self.getModel()
        model.load_weights(self.ModelFile) 
        X = normalize(X)
        y_pred = model.predict([X[:,0,:,:,:],X[:,1,:,:,:],X[:,2,:,:,:]], verbose=1)
        y_pred = y_pred > 0.5 # binary mask the vector 
        # compute the accuracy, dice 
        fone = npdice(y, y_pred, categorical) 
        acc = compute_accuracy(y, y_pred, categorical) 
        return fone, acc 
        
def normalize(X, channel_axis=-1):
    # we want to normalize along each given channel axis, which the default for is -1 
    X_norm = np.zeros(X.shape) 
    for i in range(X.shape[channel_axis]): # assume that channels come last 
        if len(X.shape) == 4: #i.e. not triplanar or LSTM:
            mean_px = X[:,:,:,i].mean().astype('float32')
            std_px = X[:,:,:,i].std().astype('float32')
            if std_px != 0:
                X_norm[:,:,:,i] = (X[:,:,:,i]-mean_px)/std_px 
            else:
                X_norm[:,:,:,i] = X[:,:,:,i] 
        elif len(X.shape) == 5:
            mean_px = X[:,:,:,:,i].mean().astype('float32')
            std_px = X[:,:,:,:,i].std().astype('float32')
            if std_px != 0:
                X_norm[:,:,:,:,i] = (X[:,:,:,:,i]-mean_px)/std_px 
            else:
                X_norm[:,:,:,:,i] = X[:,:,:,:,i]
        else:
            raise RuntimeError('X.shape should be of length 4 or 5, instead got length %d'%(int(len(X.shape))))
        
    return X_norm 
    
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
    
def loadData(XSource, YSource, categorical): 
    X = np.load(XSource)
    if YSource is not None:
        y = np.load(YSource)
        if categorical:
            y = to_categorical(y) 
        return X,y
    else:
        return X
    
if __name__ == '__main__':
    start = time.time()
    ## USER INPUT 
    train = True; test = True 
    continueTrain = False
    categorical = False
    NNParameter = {'ModelFile': r'D:\analysis\models\TRAP_triplanar.hdf5',
                   'img_dim' : 32,
                   'batch_size': 32}
    ProcessedDirectory = r'D:\analysis\data\processed\TRAP\TriPlanar\TriPlanar'
    ## END USER INPUT                
    
    
    X_train, y_train = loadData(os.path.join(ProcessedDirectory,r'X_train.npy'), 
                                os.path.join(ProcessedDirectory,r'y_train.npy'), categorical)
                   
    mynet = TriPlanarNet(**NNParameter)
    if train:
        history = mynet.train(X_train,y_train,continueTrain)
        
    
    if test:
        fone, acc = mynet.validateModel(X_train, y_train, categorical)
        print("Train DICE Coefficient:", fone)
        print("Train accuracy:", acc)
        
        X_test, y_test = loadData(os.path.join(ProcessedDirectory,r'X_test.npy'), 
                                  os.path.join(ProcessedDirectory,r'y_test.npy'), categorical)
        fone, acc = mynet.validateModel(X_test, y_test, categorical)
        print("Test DICE Coefficient:", fone)
        print("Test accuracy:", acc) 
    print('Time elapsed:',(time.time()-start)/60)