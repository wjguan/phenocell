import bcolz, os, time,sys
import numpy as np
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Reshape, merge, TimeDistributed, Bidirectional, LSTM
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Activation
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.layers.merge import concatenate 
from keras.layers.pooling import GlobalAveragePooling2D

from metrics import * 
from objectives import * 
from imblearn.over_sampling import SMOTE # in case we have a class imbalance


## User input 
ProcessedDirectory = r'D:\analysis\data\interim\Syto_PV_r2_jae'
ModelsDirectory = r'D:\\analysis\\models'
model_fname = 'PV_Leica_jae'
## First part of user input

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def get_data(categorical):
    X = load_array(ProcessedDirectory + '\\X_train.bc')
    y = load_array(ProcessedDirectory + '\\y_train.bc')
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    if categorical:
        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val) 
    return X_train, X_val, y_train, y_val

def get_test_data(categorical):
    X = load_array(ProcessedDirectory + "\\X_test.bc")
    y = load_array(ProcessedDirectory + "\\y_test.bc")
    if categorical:
        y = to_categorical(y)
    return X, y
      
def get_model(BOUND_SIZE=32, channels=1, class_weights=None):
    '''Assume that larger bound size means higher resolution images  '''
    if BOUND_SIZE == 66: 
        num_first_layer = 32
        model = Sequential([
            Conv2D(num_first_layer, (2,2), activation='relu', kernel_initializer='he_normal', input_shape=(BOUND_SIZE,BOUND_SIZE,channels)), #65
            BatchNormalization(axis=-1),
            Conv2D(num_first_layer, (2,2), activation='relu', kernel_initializer='he_normal'), # 64 
            BatchNormalization(axis=-1),
            Conv2D(num_first_layer, (2,2), strides=(2,2), activation='relu', kernel_initializer='he_normal'), # 32
            BatchNormalization(axis=-1),
            Conv2D(64, (2,2), activation='relu', kernel_initializer='he_normal')]) # 31
    else:
        model = Sequential([
            Conv2D(64,(2,2), activation='relu', kernel_initializer='he_normal', input_shape=(BOUND_SIZE,BOUND_SIZE,channels))]) #31
            
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64,(2,2), activation='relu', kernel_initializer='he_normal')) #30
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64,(2,2), strides=(2, 2), activation='relu', kernel_initializer='he_normal')) #15
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(128,(2,2), activation='relu', kernel_initializer='he_normal')) #14
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.3))
    model.add(Conv2D(128,(2,2), activation='relu', kernel_initializer='he_normal')) #13
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.3))
    model.add(Conv2D(128,(2,2), strides=(2, 2), activation='relu', kernel_initializer='he_normal')) #6
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.3))
    model.add(Conv2D(192,(2,2), activation='relu', kernel_initializer='he_normal')) #5
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.4))
    model.add(Conv2D(192,(2,2), activation='relu', kernel_initializer='he_normal')) #4
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.4))
    model.add(Conv2D(192,(2,2), strides=(2, 2), activation='relu', kernel_initializer='he_normal')) #2
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.4))
    model.add(Conv2D(256,(2,2), activation='relu', kernel_initializer='he_normal')) #1
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Conv2D(256,(1,1), activation='relu', kernel_initializer='he_normal')) #1
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(0.5))
    model.add(Conv2D(2,(1,1), activation='relu', kernel_initializer='he_normal'))#1
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    if class_weights is not None:
        model.compile(Adam(), loss=weighted_pixelwise_crossentropy(class_weights), metrics=['accuracy',dice])
    else:
        model.compile(Adam(), loss=generalizedDiceLoss, metrics=['accuracy',dice])
        # model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy',dice])
    print(model.summary())
    return model

def get_lstm_model(num_context_slices, BOUND_SIZE=32, channels=1, class_weights=None):
    inputs = Input((2*num_context_slices+1, BOUND_SIZE, BOUND_SIZE, channels))
    if BOUND_SIZE==66:
        num_first_layer = 32
        x = TimeDistributed(Conv2D(num_first_layer, (2,2), activation='relu', kernel_initializer='he_normal'))(inputs) # 65
        x = BatchNormalization()(x)
        x = TimeDistributed(Conv2D(num_first_layer, (2,2), activation='relu', kernel_initializer='he_normal'))(x) # 64
        x = BatchNormalization()(x)
        x = TimeDistributed(Conv2D(num_first_layer, (2,2), strides=(2,2), activation='relu', kernel_initializer='he_normal'))(x) # 32
        x = BatchNormalization()(x)
        x = TimeDistributed(Conv2D(64, (2,2), activation='relu', kernel_initializer='he_normal'))(x) #31
        x = BatchNormalization()(x)
    else:
        x = TimeDistributed(Conv2D(64, (2,2), activation='relu', kernel_initializer='he_normal'))(inputs) #31
        x = BatchNormalization()(x)
        
    x = TimeDistributed(Conv2D(64, (2,2), activation='relu', kernel_initializer='he_normal'))(x) #30
    x = BatchNormalization()(x)
    x = TimeDistributed(Conv2D(64, (2,2), strides=(2,2), activation='relu', kernel_initializer='he_normal'))(x) #15
    x = BatchNormalization()(x)
    
    x = TimeDistributed(Conv2D(128, (2,2), activation='relu', kernel_initializer='he_normal'))(x) #14
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Conv2D(128, (2,2), activation='relu', kernel_initializer='he_normal'))(x) #13
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Conv2D(128, (2,2), strides=(2,2), activation='relu', kernel_initializer='he_normal'))(x) #6
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = TimeDistributed(Conv2D(192, (2,2), activation='relu', kernel_initializer='he_normal'))(x) #5
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = TimeDistributed(Conv2D(192, (2,2), activation='relu', kernel_initializer='he_normal'))(x) #4
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = TimeDistributed(Conv2D(192, (2,2), strides=(2,2), activation='relu', kernel_initializer='he_normal'))(x) #2
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Determine whether to make this layer a ConvLSTM2D or a Conv2D layer 
    x = TimeDistributed(Conv2D(256, (2,2), activation='relu', kernel_initializer='he_normal'))(x) #1
    # x = Conv2D(256, (2,2), activation='relu', kernel_initializer='he_normal')(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    # x = Bidirectional(ConvLSTM2D(return_sequences=True, kernel_size=(2,2), kernel_initializer='he_normal', 
                                 # filters=256, activation='relu', dropout=0.5, recurrent_dropout=0.5), merge_mode='ave')(x)
    x = TimeDistributed(Conv2D(256, (1,1), activation='relu', kernel_initializer='he_normal'))(x) #1
    # x = Conv2D(256, (1,1), activation='relu', kernel_initializer='he_normal')(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)                             
    #x = Bidirectional(ConvLSTM2D(return_sequences=True, kernel_size=(1,1), kernel_initializer='he_normal',
                                 #filters=256, activation='relu', dropout=0.5, recurrent_dropout=0.5), merge_mode='ave')(x)
    # Note: need Lambda layer, otherwise Keras throws an error 
    x = Lambda(lambda x: concatenate([x[:,i] for i in range(2*num_context_slices+1)]))(x)
    x = Conv2D(2, (1,1), activation='relu', kernel_initializer='he_normal')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('softmax')(x)
    model = Model(inputs, x)
    if class_weights is not None:
        model.compile(Adam(), loss=weighted_pixelwise_crossentropy(class_weights), metrics=['accuracy',dice])
    else:
        # model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy',dice])
        model.compile(Adam(), loss=generalizedDiceLoss, metrics=['accuracy',dice])
    print(model.summary())
    return model
    
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
    
if __name__ == "__main__":    
    start = time.time()
    
    
    ## USER INPUT 
    train = True
    test = False
    loadModel = False # if we should load the model from a previously saved model to continue training 
    useSmote = False # if we have a class imbalance
    
    batch_size = 32
    BOUND_SIZE = 32
    patience = 300
    max_epochs = 300
    categorical = True # Whether we one-hot encode it 
    ## END USER INPUT 
    
    if train:
        X_train, X_test, y_train, y_test = get_data(categorical)
        mean_px = X_train.mean().astype(np.float32)
        std_px = X_train.std().astype(np.float32)
        
        def norm_input(x): return (x-mean_px)/std_px
        
        X_t = norm_input(X_train)
        X_test = norm_input(X_test)
        print(X_t.shape, y_train.shape)
        if useSmote:
            sm = SMOTE(ratio='auto')
            X_train, y_train = sm.fit_sample(np.reshape(X_t,(X_t.shape[0], np.prod(X_t.shape[1:]))), y_train[:,1])
            X_train = np.reshape(X_train, (X_train.shape[0],)+ X_t.shape[1:])
            y_train = to_categorical(y_train) 
        gen = image.ImageDataGenerator(rotation_range=30, shear_range=0.3, zoom_range=0.3,
               horizontal_flip=True, vertical_flip=True)
        batches = gen.flow(X_train, y_train, batch_size=batch_size)
        test_batches = gen.flow(X_test, y_test, batch_size=batch_size)
        
        model = get_model(BOUND_SIZE=BOUND_SIZE)
        
        checkpoint_file = os.path.join(ModelsDirectory, model_fname+'.hdf5')
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_acc', verbose=1, 
                                     save_best_only=True, save_weights_only=False)
        early = EarlyStopping(monitor='loss', mode = 'min', patience = patience) 
        callbacks = [checkpoint, early]
        
        if loadModel:
            model = load_model(checkpoint_file, custom_objects = {'dice':dice, 'generalizedDiceLoss':generalizedDiceLoss})

        model.fit_generator(batches, steps_per_epoch = int(np.ceil(len(X_train)/batch_size)), epochs=max_epochs, verbose=1,
                             validation_data=test_batches, validation_steps=int(np.ceil(len(X_test)/batch_size)), callbacks=callbacks)
    if test: 
        ## Compute the prediction accuracy on withheld test set 
        X_test, y_test = get_test_data(categorical)
        
        mean_px = X_test.mean().astype(np.float32)
        std_px = X_test.std().astype(np.float32)
        def norm_input(x): return (x-mean_px)/std_px
        X_test = norm_input(X_test) 
        
        model = get_model(BOUND_SIZE=BOUND_SIZE)
        model.load_weights(os.path.join(ModelsDirectory,model_fname)+r'.hdf5')
        y_pred = model.predict(X_test, verbose=1)
        y_pred = y_pred > 0.5 # binary mask the vector 
        # compute the accuracy, dice 
        fone = npdice(y_test, y_pred, categorical) 
        acc = compute_accuracy(y_test, y_pred, categorical) 
        print("Test DICE Coefficient:", fone)
        print("Test accuracy:", acc) 
        
        
    ## Do the same for the training set
    X_train, X_val, y_train, y_val = get_data(categorical)
    X_train = np.concatenate((X_train, X_val), axis=0)
    X_train = norm_input(X_train)
    y_train = np.concatenate((y_train, y_val), axis=0)   
    mean_px = X_train.mean().astype(np.float32)
    std_px = X_train.std().astype(np.float32)
    def norm_input(x): return (x-mean_px)/std_px
    y_pred = model.predict(X_train, verbose=1)
    y_pred = y_pred > 0.5 # binary mask the vector 
    fone = npdice(y_train, y_pred, categorical) 
    acc = compute_accuracy(y_train, y_pred, categorical) 
    print("Train DICE Coefficient:", fone)
    print("Train accuracy:", acc) 
        
    print("Elapsed time:",(time.time()-start)/60)
