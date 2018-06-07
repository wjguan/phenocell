import numpy as np
import scipy as sp
import multiprocessing as mp
import time, bcolz, os 
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split 
from sklearn.utils import class_weight 
from imblearn.over_sampling import SMOTE 

import clarity.IO as io
from clarity.Models.TrainModel import * 
from clarity.Models.TriPlanar import TriPlanarNet 
from clarity.Models.metrics import * 
from clarity.Models.objectives import * 
from clarity.Data.MakeData import makeTriPlanarData, makeLSTMData, makeAllConvData
from annotationgui import manualValidate 


def save_array(fname, arr): # save a bcolz array 
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def phenotype(num_test_set, initial_num_labeling, num_candidates, num_annotation_suggestions, random_fraction, max_training_examples,  
              model_file, ResultDirectory, ProcessedDirectory, X_filename, NucleusFile, SpotFile, DataFile, DataFileRange,
              BOUND_SIZE=32, modelType='allconv', batch_size=32, max_epochs=50, loadModel=False, loadImages=True):
    # Files that are saved from this:
    # X_train.bc, y_train.bc: bcolz files that are a running total of the trained samples 
   
    # Actual active learning framework: utilizes active learning (in each iteration the algorithm picks some random samples and some that are
    # the most uncertain (measured by entropy: how close is it to 0.5/0.5). This can be tuned with "random_fraction". A bounding box around the 
    # nucleus center is drawn, and the input to the convnet is a 2-channel image consisting of the cell type marker and nucleus channels. 
    
    
    ## Old test parameters that are now implemented here permanently. Code should be cleaned up so that it does not depend on these parameters.
    includeNucleusChannel = True # if true, then we feed in 2-channel (currently 3-channel, with one channel as all 0's) input
    queryType = 'random_uncertain' # choices: random, uncertain, random_uncertain
    useSMOTE = True # whether we use SMOTE to oversample the imbalanced class 
    use_class_weights = False # true if we want to use weighted cross entropy 
    categorical = True # true if we are doing softmax
    num_context_slices = 1
    num_workers = None 
    saveDirectory = os.path.join(ResultDirectory, r"y_current_annotation.npy")# Directory in which we save the currently annotated stuff
    ## End parameters 
    
    if NucleusFile is not None:
        num_channels = 2
    else:
        num_channels = 1 
    
    ## 1. Starting from scratch: load in cell type image, list of nuclei centers, and nucleus image 
    print('Loading cell centers...')
    cell_candidates = io.readPoints(SpotFile, **DataFileRange)
    if loadImages:
        print('Loading image...')
        start = time.time()
        # Serial version of loading images (could take forever) - TODO: parallel load 
        img = io.readData(DataFile, **DataFileRange) 
        if NucleusFile is not None:
            print('Loading nucleus image...')
            nucleusImg = io.readData(NucleusFile, **DataFileRange)
        else: 
            nucleusImg = None 
        print("Elapsed time to load images:",time.time()-start)
    
    ## 2. If not made, make bcolz file of the data     
    start = time.time()
    X_file = os.path.join(ProcessedDirectory, X_filename) 
    if not os.path.isfile(X_file) and not os.path.isdir(X_file): # check in case it's a bcolz  file 
        if modelType == 'allconv':
            makeAllConvData(XSink=X_file, YSink=None, img=img, negatives=cell_candidates, positives=[], 
                            BOUND_SIZE=BOUND_SIZE, nucleusImg=nucleusImg)
            if not includeNucleusChannel:
                makeAllConvData(XSink=X_file, YSink=None, img=img, negatives=cell_candidates, positives=[], 
                            BOUND_SIZE=BOUND_SIZE, nucleusImg=None)
        elif modelType == 'triplanar':
            makeTriPlanarData(XSink=X_file, YSink=None, img=img, negatives=cell_candidates, positives=[], 
                            BOUND_SIZE=BOUND_SIZE, nucleusImg=nucleusImg)
            if not includeNucleusChannel:
                makeTriPlanarData(XSink=X_file, YSink=None, img=img, negatives=cell_candidates, positives=[], 
                            BOUND_SIZE=BOUND_SIZE, nucleusImg=None)
        elif modelType == 'lstm':
            makeLSTMData(XSink=X_file, YSink=None, img=img, negatives=cell_candidates, positives=[], 
                            BOUND_SIZE=BOUND_SIZE, num_context_slices=num_context_slices, nucleusImg=nucleusImg)  
            if not includeNucleusChannel:
                makeLSTMData(XSink=X_file, YSink=None, img=img, negatives=cell_candidates, positives=[], 
                            BOUND_SIZE=BOUND_SIZE, nucleusImg=None)
        else:
            raise RuntimeError(r'Please enter a valid model type.')
    
    # Load the bcolz/numpy file of all of the sections 
    if os.path.isdir(X_file): # if bcolz directory 
        X_total = bcolz.open(X_file)[:]
    elif os.path.isfile(X_file): # if numpy file 
        X_total = np.load(X_file)
        
    print('Time elapsed to make and load data:',time.time()-start)
    
    # Shuffle if we haven't already done so 
    if not os.path.isfile(os.path.join(ResultDirectory, r'indices.npy')):
        indices = np.arange(X_total.shape[0])
        np.random.shuffle(indices)
        np.save(os.path.join(ResultDirectory, r'indices.npy'),indices) # so we know how they are shuffled 
    else:
        indices = np.load(os.path.join(ResultDirectory, r'indices.npy'))
    X_shuffled = X_total[indices]
    cells_shuffled = cell_candidates[indices] # keep track of which center corresponds to which index 
    
    # If the user wants a validation benchmark, make them label a test set 
    if num_test_set > 0:
        if not os.path.isfile(os.path.join(ResultDirectory, r'cell_centers_test.npy')) or not os.path.isdir(os.path.join(ProcessedDirectory, r'X_test.bc')):
            X_test = X_shuffled[:num_test_set]
            cells_test = cells_shuffled[:num_test_set]
            if len(X_test) > 0:
                np.save(os.path.join(ResultDirectory, r'cell_centers_test.npy'),cells_test)
                y_test = manualValidate(cells_test, img, saveDirectory, nucleusImage=nucleusImg) ## TODO: write this function 
                save_array(os.path.join(ProcessedDirectory, r'X_test.bc'), X_test) 
                save_array(os.path.join(ProcessedDirectory, r'y_test.bc'), y_test) 
            np.save(os.path.join(ResultDirectory, r'cell_centers_all.npy'),cells_shuffled)
        else:
            X_test = bcolz.open(os.path.join(ProcessedDirectory, r'X_test.bc'))[:] 
            y_test = bcolz.open(os.path.join(ProcessedDirectory, r'y_test.bc'))[:]
            cells_shuffled = np.load(os.path.join(ResultDirectory, r'cell_centers_all.npy'))
            cells_test = np.load(os.path.join(ResultDirectory, r'cell_centers_test.npy'))
            num_test_set = X_test.shape[0]
        
    # Start out with an initial training set consisting of some number of training points 
    if not os.path.isdir(os.path.join(ProcessedDirectory, r'X_train.bc')) or not os.path.isdir(os.path.join(ProcessedDirectory, r'y_train.bc')):
        X = X_shuffled[num_test_set:num_test_set+initial_num_labeling]
        X_unannotated = X_shuffled[num_test_set+initial_num_labeling:]
        cells_annotated = cells_shuffled[num_test_set:num_test_set+initial_num_labeling]
        cells_unannotated = cells_shuffled[num_test_set+initial_num_labeling:]
        if len(X) > 0:
            y = manualValidate(cells_annotated, img, saveDirectory, nucleusImage=nucleusImg)
            save_array(os.path.join(ProcessedDirectory, r'X_train.bc'), X) 
            save_array(os.path.join(ProcessedDirectory, r'y_train.bc'), y)
            save_array(os.path.join(ProcessedDirectory, r'X_unannotated.bc'), X_unannotated)
            np.save(os.path.join(ResultDirectory, r'cell_centers_annotated.npy'), cells_annotated)
            np.save(os.path.join(ResultDirectory, r'cell_centers_unannotated.npy'), cells_unannotated)
    else:
        # if there already exists a training set, then start training with those. 
        X = bcolz.open(os.path.join(ProcessedDirectory, r'X_train.bc'))[:]
        y = bcolz.open(os.path.join(ProcessedDirectory, r'y_train.bc'))[:]
        X_unannotated = bcolz.open(os.path.join(ProcessedDirectory,  r'X_unannotated.bc'))[:]
        cells_annotated = np.load(os.path.join(ResultDirectory, r'cell_centers_annotated.npy'))
        cells_unannotated = np.load(os.path.join(ResultDirectory, r'cell_centers_unannotated.npy')) 
       
    
    ## First iteration of training and then prediction 
    acc, dice, y_pred = train_and_predict(BOUND_SIZE, modelType,X,y,model_file,batch_size,max_epochs,categorical,num_channels,
                                          X_unannotated=X_unannotated,useSMOTE=useSMOTE, use_class_weights=use_class_weights, 
                                          loadModel=loadModel,callback=False, num_context_slices=num_context_slices)
    if num_test_set > 0:
        acc_test, dice_test = testPredict(BOUND_SIZE, modelType, X_test, y_test, model_file, categorical,num_channels,
                                          loadModel=True, num_context_slices=num_context_slices) # also do on test set. 
                                      
    if not os.path.isfile(os.path.join(ResultDirectory,'train_accuracies.npy')) or not os.path.isfile(os.path.join(ResultDirectory,'train_dices.npy')):    
        num_labeled_imgs = initial_num_labeling # keep track of how many labeled images we have 
        accuracies = [acc]; dices = [dice] # record the total accuracies and F1 scores
        if num_test_set > 0:
            test_accuracies = [acc_test]; test_dices = [dice_test]
            np.save(os.path.join(ResultDirectory, 'test_accuracies.npy'), test_accuracies)
            np.save(os.path.join(ResultDirectory, 'test_dices.npy'), test_dices)
        np.save(os.path.join(ResultDirectory, 'train_accuracies.npy'), accuracies)
        np.save(os.path.join(ResultDirectory, 'train_dices.npy'), dices)
    else:
        num_labeled_imgs = len(y) # the number of already labeled images 
        accuracies = np.load(os.path.join(ResultDirectory, 'train_accuracies.npy'))
        dices = np.load(os.path.join(ResultDirectory, 'train_dices.npy'))
        accuracies = accuracies.tolist(); dices = dices.tolist()
        accuracies.append(acc); dices.append(dice)
        
        if num_test_set > 0:
            test_accuracies = np.load(os.path.join(ResultDirectory, 'test_accuracies.npy'))
            test_dices = np.load(os.path.join(ResultDirectory, 'test_dices.npy'))
            test_accuracies = test_accuracies.tolist(); test_dices = test_dices.tolist()
            test_accuracies.append(acc_test); test_dices.append(dice_test)
    
    # Start loop for the rest of the active learning process 
    if max_training_examples is None:
        max_training_examples = len(cell_candidates) - num_test_set 
        print("Number of max training examples:", max_training_examples) 
    while num_labeled_imgs < max_training_examples:
        print("\n Trained on %d images"%num_labeled_imgs)
        if num_labeled_imgs + num_annotation_suggestions > max_training_examples: 
            num_annotation_suggestions = max_training_examples - num_labeled_imgs 
         
        if queryType == 'random_uncertain':
            # mix up the sampling so that we get both random and uncertainty sampling going 
            # first sample the random ones, and then take the most uncertain after these 
            num_random = int(num_annotation_suggestions*random_fraction) 
            num_uncertain = num_annotation_suggestions - num_random 
            X_new_annotation = X_shuffled[num_test_set+num_labeled_imgs:num_test_set+num_labeled_imgs+num_random]
            X_unannotated = X_shuffled[num_test_set+num_labeled_imgs+num_random:]
            X_new_annotation_uncert, X_unannotated_new, annotation_indices_uncert = annotationSuggestion(y_pred[num_random:], categorical, num_uncertain, X_unannotated)
            X_new_annotation = np.concatenate((X_new_annotation, X_new_annotation_uncert), axis=0)
            annotation_indices = np.concatenate((np.arange(num_random), annotation_indices_uncert + num_random), axis=0)
        elif queryType != 'random': # use active learning, not random querying 
            X_new_annotation, X_unannotated_new, annotation_indices = annotationSuggestion(y_pred, categorical, num_candidates, X_unannotated, num_annotation_suggestions, num_workers)
        else: # do random querying 
            X_new_annotation = X_shuffled[num_test_set+num_labeled_imgs:num_test_set+num_labeled_imgs+num_annotation_suggestions]
            X_unannotated_new = X_shuffled[num_test_set+num_labeled_imgs+num_annotation_suggestions:] 
            annotation_indices = np.arange(num_annotation_suggestions) # indices are of the unannotated samples, so it will always be the first num
        
        y_new_annotation = manualValidate(cells_unannotated[annotation_indices], img, saveDirectory, nucleusImage=nucleusImg) 
        print("Do not exit...") # Don't want the user exiting during the save process 
        X = np.concatenate((X,X_new_annotation),axis=0)
        y = np.concatenate((y,y_new_annotation),axis=0)
        cells_annotated = np.concatenate((cells_annotated, cells_unannotated[annotation_indices]), axis=0)
        cells_unannotated_new = cells_unannotated[np.delete(np.arange(cells_unannotated.shape[0]),annotation_indices)]
        # Save the current state 
        save_array(os.path.join(ProcessedDirectory, r'X_train.bc'), X) 
        save_array(os.path.join(ProcessedDirectory, r'y_train.bc'), y)
        save_array(os.path.join(ProcessedDirectory, r'X_unannotated.bc'), X_unannotated_new)
        np.save(os.path.join(ResultDirectory, r'cell_centers_annotated.npy'), cells_annotated)
        np.save(os.path.join(ResultDirectory, r'cell_centers_unannotated.npy'), cells_unannotated_new)
        print("New annotations updated. Exiting now permitted.")
        
        acc, dice, y_pred = train_and_predict(BOUND_SIZE, modelType,X,y, model_file,batch_size,max_epochs,categorical,num_channels,
                                              X_unannotated=X_unannotated_new, useSMOTE=useSMOTE, use_class_weights=use_class_weights,
                                              loadModel=True,callback=True, num_context_slices=num_context_slices)
        accuracies.append(acc); dices.append(dice) 
        
        if num_test_set > 0:
            acc_test, dice_test = testPredict(BOUND_SIZE, modelType, X_test, y_test, model_file, categorical,num_channels,
                                      loadModel=True, num_context_slices=num_context_slices)
            test_accuracies.append(acc_test); test_dices.append(dice_test)
            np.save(os.path.join(ResultDirectory, 'test_accuracies.npy'), test_accuracies)
            np.save(os.path.join(ResultDirectory, 'test_dices.npy'), test_dices)
        np.save(os.path.join(ResultDirectory, 'train_accuracies.npy'), accuracies)
        np.save(os.path.join(ResultDirectory, 'train_dices.npy'), dices)
            
        # Update state 
        num_labeled_imgs += X_new_annotation.shape[0] 
        X_unannotated = X_unannotated_new
        cells_unannotated = cells_unannotated_new 

 
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

    
def testPredict(BOUND_SIZE, model_type, X_test, y_test, ModelName, categorical, num_channels, num_context_slices=1, loadModel=True):
    ''' Makes predictions for the test set and calculates the accuracy and DICE.'''
    X_test = normalize(X_test) 
    if categorical:
        y_test = to_categorical(y_test) 
    if model_type == 'allconv':
        model = get_model(BOUND_SIZE=BOUND_SIZE,channels=num_channels)
    elif model_type == 'triplanar':
        NNParameter = {'ModelFile': ModelName,
                       'img_dim' : BOUND_SIZE,
                       'num_channels': num_channels,
                       'batch_size': 32}
        mynet = TriPlanarNet(**NNParameter)
        model = mynet.getModel() 
    elif model_type == 'lstm':
        model = get_lstm_model(num_context_slices, BOUND_SIZE=BOUND_SIZE,channels=num_channels)
    if loadModel:
        # model = load_model(ModelName, custom_objects = {'dice':dice, 'modifiedDiceLoss':modifiedDiceLoss,
                                                        # 'weighted_pixelwise_crossentropy':weighted_pixelwise_crossentropy, 'loss':loss}) 
        model.load_weights(ModelName)
    if model_type == 'allconv' or model_type == 'lstm':
        y_pred = model.predict(X_test, verbose=1)
    elif model_type == 'triplanar':
        y_pred = model.predict([X_test[:,0,:,:,:],X_test[:,1,:,:,:],X_test[:,2,:,:,:]],verbose=1)
    # np.save(os.path.join(ResultDirectory,'y_test_pred.npy'), y_pred)
    np.save('y_pred.npy', y_pred)
    y_pred = y_pred > 0.5 
    total_accuracy = compute_accuracy(y_test, y_pred, categorical)
    total_dice = npdice(y_test, y_pred, categorical) 
    print("Current Test Accuracy:", total_accuracy)
    print("Current Test DICE:", total_dice) 
    
    return total_accuracy, total_dice
    
    
def train_and_predict(BOUND_SIZE, model_type, X, y, ModelName, batch_size, max_epochs, categorical, num_channels, 
                      X_unannotated=None, num_context_slices=1, useSMOTE=True, use_class_weights=False, loadModel=True, callback=True):
    ''' Trains a model and predicts unannotated samples for next iteration of active learning.'''
    #X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.1,random_state=21)
    X_t = normalize(X)
    y_train = y
    if use_class_weights:
        class_weights = class_weight.compute_class_weight('balanced',[0,1],y_train)
    else:
        class_weights=None
    if useSMOTE:
        sm = SMOTE(ratio='auto')
        # X_train, y_train = sm.fit_sample(np.reshape(X_train,(X_train.shape[0], X_train.shape[1]*X_train.shape[2]*X_train.shape[3])), y_train)
        # X_train = np.reshape(X_train, (X_train.shape[0], X.shape[1], X.shape[2], num_channels))
        X_train, y_train = sm.fit_sample(np.reshape(X_t,(X_t.shape[0], np.prod(X_t.shape[1:]))), y_train)
        X_train = np.reshape(X_train, (X_train.shape[0],)+ X.shape[1:])
    if categorical:
        y_train = to_categorical(y_train)
        y = to_categorical(y) 
    if model_type == 'allconv':
        model = get_model(BOUND_SIZE=BOUND_SIZE, channels=num_channels, class_weights=class_weights)
    elif model_type == 'lstm':
        model = get_lstm_model(num_context_slices, BOUND_SIZE=BOUND_SIZE, channels=num_channels, class_weights=class_weights) 
    elif model_type == 'triplanar':
        NNParameter = {'ModelFile': ModelName,
                       'img_dim' : BOUND_SIZE,
                       'num_channels': num_channels,
                       'batch_size': batch_size}
        mynet = TriPlanarNet(**NNParameter)
        model = mynet.getModel(class_weights) 
        
    if loadModel: # If loadModel is True, then load an existing model 
        model = load_model(ModelName, custom_objects = {'dice':dice, 'modifiedDiceLoss':modifiedDiceLoss,'generalizedDiceLoss':generalizedDiceLoss})
                                                        # 'weighted_pixelwise_crossentropy':weighted_pixelwise_crossentropy, 'loss':loss}) 
        # model.load_weights(ModelName)
        
    callbacks = []
    if callback:
        checkpoint = ModelCheckpoint(ModelName, monitor='loss', verbose=1, 
                                     save_best_only=True, save_weights_only=False, mode='min')
        callbacks.append(checkpoint) 
        
    if model_type=='allconv': # haven't figured out how to do image data generator for triplanar yet: 
        gen = image.ImageDataGenerator(rotation_range=30, shear_range=0.3, zoom_range=0.3,
               horizontal_flip=True, vertical_flip=True)
        batches = gen.flow(X_train, y_train, batch_size=batch_size)
        #test_batches = gen.flow(X_val, y_val, batch_size=batch_size) 
    
        # model.fit_generator(batches, steps_per_epoch = int(np.ceil(len(X_train)/batch_size)), epochs=max_epochs, verbose=1,
                             # validation_data=test_batches, validation_steps=int(np.ceil(len(X_val)/batch_size)), callbacks=callbacks)
        model.fit_generator(batches, steps_per_epoch = int(np.ceil(len(X_train)/batch_size)), epochs=max_epochs, verbose=1, callbacks=callbacks)
        
    elif model_type=='lstm':
        gen = image.ImageDataGenerator3D(rotation_range=30, shear_range=0.3, zoom_range=0.3,
               horizontal_flip=True, vertical_flip=True)
        batches = gen.flow(X_train, y_train, batch_size=batch_size)
        model.fit_generator(batches, steps_per_epoch = int(np.ceil(len(X_train)/batch_size)), epochs=max_epochs, verbose=1, callbacks=callbacks)
        
    elif model_type == 'triplanar':
        X_train_xy = X_train[:,0,:,:,:]
        X_train_xz = X_train[:,1,:,:,:]
        X_train_yz = X_train[:,2,:,:,:]
        model.fit([X_train_xy, X_train_xz, X_train_yz], y_train, batch_size=batch_size, epochs=max_epochs,shuffle=True, 
                   verbose=1, callbacks=callbacks)
    
    if callbacks == []:
        model.save(ModelName) 
    ## Make predictions on the training set to see stats 
    # model = load_model(ModelName, custom_objects = {'dice':dice, 'modifiedDiceLoss':modifiedDiceLoss,
                                                    # 'weighted_pixelwise_crossentropy': weighted_pixelwise_crossentropy, 'loss':loss})
    model.load_weights(ModelName)
    
    if model_type == 'allconv' or model_type == 'lstm':
        y_pred_train = model.predict(X_t, verbose=1) 
    elif model_type == 'triplanar':
        y_pred_train = model.predict([X_t[:,0,:,:,:],X_t[:,1,:,:,:],X_t[:,2,:,:,:]], verbose=1)
    # np.save(os.path.join(ResultDirectory,'y_train_pred.npy'),y_pred_train) 
    y_pred_train = y_pred_train > 0.5 
    ## Compute statistics so we know 
    total_accuracy = compute_accuracy(y, y_pred_train, categorical)
    total_dice = npdice(y, y_pred_train, categorical) 
    print("Current Training Accuracy:", total_accuracy)
    print("Current Training DICE:", total_dice) 
    
    if X_unannotated is not None: # i.e. if we want to predict 
        X_u = normalize(X_unannotated)
        if model_type == 'allconv' or model_type == 'lstm':
            y_pred = model.predict(X_u, verbose=1) # predict unannotated y so we can make next annotation suggestion
        else:
            y_pred = model.predict([X_u[:,0,:,:,:],X_u[:,1,:,:,:],X_u[:,2,:,:,:]], verbose=1)
        return total_accuracy, total_dice, y_pred
    else:
        return total_accuracy, total_dice 

    
def annotationSuggestion(y_pred, categorical, num_candidates, X_unannotated, num_annotation_suggestions=None, num_workers=None): 
    ''' Main function for performing annotation suggestion.'''
    max_indices = makeMostUncertainSet(y_pred, num_candidates, categorical)
    X_new_annotation, annotation_indices = makeAnnotationSet(X_unannotated, max_indices, num_annotation_suggestions, num_workers)
    # Subtract the final annotation suggestions from the unannotated pool 
    indices_to_keep = np.delete(np.arange(X_unannotated.shape[0]), annotation_indices) 
    X_unannotated_new = X_unannotated[indices_to_keep]
    
    return X_new_annotation, X_unannotated_new, annotation_indices # returns annotation indices so we can also change labels
    
def computeEntropy(y_pred, categorical):
    # Given the predicted y of an array of unlabeled data points, calculate the entropys
    # Categorical = True or False: whether the binary classification is represented as one hot or binary
    epsilon = 1e-5 # so that we don't encounter any log issues 
    if not categorical:
        y_pred = np.reshape(y_pred, (y_pred.shape[0], 1))
        y_pred = np.hstack((1-y_pred,y_pred))
    return np.sum(-(epsilon+y_pred)*np.log(epsilon+y_pred), axis=1)
    
    
def makeMostUncertainSet(y_pred, num_candidates, categorical):
    ''' Computes a candidate set based on uncertainty '''
    # y_pred = array containing all of the predictions on an unannotated set of data that need to be annotated 
    # categorical = whether we use binary or one-hot 
    # num_candidates = number of most uncertain samples to label 
    entropys = computeEntropy(y_pred, categorical)
    max_indices = np.argpartition(entropys, -num_candidates)[-num_candidates:]
    #np.save(r'D:\analysis\results\activeLearningTest\all_entropies.npy',entropys) # so we can see what the entropies look like. 
    return max_indices 

def makeAnnotationSet(X_unannotated, max_uncertainty_indices, num_annotation_suggestions=None, num_workers=None):
    # X_unannotated: (num_data_points, num_zslices, rows, cols, 1) of region proposal images 
    # max_uncertainty_indices = calculated indices that are the most uncertain 
    # num_workers = number of parallel tasks to run 
    # num_annotation_suggestions: number of final annotation suggestions to select. if None, don't use similarity metric. 
    # Returns: candidates = array of annotation suggestion images, annotation_indices = indices in X_unannotated 
    candidates = X_unannotated[max_uncertainty_indices]
    if num_annotation_suggestions is None: ## If we only use uncertainty estimation as annotation suggestion 
        return candidates, max_uncertainty_indices
    elif num_annotation_suggestions >= len(max_uncertainty_indices):
        print('Number of final annotation suggestions needs to be less than the number of candidates')
        print('Proceeding without calculating representativeness metric')
        return candidates, max_uncertainty_indices
    else: ## If we want to use representativeness as a factor in annotation set 
        current_num_suggestions = 0 
        annotation_indices = [] 
        final_annotation_suggestion = [] 
        while current_suggestions < num_annotation_suggestions: # iteratively add samples to the annotation suggestion 
            max_representativeness = 0 # keep track of what the maximum representativeness is
            index = 0 # the index of the given candidate data point 
            for x in candidates: # iterate through candidate pool to see which one to add to the set 
                representativeness = 0 # keep track of current representativeness for a given addition 
                possible_suggestion = final_annotation_suggestion.copy().append(x) # append a candidate to see if it maximizes similarity 
                if num_workers is not None:
                    start = time.time()
                    representativeness = parallelComputeRep(num_workers, possible_suggestion, X_unannotated) 
                    print("Time elapsed for computing representativeness:",time.time()-start)
                else:
                    # run serial version 
                    start = time.time()
                    for image in X_unannotated:
                        representativeness += computeRepresentativeness(possible_suggestion, image)
                    print("Time elapsed for computing representativeness:",time.time()-start)
                if representativeness > max_representativeness:
                    max_representativeness = representativeness
                    best_candidate = x 
                    best_index = index 
                index += 1
            final_annotation_suggestion.append(best_candidate) 
            annotation_indices.append(max_uncertainty_indices[best_index])
            current_num_suggestions += 1 
        return np.asarray(final_annotation_suggestion), np.asarray(annotation_indices) 


# Obsolete functions used to compute the representativeness - don't use because it's too computationally intensive.         
def calcSimilarity(image1, image2):
    ''' Calculate the cosine similarity.'''
    image1 = np.flatten(image1)
    image2 = np.flatten(image2)
    return 1 - sp.spatial.distance.cosine(image1, image2) 

def computeRepresentativeness(suggestions, image):
    ''' Computes the representativeness of a given datapoint in unannotated data set.'''
    # suggestions is a list of the annotation suggestion set  
    # image is a single unannotated data point for which we compute the representativeness
    max_similarity = 0 
    for x in suggestions:
        sim = calcSimilarity(image, x)
        if sim > max_similarity:
            max_similarity = sim 
    return max_similarity 
    
def parallelComputeRep(num_workers, suggestions, images):
    # Compute the representativeness for a given set of annotation suggestions and the unannotated images in parallel. 
    g = mp.Pool(num_workers)
    func = partial(computeRepresentativeness, suggestions)
    rep_vector = g.map(func, images)
    g.close()
    g.join()
    return np.sum(rep_vector)   