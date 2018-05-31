## THis module / script makes data sets ready for neural network processing based on different networks. 

import bcolz, glob, os, sys
sys.path.append(r'D:\analysis')
import numpy as np
from sklearn.model_selection import train_test_split 
import clarity.IO as io

def main():
    ''' This script makes data from already pre-labeled data.'''
    ## USER INPUT 
    # What type of Data are we making? 
    typeOfData = 'allconv' # options: triplanar, lstm, and allconv 
    make_test_set = False
        ## for training old method 
    InterimDirectory = r'D:\analysis\data\interim\Syto_PV_r2_jae' # directory that contains negatives and positives
    ProcessedDirectory = r'D:\analysis\data\interim\Syto_PV_r2_jae' # directory that contains organized image sections 
    ImageDirectory = r'D:\analysis\data\raw\eTango\Syto_PV_r2\PV.tif' # where the raw image is 
        ## end for training old method 
    NucleusImageDirectory = None # if not None , then we use two channel data including nucleus channel 
    BOUND_SIZE = 32
    DataFileRange = {'x':all, 'y':all, 'z':all} # change in case just want to load a certain section 
    
    # if LSTM
    num_context_slices = 1 
    ## END USER INPUT 
    
    # Make training and withheld test set
    if make_test_set:
        if not os.path.isfile(os.path.join(InterimDirectory,'negatives_train.npy')) or not os.path.isfile(os.path.join(InterimDirectory,'positives_train.npy')):
            negatives = read_examples(glob.glob(os.path.join(InterimDirectory+r'\\negatives\\',"*.npy"))).astype('uint16')
            positives = read_examples(glob.glob(os.path.join(InterimDirectory+r'\\positives\\',"*.npy"))).astype('uint16')
            negatives = unique_rows(negatives)
            positives = unique_rows(positives)
            
            negtrain, negtest = train_test_split(negatives, test_size=0.1, random_state=42)
            np.save(InterimDirectory+r'\negatives_train.npy', negtrain); np.save(InterimDirectory+r'\negatives_test.npy', negtest)
            postrain, postest = train_test_split(positives, test_size=0.1, random_state=32)
            np.save(InterimDirectory+r'\positives_train.npy', postrain); np.save(InterimDirectory+r'\positives_test.npy', postest)
        else:
            print('Loading centers...')
            negtrain = io.readPoints(InterimDirectory+r'\negatives_train.npy',**DataFileRange)
            negtest = io.readPoints(InterimDirectory+r'\negatives_test.npy',**DataFileRange)
            postrain = io.readPoints(InterimDirectory+r'\positives_train.npy',**DataFileRange)
            postest = io.readPoints(InterimDirectory+r'\positives_test.npy',**DataFileRange)
    else:
        negatives = read_examples(glob.glob(os.path.join(InterimDirectory+r'\\negatives\\',"*.npy"))).astype('uint16')
        positives = read_examples(glob.glob(os.path.join(InterimDirectory+r'\\positives\\',"*.npy"))).astype('uint16')
        negtrain = unique_rows(negatives)
        postrain = unique_rows(positives)
    
    print('Loading image...')
    # img = np.load(InterimDirectory + r'\\image.npy')
    img = io.readData(ImageDirectory, **DataFileRange) 
    if NucleusImageDirectory is not None:
        nucleusImg = io.readData(NucleusImageDirectory, **DataFileRange)
    else: 
        nucleusImg = None 
    
    print('Loading train images...') 
    if typeOfData == 'triplanar':
        makeTriPlanarData(os.path.join(ProcessedDirectory,r'X_train.bc'), os.path.join(ProcessedDirectory,r'y_train.bc'), 
                                         img, negtrain, postrain, BOUND_SIZE, nucleusImg=nucleusImg)
    elif typeOfData == 'lstm':
        makeLSTMData(os.path.join(ProcessedDirectory,r'X_train.bc'), os.path.join(ProcessedDirectory,r'y_train.bc'), 
                                         img, negtrain, postrain, BOUND_SIZE, num_context_slices, nucleusImg=nucleusImg) 
    elif typeOfData == 'allconv':
        makeAllConvData(os.path.join(ProcessedDirectory,r'X_train.bc'), os.path.join(ProcessedDirectory,r'y_train.bc'), 
                                         img, negtrain, postrain, BOUND_SIZE, nucleusImg=nucleusImg)

    ## Do the same for the test sets 
    if make_test_set:
        print('Loading test images...') 
        if typeOfData == 'triplanar':
            makeTriPlanarData(os.path.join(ProcessedDirectory,r'X_test.bc'), os.path.join(ProcessedDirectory,r'y_test.bc'), 
                                               img, negtest, postest, BOUND_SIZE, nucleusImg=nucleusImg)
        elif typeOfData == 'lstm':
            makeLSTMData(os.path.join(ProcessedDirectory,r'X_test.bc'), os.path.join(ProcessedDirectory,r'y_test.bc'), 
                                             img, negtest, postest, BOUND_SIZE, num_context_slices, nucleusImg=nucleusImg) 
        elif typeOfData == 'allconv':
            makeAllConvData(os.path.join(ProcessedDirectory,r'X_test.bc'), os.path.join(ProcessedDirectory,r'y_test.bc'), 
                                             img, negtest, postest, BOUND_SIZE, nucleusImg=nucleusImg)

                                             
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def read_examples(files):
    examples = None
    for file in files:
        data = np.load(file)
        if data.size:
            if examples is None:
                examples = data
            else:
                examples = np.append(examples, data, axis=0)
    return examples

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def makeAllConvData(XSink, YSink, img, negatives, positives, BOUND_SIZE, nucleusImg=None):
    # In order to make unlabeled data, just input:
    # negatives = array of points; positives = []; YSink = None 
    ## TODO:  Parallelize this. 
    img_padded = np.pad(img, BOUND_SIZE//2, 'constant')
    del img 
    if nucleusImg is not None:
        nucleus_padded = np.pad(nucleusImg, BOUND_SIZE//2, 'constant')
        X = np.zeros((len(positives)+len(negatives),BOUND_SIZE,BOUND_SIZE,2),dtype='uint16')
    else:
        X = np.zeros((len(positives)+len(negatives),BOUND_SIZE,BOUND_SIZE,1),dtype='uint16')
    if YSink is not None:
        y = np.concatenate((np.zeros(len(negatives),dtype='uint16'),np.ones(len(positives),dtype='uint16')))

    for i in range(len(positives)+len(negatives)):
        if i < len(negatives):
            s = negatives[i]
        else:
            s = positives[i-len(negatives)]
        X[i,:,:,0] = img_padded[s[0]:s[0]+BOUND_SIZE, s[1]:s[1]+BOUND_SIZE, s[2]+BOUND_SIZE//2]
        if nucleusImg is not None:
            X[i,:,:,1] = nucleus_padded[s[0]:s[0]+BOUND_SIZE, s[1]:s[1]+BOUND_SIZE, s[2]+BOUND_SIZE//2]
        
    if XSink is not None:
        save_array(XSink, X)
        # np.save(XSink, X)
    if YSink is not None:
        save_array(YSink, y)
        # np.save(YSink, Y)
    
    
def makeTriPlanarData(XSink, YSink, img, negatives, positives, BOUND_SIZE, nucleusImg = None):
    img_padded = np.pad(img, BOUND_SIZE//2, 'constant')
    del img 
    if nucleusImg is not None:
        nucleus_padded = np.pad(nucleusImg, BOUND_SIZE//2, 'constant')
        X_train = np.zeros((len(positives)+len(negatives),3,BOUND_SIZE,BOUND_SIZE,2),dtype='uint16')
    else:
        X_train = np.zeros((len(positives)+len(negatives),3,BOUND_SIZE,BOUND_SIZE,1),dtype='uint16')
    if YSink is not None:
        y_train = np.concatenate((np.zeros(len(negatives),dtype='uint16'),np.ones(len(positives),dtype='uint16')))

    for i in range(len(positives)+len(negatives)):
        if i % 100 == 0:
            print('Finished processing %d train images..'%i)
        if i < len(negatives):
            s = negatives[i]
        else:
            s = positives[i-len(negatives)]
        X_train[i,0,:,:,0] = img_padded[s[0]:s[0]+BOUND_SIZE, s[1]:s[1]+BOUND_SIZE, s[2]+BOUND_SIZE//2] # xy
        X_train[i,1,:,:,0] = img_padded[s[0]:s[0]+BOUND_SIZE, s[1]+BOUND_SIZE//2, s[2]:s[2]+BOUND_SIZE] # xz
        X_train[i,2,:,:,0] = img_padded[s[0]+BOUND_SIZE//2, s[1]:s[1]+BOUND_SIZE, s[2]:s[2]+BOUND_SIZE] # yz
        
        if nucleusImg is not None:
            X_train[i,0,:,:,1] = nucleus_padded[s[0]:s[0]+BOUND_SIZE, s[1]:s[1]+BOUND_SIZE, s[2]+BOUND_SIZE//2] # xy
            X_train[i,1,:,:,1] = nucleus_padded[s[0]:s[0]+BOUND_SIZE, s[1]+BOUND_SIZE//2, s[2]:s[2]+BOUND_SIZE] # xz
            X_train[i,2,:,:,1] = nucleus_padded[s[0]+BOUND_SIZE//2, s[1]:s[1]+BOUND_SIZE, s[2]:s[2]+BOUND_SIZE] # yz
    
    if XSink is not None:
        # np.save(XSink, X_train)
        save_array(XSink, X_train)
    if YSink is not None:
        # np.save(YSink, y_train)
        save_array(YSink, y_train)
    
    # return X_train, y_train 
     
     
def makeLSTMData(XSink, YSink, img, negatives, positives, BOUND_SIZE, num_context_slices, nucleusImg=None):  
    img_padded = np.pad(img, ((BOUND_SIZE//2, BOUND_SIZE//2), (BOUND_SIZE//2, BOUND_SIZE//2), (num_context_slices, num_context_slices)), 'constant')
    del img 
    if nucleusImg is not None:
        nucleus_padded = np.pad(nucleusImg, ((BOUND_SIZE//2, BOUND_SIZE//2), (BOUND_SIZE//2, BOUND_SIZE//2), (num_context_slices, num_context_slices)), 'constant')
        X_train = np.zeros((len(positives)+len(negatives),2*num_context_slices+1,BOUND_SIZE,BOUND_SIZE,2),dtype='uint16')
    else:
        X_train = np.zeros((len(positives)+len(negatives),2*num_context_slices+1,BOUND_SIZE,BOUND_SIZE,1),dtype='uint16')
    if YSink is not None:
        y_train = np.concatenate((np.zeros(len(negatives),dtype='uint16'),np.ones(len(positives),dtype='uint16')))

    for i in range(len(positives)+len(negatives)):
        if i % 100 == 0:
            print('Finished processing %d train images..'%i)
        if i < len(negatives):
            s = negatives[i]
        else:
            s = positives[i-len(negatives)]
        X_train[i,:,:,:,0] = np.swapaxes(np.swapaxes(img_padded[s[0]:s[0]+BOUND_SIZE, s[1]:s[1]+BOUND_SIZE, s[2]:s[2]+2*num_context_slices+1],1,2),0,1)
        
        if nucleusImg is not None:
             X_train[i,:,:,:,1] = np.swapaxes(np.swapaxes(nucleus_padded[s[0]:s[0]+BOUND_SIZE, s[1]:s[1]+BOUND_SIZE, s[2]:s[2]+2*num_context_slices+1],1,2),0,1)
        
    if XSink is not None:
        # np.save(XSink, X_train)  
        save_array(XSink, X_train)
    if YSink is not None:
        # np.save(YSink, y_train) 
        save_array(YSink, y_train)
    
if __name__ == "__main__":
    main() 