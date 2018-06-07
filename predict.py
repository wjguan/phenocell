import numpy as np
import sys, bcolz, os
sys.path.append('phenotyping')
from clarity.Models.TrainModel import *
import clarity.IO as io 
from clarity.Models.metrics import * 
from clarity.Models.objectives import * 
from activelearning import normalize 
from clarity.Data.MakeData import makeAllConvData
from keras.models import load_model 

def main():
    ''' Predicts a set of points and then returns an expression vector of all points that are positive.'''
    ## USER INPUT
    ModelName = r'GFAPr2_test_results\models\GFAPr2.hdf5'
    NeuralNetInputFile = r'GFAPr2_test_results\processed\X_total.bc'
    SpotFile = r'GFAPr2_test_results\inputs\X_total.bc'# nuclei centers to predict
    DataFile = r'GFAPr2_test_results\inputs\GFAP.tif' # cell type marker channel
    NucleusFile = r'GFAPr2_test_results\inputs\Syto.tif'
    # NucleusFile = None
    ValidationFileRange = {'x':all,'y':all,'z':all} 
    CellsSink = r'GFAPr2_test_results\positive_cells.npy' # resultfile: where the positive cells go 
    
    
    BOUND_SIZE = 32
    ## END USER INPUT 
    
    
    points = io.readPoints(SpotFile, **ValidationFileRange, shift=True) 
    
    if not os.path.isdir(NeuralNetInputFile):
        img = io.readData(DataFile, **ValidationFileRange)
        if NucleusFile is not None:
            nucleusImg = io.readData(NucleusFile, **ValidationFileRange)
        else:
            nucleusImg = None 
        makeAllConvData(NeuralNetInputFile, None, img, points, [], BOUND_SIZE, nucleusImg=nucleusImg)
    X = bcolz.open(NeuralNetInputFile)[:]
    X = normalize(X) 
    # model = get_model(BOUND_SIZE=BOUND_SIZE)
    # model.load_weights(ModelName)
    model = load_model(ModelName, custom_objects = {'generalizedDiceLoss':generalizedDiceLoss, 'dice':dice})
    y_pred = model.predict(X, verbose=1)
    y_pred = y_pred[:,1] > 0.5 
    
    cells = points[np.argwhere(y_pred)[:,0]] # the "positive" locations 
    np.save(CellsSink, cells) 
    return 0 
    


if __name__ == '__main__':
    main()