import numpy as np
import cv2, sklearn, os, time, sys  
import scipy.ndimage.filters as fi
import matplotlib.pyplot as pyplt
sys.path.append(r'D:\analysis')
import clarity.IO as io 
from clarity.CellTypeDetection.phenotypeValidation import * 
import multiprocessing as mp 
from functools import partial 
import pandas as pd 

def profileCells(centers, innerRadius, outerRadius, marker_channel_img, intensity_threshold, stddev_threshold):
    ## TODO: add in standard deviation / intensity of inside vs. intensity of outside metric 
    intensities = []; stds = [] 
    # Pad the image to deal with edge cases 
    if not outerRadius>=innerRadius:
        raise RuntimeError('Please make outer radius larger than inner radius')
    marker_channel_img = np.pad(marker_channel_img, ((outerRadius[0],outerRadius[0]),(outerRadius[1],outerRadius[1]),(outerRadius[2],outerRadius[2])), 'reflect')
    start = time.time()
    for center in centers:
        innerBox = marker_channel_img[(center[0]+outerRadius[0]-innerRadius[0]):(center[0]+outerRadius[0]+innerRadius[0]+1),
                                      (center[1]+outerRadius[1]-innerRadius[1]):(center[1]+outerRadius[1]+innerRadius[1]+1),
                                      (center[2]+outerRadius[2]-innerRadius[2]):(center[2]+outerRadius[2]+innerRadius[2]+1)]
        outerBox = marker_channel_img[center[0]:(center[0]+2*outerRadius[0]+1),
                                      center[1]:(center[1]+2*outerRadius[1]+1),
                                      center[2]:(center[2]+2*outerRadius[2]+1)]
        annulusVolume = np.prod(1+2*np.asarray(outerRadius))-np.prod(1+2*np.asarray(innerRadius))
        inner_intensity = np.mean(innerBox)
        outer_intensity = (np.sum(outerBox) - np.sum(innerBox))/ annulusVolume
        intensity = inner_intensity / outer_intensity # ratio of the mean intensities 
        intensities.append(intensity)
        
        # calculate the standard deviation ratio 
        inner_stddev = np.std(innerBox.flatten())
        # useful quantities
        z_diff = outerRadius[2] - innerRadius[2] 
        x_diff = outerRadius[0] - innerRadius[0]
        y_diff = outerRadius[1] - innerRadius[1] 
        # region in between the boxes 
        annulus = np.concatenate((outerBox[:,:,:z_diff].flatten(), outerBox[:,:,-z_diff:].flatten(),
                   outerBox[:x_diff,:,z_diff:-z_diff].flatten(), outerBox[-x_diff:,:,z_diff:-z_diff].flatten(), 
                   outerBox[x_diff:-x_diff,:y_diff,z_diff:-z_diff].flatten(), outerBox[x_diff:-x_diff,-y_diff:,z_diff:-z_diff].flatten()))
        outer_stddev = np.std(annulus)
        stddev = inner_stddev / outer_stddev 
        stds.append(stddev) 
    print("Time elapsed, in seconds:", time.time()-start) 
    # pyplt.hist(stds); pyplt.title('Std dev'); pyplt.show()
    # pyplt.hist(intensities); pyplt.title('Intensities'); pyplt.show()
    expressionVector1 = np.asarray(intensities) > intensity_threshold 
    expressionVector2 = np.asarray(stds) > stddev_threshold 
    expressionVector = expressionVector1 * expressionVector2
    
    return expressionVector
    
def compareRatios(ResultDirectory, DataFile, DataFileRange, innerRadius, outerRadius, intensity_threshold, stddev_threshold, save_cells=True):
    # datafilerange: in case we don't want to run the entire script on the entire data file 
    bdir = lambda f: os.path.join(ResultDirectory, f)
    points = io.readPoints(bdir(r'spots_filtered.npy'))
    marker_channel_img = io.readData(DataFile)
        
    expressionVector = profileCells(points, innerRadius, outerRadius, marker_channel_img, intensity_threshold, stddev_threshold)
    
    expressedPoints = points[np.argwhere(np.asarray(expressionVector))]
    expressedPoints = np.squeeze(expressedPoints, axis=1)
    #print('Number of positive cells:',expressedPoints.shape[0])
    if save_cells:
        np.save(bdir('positive_cells_YGmethod.npy'),expressedPoints)
    
    return expressedPoints 


if __name__ == '__main__':
    ResultDirectory = r'D:\analysis\results\etango\Sytotest_IBA1_.6x.6x3'
    DataFile = r'D:\analysis\data\raw\Sytotest_IBA1_.6x.6x3\IBA1.tif'
    innerRadius = (5,5,2)
    outerRadius = (12,12,4)
    intensity_threshold = 1.1
    stddev_threshold = 0
    ## Validation parameters 
    valParameter = {
        'ResultDirectory' : ResultDirectory,
        'DataFile' : DataFile,
        'num_workers' : 2,
        'ValidationRange' : [{'x':all, 'y':all,'z':all}], # List of dictionaries containing [xrange, yrange, xrange]: Add as many as you want 
        'CorrectThreshold' : 2, # In microns: if our detected center is within this distance, it is considered correct 
        'DataResolution' : np.array([0.6,0.6,3]), # microns per voxel 
        'visualizeErrors' : False # overlay original image with ground truth (without nucleus detection errors) and detected cells 
    }
    mainParameter = {'ResultDirectory': ResultDirectory,
                     'DataFile' : DataFile, 
                     
                     'innerRadius': innerRadius, 
                     'outerRadius': outerRadius, 
                     'intensity_threshold': intensity_threshold,
                     'stddev_threshold': stddev_threshold}
    # gridSearchParameter = None # if None, then just do one run through 
    gridSearchParameter = { 
        'innerRadiusRange' :  [(5,5,2)],
        'outerRadiusRange'   : [(8,8,4),(10,10,4),(12,12,5)],
        'intensityRange' : [1.1,1.2,1.3,1.4],
        'stddevRange'  : [0]
    } 
    
    '''Parameters to vary for gridsearch.'''
    if gridSearchParameter is not None: 
        listOfValues = [] 
        start = time.time()
        # Make a dataframe to write to a csv file 
        for innerRadius in gridSearchParameter['innerRadiusRange']:
            mainParameter['innerRadius'] = innerRadius
            for outerRadius in gridSearchParameter['outerRadiusRange']:
                mainParameter['outerRadius'] = outerRadius
                for intensity in gridSearchParameter['intensityRange']:
                    mainParameter['intensity_threshold'] = intensity
                    for stddev in gridSearchParameter['stddevRange']:
                        mainParameter['stddev_threshold'] = stddev
                        expressedPoints = compareRatios(**mainParameter)
                        valParameter['expressedPoints'] = expressedPoints 
                        _, F1_mod, precision, recall_mod, recall = validate(**valParameter) 
                        listOfValues.append([innerRadius, outerRadius, intensity, stddev, precision, recall_mod, F1_mod])
        df = pd.DataFrame(listOfValues)
        df.to_csv(os.path.join(ResultDirectory, 'gridSearchResults_method3.csv'))
    else:
        expressedPoints = compareRatios(**mainParameter) 
        valParameter['expressedPoints'] = expressedPoints 
        _, F1_mod, precision, recall_mod,_ = validate(**valParameter) 
     
        