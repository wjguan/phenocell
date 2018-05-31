import numpy as np
import matplotlib.pyplot as pyplt 
import sys
sys.path.append(r'D:\analysis')
import clarity.IO as io 
import cv2, sklearn, os, time
import scipy.ndimage.filters as fi
import matplotlib.pyplot as pyplt
import clarity.Visualization.Plot as plt
import multiprocessing as mp 
from functools import partial 
from clarity.CellTypeDetection.parallelProcess import kernel 
from clarity.CellTypeDetection.phenotypeValidation import validate
import pandas as pd

def profileCells(centers, searchRadius, marker_channel_img, sigma = 1.0, threshold=None,
                 outerRadius=None, intensity_threshold=None, stddev_threshold=None):
    inner_intensities = []
    ratio_intensities = []
    stds = []
    if sigma is not None: 
        weight_map = calcGaussianFilter(searchRadius, sigma)
    else:
        weight_map = 1 
    # Pad the image to deal with edge cases 
    if outerRadius is None or intensity_threshold is None:
        marker_channel_img = np.pad(marker_channel_img, ((searchRadius[0],searchRadius[0]),(searchRadius[1],searchRadius[1]),(searchRadius[2],searchRadius[2])), 'reflect')
        for center in centers:
            intensity = np.sum(weight_map * marker_channel_img[center[0]:(center[0]+2*searchRadius[0]+1),
                                                               center[1]:(center[1]+2*searchRadius[1]+1),
                                                               center[2]:(center[2]+2*searchRadius[2]+1)])
            inner_intensities.append(intensity)
    else:
        marker_channel_img = np.pad(marker_channel_img, ((outerRadius[0],outerRadius[0]),(outerRadius[1],outerRadius[1]),(outerRadius[2],outerRadius[2])), 'reflect')
        for center in centers:
            innerBox = marker_channel_img[(center[0]+outerRadius[0]-searchRadius[0]):(center[0]+outerRadius[0]+searchRadius[0]+1),
                                          (center[1]+outerRadius[1]-searchRadius[1]):(center[1]+outerRadius[1]+searchRadius[1]+1),
                                          (center[2]+outerRadius[2]-searchRadius[2]):(center[2]+outerRadius[2]+searchRadius[2]+1)]
            outerBox = marker_channel_img[center[0]:(center[0]+2*outerRadius[0]+1),
                                          center[1]:(center[1]+2*outerRadius[1]+1),
                                          center[2]:(center[2]+2*outerRadius[2]+1)]
            # First method: 
            inner_intensities.append(np.sum(weight_map*innerBox)) 
            # Second method (looking at intensity ratios)                                    
            annulusVolume = np.prod(1+2*np.asarray(outerRadius))-np.prod(1+2*np.asarray(searchRadius))
            inner_intensity = np.mean(innerBox)
            outer_intensity = (np.sum(outerBox) - np.sum(innerBox))/ annulusVolume
            ratio_intensity = inner_intensity / outer_intensity # ratio of the mean intensities 
            ratio_intensities.append(ratio_intensity)

            if stddev_threshold is not None:
                # calculate the standard deviation ratio 
                inner_stddev = np.std(innerBox.flatten())
                # useful quantities
                z_diff = outerRadius[2] - searchRadius[2] 
                x_diff = outerRadius[0] - searchRadius[0]
                y_diff = outerRadius[1] - searchRadius[1] 
                # region in between the boxes 
                annulus = np.concatenate((outerBox[:,:,:z_diff].flatten(), outerBox[:,:,-z_diff:].flatten(),
                           outerBox[:x_diff,:,z_diff:-z_diff].flatten(), outerBox[-x_diff:,:,z_diff:-z_diff].flatten(), 
                           outerBox[x_diff:-x_diff,:y_diff,z_diff:-z_diff].flatten(), outerBox[x_diff:-x_diff,-y_diff:,z_diff:-z_diff].flatten()))
                outer_stddev = np.std(annulus)
                stddev = inner_stddev / outer_stddev 
                stds.append(stddev) 
                
        if stddev_threshold is not None:
            ratioExpressionVector = (np.asarray(ratio_intensities) > intensity_threshold) * (np.asarray(stds) > stddev_threshold)
        else:
            ratioExpressionVector = np.asarray(ratio_intensities) > intensity_threshold 
        
    adaptExpressionVector = np.asarray(inner_intensities) > threshold # this is from the first method 
    if outerRadius is not None and intensity_threshold is not None:
        expressionVector = adaptExpressionVector * ratioExpressionVector # Take the intersection?    
        # expressionVector = ratioExpressionVector | adaptExpressionVector 
    else:
        expressionVector = adaptExpressionVector
    
    return expressionVector

def calcGaussianFilter(searchRadius, sigma=1.0):
    x,y,z = searchRadius 
    inp = np.zeros((2*x+1,2*y+1,2*z+1))
    inp[x,y,z] = 1
    weight_map = fi.gaussian_filter(inp,sigma)
    weight_map = weight_map / np.amax(weight_map)
    return weight_map
    
def calcThreshold(marker_channel_img, searchRadius, percent):
    ## Calculate waht the threshold should be for option 1 
    # pyplt.hist(marker_channel_img.ravel())
    # pyplt.show()
    x,y,z = searchRadius
    threshold = np.percentile(marker_channel_img, percent)*(2*x+1)*(2*y+1)*(2*z+1)
    print("Threshold:",threshold)
    return threshold 
    
def calcAdaptiveThreshold(marker_channel_img, searchRadius, percent, centers, localArea):
    ## Calculate an adaptive threshold based on local images
    # Serial version of the code 
    # localArea: (x,y,z) = histogram area for local thresholding. if just one value, assume it's just z 
    # Returns a vector of thresholds 
    start = time.time()
    xr, yr, zr = searchRadius 
    if isinstance(localArea, int):
        z = localArea 
    elif len(localArea) == 2:
        x = localArea[0]; y = localArea[0]; z = localArea[1] 
    elif len(localArea) == 3:
        x = localArea[0]; y = localArea[1]; z = localArea[2] 
        
    threshold = [] 
    for center in centers:
        if isinstance(localArea, int):
            newimg = marker_channel_img[:,:,max(0,center[2]-z):min(marker_channel_img.shape[2],center[2]+z)]
        else:
            newimg = marker_channel_img[max(0,center[0]-x):min(marker_channel_img.shape[0],center[0]+x),
                                        max(0,center[1]-y):min(marker_channel_img.shape[1],center[1]+y),
                                        max(0,center[2]-z):min(marker_channel_img.shape[2],center[2]+z)]
        threshold.append(np.percentile(newimg, percent)*(2*xr+1)*(2*yr+1)*(2*zr+1))
    print("Adaptive threshold took %f seconds" %(time.time()-start))
    return threshold 
    
    
def parallel_calcAdaptiveThreshold(marker_channel_img, searchRadius, percent, points, localArea, num_workers):
    # Parallel kernel for doing this 
    start = time.time()
    g = mp.Pool(num_workers)
    func = partial(kernel, marker_channel_img, searchRadius, percent, localArea)
    threshold = g.map(func, points)
    g.close()
    g.join()
    return threshold

def calcImageShape(DataFile):
    img_shape = np.array([0,0,0])
    r = {'x':all,'y':[0,1],'z':[0,1]}
    tempImg = io.readData(DataFile, **r)
    img_shape[0] = tempImg.shape[0]
    r = {'x':[0,1],'y':all,'z':[0,1]}
    tempImg = io.readData(DataFile, **r)
    img_shape[1] = tempImg.shape[1]
    r = {'x':[0,1],'y':[0,1],'z':all}
    tempImg = io.readData(DataFile, **r)
    img_shape[2] = tempImg.shape[2]
    return img_shape
    
    
def main(ResultDirectory, DataFile, searchRadius, percent, sigma, thresholdType, localArea, num_workers=None, 
         outerRadius=None, intensity_threshold=None, stddev_threshold=None, 
         DataFileRange=[{'x':all,'y':all,'z':all}],save_cells=True):
         
    bdir = lambda f: os.path.join(ResultDirectory, f)
    finalPoints = np.array([[0,0,0]])
    for ValRange in DataFileRange: # list of dictionaries 
        points = io.readPoints(bdir(r'spots_filtered.npy'),**ValRange)
        new_points = points.copy() # so we can modify this
        
        # account for if we load in the middle of the image:
        if ValRange['x'] is not all or ValRange['y'] is not all or ValRange['z'] is not all:
            img_shape = calcImageShape(DataFile) 
            newValRange = ValRange.copy()
            if ValRange['x'] is not all:
                # Change the range for which we load image which will be larger by the searchRadius 
                newValRange['x'] = [max(0,ValRange['x'][0]-searchRadius[0]),min(img_shape[0], ValRange['x'][1]+searchRadius[0]+1)]
                new_points[:,0] = points[:,0] - newValRange['x'][0]
            if ValRange['y'] is not all:
                newValRange['y'] = [max(0,ValRange['y'][0]-searchRadius[1]),min(img_shape[1], ValRange['y'][1]+searchRadius[1]+1)]
                new_points[:,1] = points[:,1] - newValRange['y'][0]
            if ValRange['z'] is not all:
                newValRange['z'] = [max(0,ValRange['z'][0]-searchRadius[2]),min(img_shape[2], ValRange['z'][1]+searchRadius[2]+1)]
                new_points[:,2] = points[:,2] - newValRange['z'][0]
        else:
            newValRange = ValRange.copy()
        marker_channel_img = io.readData(DataFile,**newValRange)
        if thresholdType == 'global':
            threshold=calcThreshold(marker_channel_img, searchRadius, percent)
        elif thresholdType == 'adaptive':
            # Make parallel for faster run time. 
            if num_workers is not None:
                threshold=parallel_calcAdaptiveThreshold(marker_channel_img, searchRadius, percent, new_points, localArea, num_workers) 
            else:
                threshold=calcAdaptiveThreshold(marker_channel_img, searchRadius, percent, new_points, localArea) # Serial version of code 
            
        expressionVector = profileCells(new_points, searchRadius, marker_channel_img, sigma=sigma,threshold=threshold, 
                                        outerRadius=outerRadius, intensity_threshold=intensity_threshold, stddev_threshold=stddev_threshold)
        
        expressedPoints = points[np.argwhere(np.asarray(expressionVector))]
        expressedPoints = np.squeeze(expressedPoints, axis=1)
        #print('Number of positive cells:',expressedPoints.shape[0])
        finalPoints = np.concatenate((finalPoints, expressedPoints),axis=0)
    if save_cells:
        np.save(bdir('positive_cells.npy'),finalPoints[1:,:])
            
    return finalPoints[1:,:]

     
        
        