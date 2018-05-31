import numpy as np
import matplotlib.pyplot as plt 
import clarity.IO as io 
import cv2, sklearn, os, time, sys 
import scipy.ndimage.filters as fi
import matplotlib.pyplot as pyplt
import clarity.Visualization.Plot as plt
import multiprocessing as mp 
from functools import partial 
from clarity.CellTypeDetection.parallelProcess import *

## TODO: pass in only a small portion fo teh image into each thread (instead of the full image) 

## Serial versions of code have not been updated 
def phenotypeCells(centers, img_shape, loadRadius, SegmentedNucleusFile, DataFile, thickness, threshold):
    intensities = [] # Records the average intensity of the region aroudn the nucleus 
    # Serial version 
    for center in centers:
        if  (center[0]-loadRadius[0]<0 or center[0]+loadRadius[0]+1>img_shape[0] or 
            center[1]-loadRadius[1]<0 or center[1]+loadRadius[1]+1>img_shape[1] or 
            center[2]-loadRadius[2]<0 or center[2]+loadRadius[2]+1>img_shape[2]):
            intensities.append(0) ## TODO: add in functionality that allows for detection on edges 
            
            # Test for the edges 
            
        else:
            cell, marker = loadImageSection(center, loadRadius, SegmentedNucleusFile, DataFile)
            weight_map = []
            for slice in range(cell.shape[2]):
                weight_map.append(find2DBorder(cell[:,:,slice], thickness))
            weight_map = np.swapaxes(np.swapaxes(np.asarray(weight_map), 0,1), 1, 2)
            intensity = np.sum(weight_map * marker) / np.sum(weight_map > 0) # Normalize by the num_pixels  
            intensities.append(intensity)
            
    # pyplt.hist(intensities)
    # pyplt.title('Intensities histogram')
    # pyplt.show()     
    
    return np.asarray(intensities) > threshold 
    
    
def newKernel(img_shape, loadRadius, SegmentedNucleusFile, DataFile, thickness, numPixels, searchAroundNucleus, overallSytoImg, overallDataImg, center):
    # Kernel for parallel calculation 
    # overallSytoImg = the  overall image file (in case the image is small enough) 
    # same with overallDataImg. If these are None, then we load. 
    # numPixels = percent of top pixels to average when calculating intensity 
    start = time.time()
    if  (center[0]-loadRadius[0]<0 or center[0]+loadRadius[0]+1>img_shape[0] or 
        center[1]-loadRadius[1]<0 or center[1]+loadRadius[1]+1>img_shape[1] or 
        center[2]-loadRadius[2]<0 or center[2]+loadRadius[2]+1>img_shape[2]):
        intensity = 0 
        
        # Test for the edges 
        
    else:
        cell, marker = loadImageSection(center, loadRadius, SegmentedNucleusFile, DataFile, overallSytoImg, overallDataImg)
        weight_map = []
        for slice in range(cell.shape[2]):
            weight_map.append(find2DBorder(cell[:,:,slice], thickness, searchAroundNucleus))
        weight_map = np.swapaxes(np.swapaxes(np.asarray(weight_map), 0,1), 1, 2)
        if numPixels == 'all':
            intensity = np.sum(weight_map * marker) / np.sum(weight_map > 0) # Normalize by the num_pixels  
        else: # take the top numPixel percentile of intensities found in that region 
            mask = weight_map * marker 
            pixelIntensities = [mask[j] for j in tuple(map(tuple, np.argwhere(weight_map > 0)))]
            cutoffIntensity = np.percentile(pixelIntensities, numPixels)  
            intensity = np.sum(mask[mask >= cutoffIntensity]) / np.sum(mask >= cutoffIntensity)
    #print('Time elapsed for one worker:',time.time()-start)
    return intensity     
    
    
def parallel_phenotypeCells(centers, img_shape, loadRadius, SegmentedNucleusFile, DataFile, thickness, threshold, numPixels, searchAroundNucleus, num_workers, sytoImg=None, markerImg=None):
    # Parallel kernel for doing this 
    start = time.time()
    g = mp.Pool(num_workers)
    func = partial(newKernel, img_shape, loadRadius, SegmentedNucleusFile, DataFile, thickness, numPixels, searchAroundNucleus, sytoImg, markerImg)
    intensities = g.map(func, centers)
    g.close()
    g.join()
    print('Time elapsed:',time.time()-start) 
    # pyplt.hist(intensities)
    # pyplt.show()
    return np.asarray(intensities) > threshold 
    
    
def calcGlobalThreshold(DataFile, percent):
    ## Calculate waht the threshold should be (global) 
    # Load the entire thing 
    marker_channel_img = io.readData(DataFile) 
    # pyplt.hist(marker_channel_img.ravel())
    # pyplt.show()
    threshold = np.percentile(marker_channel_img, percent)
    print("Threshold:",threshold)
    return threshold 
    
    
def parallel_calcAdaptiveThreshold(marker_channel_img, percent, points, localArea, num_workers):
    # Parallel kernel for doing this 
    start = time.time()
    g = mp.Pool(num_workers)
    func = partial(kernel, marker_channel_img, (0,0,0), percent, localArea)
    threshold = g.map(func, points)
    g.close()
    g.join()
    print('Time elapsed (adaptive threshold):',time.time()-start) 
    # pyplt.hist(threshold)
    # pyplt.show()
    return threshold 
    
    
def find2DBorder(image, thickness, searchAroundNucleus=False):
    # img is a binary 2D image with one object in it. of type Bool 
    # thickness is the pixel thickness of the contour around the cell 
    # Returns the weight_map of the border we wish to sum to determine cell type expression 
    
    img = image.astype('uint8')
    _, contours,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # # This gives an array that only has the contour labeled
    # arr = np.zeros_like(img)
    # arr[tuple(contours[0].T)] = 1
    # arr = arr.T

    # Contour[0] is a list of the indices of the border. arr is the mask array of the contour 
    # Now return an array of the 2D border mask 
    upsized = cv2.drawContours(img, contours, -1, 1, thickness = thickness) 

    if searchAroundNucleus: # Do we only consider region aroudn nucleus 
        weight_map = upsized - image
    else:
        weight_map = upsized 
        
    # pyplt.imshow(image, 'gray')
    # pyplt.title('original')
    # pyplt.show()
    # pyplt.imshow(weight_map, 'gray')
    # pyplt.show() 
    
    return weight_map

def loadImageSection(center, loadRadius, sytoFileName, markerFileName, sytoImg=None, markerImg=None):
    # Loads an image section around a cell center 
    # SytoFileName = location of the watershedded/segmented nuclei 
    # Returns filteredSytoFile, a mask of the cell in question, and markerFile, an image of the same 
    #   dimensions that is of the marker channel that we will search for. 
    # if sytoImg and markerImg are None, then we will load the small section. 
    
    DataFileRange = {'x': [center[0]-loadRadius[0],center[0]+loadRadius[0]+1],
                     'y': [center[1]-loadRadius[1],center[1]+loadRadius[1]+1],
                     'z': [center[2]-loadRadius[2],center[2]+loadRadius[2]+1]}
    if sytoImg is None or markerImg is None:
        sytoFile = io.readData(sytoFileName, **DataFileRange)
        markerFile = io.readData(markerFileName, **DataFileRange) 
    else:
        sytoFile = sytoImg[center[0]-loadRadius[0]:center[0]+loadRadius[0]+1,
                           center[1]-loadRadius[1]:center[1]+loadRadius[1]+1,
                           center[2]-loadRadius[2]:center[2]+loadRadius[2]+1]
        markerFile = markerImg[center[0]-loadRadius[0]:center[0]+loadRadius[0]+1,
                           center[1]-loadRadius[1]:center[1]+loadRadius[1]+1,
                           center[2]-loadRadius[2]:center[2]+loadRadius[2]+1]               
    cell_label = sytoFile[loadRadius[0], loadRadius[1], loadRadius[2]] # value of watershedded value 
    filteredSytoFile = sytoFile == cell_label # mask the image section to only include the cell 
    return filteredSytoFile, markerFile 

def main(load_full_images, SegmentedNucleusFile, DataFile, ResultDirectory, thresholdType, num_workers, img_shape, loadRadius, searchAroundNucleus, percent, localArea, thickness, numPixels):    
    # Main script for this function 
    if load_full_images:
        sytoImg = io.readData(SegmentedNucleusFile)
        markerImg = io.readData(DataFile) 
    else:
        sytoImg = None; markerImg = None 
        
        
    centers = np.load(os.path.join(ResultDirectory, r'spots_filtered.npy'))
    if thresholdType == 'global':
        threshold = calcGlobalThreshold(DataFile, percent)
    elif thresholdType == 'adaptive':
        threshold = parallel_calcAdaptiveThreshold(markerImg, percent, centers, localArea, num_workers) 
    
    #expressionVector = phenotypeCells(centers, img_shape, loadRadius, SegmentedNucleusFile, DataFile, thickness, threshold)
    expressionVector = parallel_phenotypeCells(centers, img_shape, loadRadius, SegmentedNucleusFile, DataFile, thickness, threshold, numPixels, searchAroundNucleus, 4, sytoImg, markerImg)
    # Above line not that good with a lot of workers: ideal is around 4 
    
    expressedPoints = centers[np.argwhere(np.asarray(expressionVector))]
    expressedPoints = np.squeeze(expressedPoints, axis=1)
    #print('Number of positive cells:',expressedPoints.shape[0])
    np.save(os.path.join(ResultDirectory,'positive_cells.npy'),expressedPoints)
    return expressedPoints 
