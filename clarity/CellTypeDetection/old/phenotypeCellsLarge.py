import numpy as np
import matplotlib.pyplot as pyplt 
import clarity.IO as io 
import cv2, sklearn, os, time, sys, bcolz, pickle
import scipy.ndimage.filters as fi
import matplotlib.pyplot as pyplt
import clarity.Visualization.Plot as plt
import multiprocessing as mp 
from functools import partial 

def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def profileCells(centers, searchRadius, marker_channel_img, option='1', sigma = 1.0, threshold=None):
    ## TODO: add in standard deviation / intensity of inside vs. intensity of outside metric 
    # Threshold = some predetermined threshold for determining whether something is a cell? 
    #       threshold = also can be a vector of local thresholds for each given cell center 
    # SearchRadius = tuple of cube half length of pixels around the detected cneter over which to sum the intensities 
    
    # Option 1: Choose an intensity threshold over which we consider it to be a positive cell: global or adaptive threshold  
    # Option 2: Use some kind of adaptive/global histogramming technique to determine a good threshold with a weight map 
    intensities = []
    if option == '2':
        weight_map = calcGaussianFilter(searchRadius, sigma)
    else:
        weight_map = 1 
    # Pad the image to deal with edge cases 
    marker_channel_img = np.pad(marker_channel_img, ((searchRadius[0],searchRadius[0]),(searchRadius[1],searchRadius[1]),(searchRadius[2],searchRadius[2])), 'reflect')
    for center in centers:
        intensity = np.sum(weight_map * marker_channel_img[center[0]:(center[0]+2*searchRadius[0]+1),
                                      center[1]:(center[1]+2*searchRadius[1]+1),
                                      center[2]:(center[2]+2*searchRadius[2]+1)])
        intensities.append(intensity)

    if option == '1' or option == '2' and threshold is not None:
        expressionVector = np.asarray(intensities) > threshold 
    else:
        raise RuntimeError('Enter a valid option (1 or 2)')
    
    return expressionVector

def calcGaussianFilter(searchRadius, sigma=1.0):
    x,y,z = searchRadius 
    inp = np.zeros((2*x+1,2*y+1,2*z+1))
    inp[x,y,z] = 1
    weight_map = fi.gaussian_filter(inp,sigma)
    weight_map = weight_map / np.amax(weight_map)
    return weight_map
    
def calcThreshold(marker_channel_img, searchRadius, percent):
    ## for global thresholding 
    x,y,z = searchRadius
    threshold = np.percentile(marker_channel_img, percent)*(2*x+1)*(2*y+1)*(2*z+1)
    print("Global threshold:",threshold)
    return threshold 

def largeKernel(searchRadius, percent, sigma, X):
    # Kernel to run for parallel processing: for deciding adaptive threshold of large z stacks. 
    # X = (img_section, img_center) is an array of size 
    img_section, center = X
    xr, yr, zr = searchRadius 
    threshold = np.percentile(img_section, percent)*(2*xr+1)*(2*yr+1)*(2*zr+1)
    if sigma is not None:
        weight_map = calcGaussianFilter(searchRadius, sigma)
    else:
        weight_map = 1.0 
    # Pad the image if we need to 
    if (center[0]-xr<0 or center[0]+xr+1 > img_section.shape[0] or
        center[1]-yr<0 or center[1]+yr+1 > img_section.shape[1] or 
        center[2]-zr<0 or center[2]+zr+1 > img_section.shape[2]):
        img_section = np.pad(img_section, ((xr,xr),(yr,yr),(zr,zr)), 'reflect')
        intensity = np.sum(weight_map * img_section[int(center[0]):int(center[0])+2*xr+1,
                                                    int(center[1]):int(center[1])+2*yr+1,
                                                    int(center[2]):int(center[2])+2*zr+1])
    else:
        intensity = np.sum(weight_map * img_section[int(center[0])-xr:int(center[0])+xr+1,
                                                    int(center[1])-yr:int(center[1])+yr+1,
                                                    int(center[2])-zr:int(center[2])+zr+1])
    expression = intensity > threshold 
    return expression      
    
def parallelProfileCells(searchRadius, percent, sigma, num_workers, X):
    # Parallel profile cells for adaptive threshold
    # X is a list of all the adaptive threshold regions 
    g = mp.Pool(num_workers)
    func = partial(largeKernel, searchRadius, percent, sigma)
    expressionVector= g.map(func, X)
    g.close()
    g.join()
    return expressionVector

def makeListOfRegions(DataSink, localArea, marker_channel_img, points):
    ''' Makes a list of regions (tuples) around each center and serializes it using pickle.'''
    # Each tuple has the array representing the part of the image to be thresholded, and the center location 
    if isinstance(localArea, int):
        z = localArea; x = marker_channel_img.shape[0]; y = marker_channel_img.shape[1] # only consider adaptive threshold in z stack 
    elif len(localArea) == 2:
        x = localArea[0]; y = localArea[0]; z = localArea[1] 
    elif len(localArea) == 3:
        x = localArea[0]; y = localArea[1]; z = localArea[2] 
    # print("Padding image...")
    # marker_channel_img = np.pad(marker_channel_img, ((x,x),(y,y),(z,z)), 'reflect')
    print("Making list of regions...")
    itemlist = []
    for center in points[:10000]: # serial because multiprocessing can't handle these data sizes 
        x_new_0 = max(0,center[0]-x); x_new_1 = min(marker_channel_img.shape[0],center[0]+x+1)
        y_new_0 = max(0,center[1]-y); y_new_1 = min(marker_channel_img.shape[1],center[1]+y+1)
        z_new_0 = max(0,center[2]-z); z_new_1 = min(marker_channel_img.shape[2],center[2]+z+1)
        newimg = marker_channel_img[x_new_0:x_new_1, y_new_0:y_new_1, z_new_0:z_new_1]
        new_center = np.array([center[0]-x_new_0, center[1]-y_new_0, center[2]-z_new_0])
        itemlist.append((newimg, new_center))
    # Use pickle to save the data 
    print("Saving list of regions...")
    if DataSink is not None:
        with open(DataSink, 'wb') as fp:
            pickle.dump(itemlist, fp)
    return itemlist 
    
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
    
def main(ResultDirectory, DataFile, searchRadius, percent, sigma, option, thresholdType, localArea, num_workers, 
         DataFileRange=[{'x':all,'y':all,'z':all}], save_cells=True):
    # Start out by loading data 
    finalPoints = np.array([[0,0,0]])
    bdir = lambda f: os.path.join(ResultDirectory, f)
    for ValRange in DataFileRange: # process whole data set 
        print('Reading points...')
        points = io.readPoints(bdir(r'spots_filtered.npy'), **ValRange)
        
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
        if thresholdType == 'adaptive' and not os.path.isfile(bdir(r'region_list')): # if we have not yet made the file with the bounding box around each center 
            start = time.time()
            print('Reading image...')
            marker_channel_img = io.readData(DataFile, **newValRange) 
            X = makeListOfRegions(None, localArea, marker_channel_img, new_points) 
            print("Time (min) taken to read image and make list of regions:", (time.time()-start)/60)
        elif thresholdType == 'global':
            start = time.time()
            marker_channel_img = io.readData(DataFile, **newValRange) 
            threshold = calcThreshold(marker_channel_img, searchRadius, percent)
            expressionVector = profileCells(new_points, searchRadius, marker_channel_img, option=option,sigma=sigma,threshold=threshold)
            print("Time (min) taken for global threshold:", (time.time()-start)/60)
        elif thresholdType == 'adaptive':
            with open(bdir(r'region_list'), 'rb') as fp:
                X = pickle.load(fp) 
                
        start = time.time()        
        if thresholdType == 'adaptive':
            expressionVector = parallelProfileCells(searchRadius, percent, sigma, num_workers, X)  
        print("Time elapsed for adaptive cell profiling: %f minutes" %((time.time()-start)/60))
        
        expressedPoints = points[np.argwhere(np.asarray(expressionVector))]
        expressedPoints = np.squeeze(expressedPoints, axis=1)
        
        finalPoints = np.concatenate((finalPoints,expressedPoints),axis=0)
    if save_cells:
        np.save(bdir('positive_cells.npy'),finalPoints[1:,:])
    
    return finalPoints[1:,:]