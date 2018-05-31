import numpy as np 

## All this does is find the adaptive threshold for each point 
def kernel(marker_channel_img, searchRadius, percent, localArea, center):
    # Kernel to run for parallel processing: for deciding adaptive threshold of large z stacks. 
    xr,  yr, zr = searchRadius 
    if isinstance(localArea, int):
        z = localArea 
    elif len(localArea) == 2:
        x = localArea[0]; y = localArea[0]; z = localArea[1] 
    elif len(localArea) == 3:
        x = localArea[0]; y = localArea[1]; z = localArea[2] 
        
    if isinstance(localArea, int):
        newimg = marker_channel_img[:,:,max(0,center[2]-z):min(marker_channel_img.shape[2],center[2]+z)]
    else:
        newimg = marker_channel_img[max(0,center[0]-x):min(marker_channel_img.shape[0],center[0]+x),
                                    max(0,center[1]-y):min(marker_channel_img.shape[1],center[1]+y),
                                    max(0,center[2]-z):min(marker_channel_img.shape[2],center[2]+z)]
    return np.percentile(newimg, percent)*(2*xr+1)*(2*yr+1)*(2*zr+1)   
    

    
