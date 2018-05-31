import sys
import numpy
import scipy.ndimage.measurements
from skimage.morphology import watershed
from clarity.Analysis.Voxelization import voxelizePixel
from clarity.ImageProcessing.StackProcessing import writeSubStack
from clarity.Utils.Timer import Timer
from clarity.Utils.ParameterTools import getParameter, writeParameter
from clarity.Visualization.Plot import plotOverlayLabel
import clarity.IO as io


def detectCellShape(img, peaks, detectCellShapeParameter = None, compactWatershedParameter=0,threshold = None, save = None, verbose = False, 
                    subStack = None, out = sys.stdout, **parameter):
    """Find cell shapes as labeled image
    
    Arguments:
        img (array): image data
        peaks (array): point data of cell centers / seeds
        detectCellShape (dict):
            ============ =================== ===========================================================
            Name         Type                Descritption
            ============ =================== ===========================================================
            *threshold*  (float or None)     threshold to determine mask, pixel below this are background
                                             if None no mask is generated
            *save*       (tuple)             size of the box on which to perform the *method*
            *verbose*    (bool or int)       print / plot information about this step 
            ============ =================== ===========================================================
        verbose (bool): print progress info 
        out (object): object to write progress info to
        
    Returns:
        array: labeled image where each label indicates a cell 
    """    
    
    threshold = getParameter(detectCellShapeParameter, "threshold", threshold)
    save      = getParameter(detectCellShapeParameter, "save", save)
    verbose   = getParameter(detectCellShapeParameter, "verbose", verbose)
    
    if verbose:
        writeParameter(out = out, head = 'Cell shape detection:', threshold = threshold, save = save)
    
    # extended maxima
    timer = Timer()
    
    if threshold is None:
        imgmask = None
    else:
        imgmask = img > threshold
        
    imgpeaks = voxelizePixel(peaks, dataSize = img.shape, weights = numpy.arange(1, peaks.shape[0]+1))
    #imgpeaks = voxelizePixel(peaks, dataSize = img.shape, weights = numpy.arange(2, peaks.shape[0]+2))
    
    
    #imgws = cv2.watershed(img, imgpeaks)
    
    imgws = watershed(-img, imgpeaks, mask = imgmask)
    
    #imgws = watershed_ift(-img.astype('uint16'), imgpeaks)
    #imgws[numpy.logical_not(imgmask)] = 0
    
    if not save is None:
        '''
        Edit: WG, 8/4/17: writeSubStack won't work because we can't specify a start slice.
        '''
        writeSubStack(save, imgws.astype('int32'), subStack = subStack)
        # io.writeData(save, imgws.astype('int32'))
    
    
    if verbose > 1:
        #plotTiling(img)
        plotOverlayLabel(img * 0.01, imgws, alpha = False)
        #plotOverlayLabel(img, imgmax.astype('int64'), alpha = True)     
    
    if verbose:
        out.write(timer.elapsedTime(head = 'Cell Shape:') + '\n')
    
    return imgws


def findCellSize(imglabel, findCelSizeParameter = None, maxLabel = None, verbose = False, 
                 out = sys.stdout, **parameter):
    """Find cell size given cell shapes as labled image

    Arguments:
        imglabel (array or str): labeled image, where each cell has its own label
        findCelSizeParameter (dict):
            =========== =================== ===========================================================
            Name        Type                Descritption
            =========== =================== ===========================================================
            *maxLabel*  (int or None)       maximal label to include, if None determine automatically
            *verbose*   (bool or int)       print / plot information about this step 
            =========== =================== ===========================================================
        verbose (bool): print progress info 
        out (object): object to write progress info to
        
    Returns:
        array: measured intensities 
    """    
       
    maxLabel = getParameter(findCelSizeParameter, "maxLabel", maxLabel)
    verbose  = getParameter(findCelSizeParameter, "verbose",  verbose)
    
    if verbose:
        writeParameter(out = out, head = 'Cell size detection:', maxLabel = maxLabel)
    
    timer = Timer()
    
    if maxLabel is None:
        maxLabel = int(imglabel.max())
     
    size = scipy.ndimage.measurements.sum(numpy.ones(imglabel.shape, dtype = bool), labels = imglabel, index = numpy.arange(1, maxLabel + 1))
    
    if verbose:
        out.write(timer.elapsedTime(head = 'Cell size detection:') + '\n')
    
    return size



def findCellIntensity(img, imglabel, findCellIntensityParameter = None, maxLabel = None, method = 'sum', verbose = False, 
                      out = sys.stdout, **parameter):
    """Find integrated cell intensity given cell shapes as labled image
        
    Arguments:
        img (array or str): image data
        imglabel (array or str): labeled image, where each cell has its own label
        findCellIntensityParameter (dict):
            =========== =================== ===========================================================
            Name        Type                Descritption
            =========== =================== ===========================================================
            *maxLabel*  (int or None)       maximal label to include, if None determine automatically
            *method*    (str)               method to use for measurment: 'Sum', 'Mean', 'Max', 'Min'
            *verbose*   (bool or int)       print / plot information about this step 
            =========== =================== ===========================================================
        verbose (bool): print progress info 
        out (object): object to write progress info to
        
    Returns:
        array: measured intensities 
    """    
    
    maxLabel = getParameter(findCellIntensityParameter, "maxLabel", maxLabel)
    method   = getParameter(findCellIntensityParameter, "method", method)
    verbose  = getParameter(findCellIntensityParameter, "verbose", verbose)
    
    if verbose:
        writeParameter(out = out, head = 'Cell intensity detection:', method = method, maxLabel = maxLabel)
    
    timer = Timer()
    
    if maxLabel is None:
        maxLabel = imglabel.max()
    
    if method.lower() == 'sum':
        i = scipy.ndimage.measurements.sum(img, labels = imglabel, index = numpy.arange(1, maxLabel + 1))
    elif method.lower() == 'mean':
        i = scipy.ndimage.measurements.mean(img, labels = imglabel, index = numpy.arange(1, maxLabel + 1))
    elif method.lower() == 'max':
        i = scipy.ndimage.measurements.maximum(img, labels = imglabel, index = numpy.arange(1, maxLabel + 1))
    elif method.lower() == 'min':
        i = scipy.ndimage.measurements.minimum(img, labels = imglabel, index = numpy.arange(1, maxLabel + 1))
    else:
        raise RuntimeError('cellIntensity: unkown method %s!' % method)
    
    if verbose:
        out.write(timer.elapsedTime(head = 'Cell intensity detection:') + '\n')
    
    return i
    