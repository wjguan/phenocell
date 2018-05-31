import sys
import numpy


from clarity.ImageProcessing.BackgroundRemoval import removeBackground
from clarity.ImageProcessing.Filter.DoGFilter import filterDoG
from clarity.ImageProcessing.MaximaDetection import findExtendedMaxima, findPixelCoordinates, findIntensity, findCenterOfMaxima
from clarity.ImageProcessing.CellSizeDetection import detectCellShape, findCellSize, findCellIntensity

from clarity.Utils.Timer import Timer
from clarity.Utils.ParameterTools import getParameter

def detectSpots(img, detectSpotsParameter = None, removeBackgroundParameter = None,
                filterDoGParameter = None, findExtendedMaximaParameter = None, detectCellShapeParameter = None, compactWatershedParameter = 0,
                verbose = False, out = sys.stdout, **parameter):
    """Detect Cells in 3d grayscale image using DoG filtering and maxima detection
    
    Effectively this function performs the following steps:
        * illumination correction via :func:`~clarity.ImageProcessing.IlluminationCorrection.correctIllumination`
        * background removal via :func:`~clarity.ImageProcessing.BackgroundRemoval.removeBackground`
        * difference of Gaussians (DoG) filter via :func:`~clarity.ImageProcessing.Filter.filterDoG`
        * maxima detection via :func:`~clarity.ImageProcessing.MaximaDetection.findExtendedMaxima`
        * cell shape detection via :func:`~clarity.ImageProcessing.CellSizeDetection.detectCellShape`
        * cell intensity and size measurements via: :func:`~clarity.ImageProcessing.CellSizeDetection.findCellIntensity`,
          :func:`~clarity.ImageProcessing.CellSizeDetection.findCellSize`. 
    detectCells
    Note: 
        Processing steps are done in place to save memory.
        
    Arguments:
        img (array): image data
        detectSpotParameter: image processing parameter as described in the individual sub-routines
        verbose (bool): print progress information
        out (object): object to print progress information to
        
    Returns:
        tuple: tuple of arrays (cell coordinates, raw intensity, fully filtered intensty, illumination and background corrected intensity [, cell size])
    """

    timer = Timer()
    
    removeBackgroundParameter = getParameter(detectSpotsParameter, "removeBackgroundParameter", removeBackgroundParameter)
    img = removeBackground(img, removeBackgroundParameter = removeBackgroundParameter, verbose = verbose, out = out, **parameter)   
    
    filterDoGParameter = getParameter(detectSpotsParameter, "filterDoGParameter", filterDoGParameter)
    dogSize = getParameter(filterDoGParameter, "size", None)
    if not dogSize is None:
        img = filterDoG(img, filterDoGParameter = filterDoGParameter, verbose = verbose, out = out, **parameter)
    
    findExtendedMaximaParameter = getParameter(detectSpotsParameter, "findExtendedMaximaParameter", findExtendedMaximaParameter)
    hMax = getParameter(findExtendedMaximaParameter, "hMax", None)
   
    
    imgmax = findExtendedMaxima(img, findExtendedMaximaParameter = findExtendedMaximaParameter, verbose = verbose, out = out, **parameter)
    if not hMax is None:
        centers = findCenterOfMaxima(img, imgmax, verbose = verbose, out = out, **parameter)
    else:
        centers = findPixelCoordinates(imgmax, verbose = verbose, out = out, **parameter)
    del imgmax

    detectCellShapeParameter = getParameter(detectSpotsParameter, "detectCellShapeParameter", detectCellShapeParameter)
    cellShapeThreshold = getParameter(detectCellShapeParameter, "threshold", None)
    if not cellShapeThreshold is None:        
        imgshape = detectCellShape(img, centers, detectCellShapeParameter = detectCellShapeParameter, compactWatershedParameter = compactWatershedParameter, verbose = verbose, out = out, **parameter)        
        csize = findCellSize(imgshape, maxLabel = centers.shape[0], out = out, **parameter)        
        cintensity = findCellIntensity(img, imgshape,  maxLabel = centers.shape[0], verbose = verbose, out = out, **parameter)
        idz = csize > 0
        return (centers[idz], numpy.vstack((cintensity[idz], csize[idz])).transpose())
    else:
        cintensity = findIntensity(img, centers, verbose = verbose, out = out, **parameter)
        return (centers, numpy.vstack((cintensity)).transpose())
