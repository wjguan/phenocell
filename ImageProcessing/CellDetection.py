import clarity.ImageProcessing.SpotDetection
import clarity.ImageProcessing.IlastikClassification

from clarity.ImageProcessing.StackProcessing import parallelProcessStack, sequentiallyProcessStack

from clarity.Utils.Timer import Timer
    

def detectCells(source, sink = None, method ="SpotDetection", processMethod = all, verbose = False, **parameter):
    timer = Timer()
        
    if method == "SpotDetection":
        detectCells = clarity.ImageProcessing.SpotDetection.detectSpots
    elif method == 'Ilastik':
        if clarity.ImageProcessing.Ilastik.Initialized:
            detectCells = clarity.ImageProcessing.IlastikClassification.classifyCells
            processMethod = 'sequential';  #ilastik does parallel processing so force sequential processing here
        else:
            raise RuntimeError("detectCells: Ilastik not initialized, fix in Settings.py or use SpotDectection method instead!")
    else:
        raise RuntimeError("detectCells: invalid method %s" % str(method))
    
    if processMethod == 'sequential':
        result = sequentiallyProcessStack(source, sink = sink, function = detectCells, verbose = verbose, **parameter)
    elif processMethod is all or processMethod == 'parallel':
        result = parallelProcessStack(source, sink = sink, function = detectCells, verbose = verbose, **parameter)
    else:
        raise RuntimeError("detectCells: invalid processMethod %s" % str(processMethod))
    
    if verbose:
        timer.printElapsedTime("Total Cell Detection")
    
    return result


