if __name__ == "__main__":
    import os, numpy, math, pandas, csv, time
    import clarity.Settings as settings
    import clarity.IO as io
    import clarity.IO.RAW as raw
    import clarity.Visualization.Plot as plt
    from clarity.ImageProcessing.SpotDetection import detectSpots
    from clarity.ImageProcessing.StackProcessing import parallelProcessStack
    from clarity.Utils.ParameterTools import joinParameter
    from clarity.ImageProcessing.FusePoints import fusePoints

    ## USER INPUTS 
    # Note: the "Spot detection" must be uncommented to do spot detection again. 
    #       Otherwise, this script is just for using a neural network model to predict and classify cells. 
    
    InterimDirectory = r'GFAPr2_test_results\inputs'# Where the spots are stored
    DataFile      = r'GFAPr2_test_results\inputs\Syto.tif'
    
    DataFileRange = {'x' : all, 'y' : all, 'z' : all}
    DataOrientation = (1,2,3) # orientation of data. +/- reflects, while 1=x,2=y,3=z which allows for transposition. 
    shapeThresh = 500 # number of pixels in volume 
    processes = 6 # number of parallel processes to run. don't make too high 
    ## END USER INPUTS 
    
    # Setup     
    if not os.path.exists(InterimDirectory):
        os.makedirs(InterimDirectory)
    idir = lambda g: os.path.join(InterimDirectory, g)
    
    # Spot detection
    # Note that filterDoGParameter can be turned on for more accuracy but an increase in computational expense. 
    detectSpotsParameter = {
        "removeBackgroundParameter"    : {"size": (25,25)},
        #"filterDoGParameter"           : {"size": (13,13,7)},
        "findExtendedMaximaParameter"  : {"size": (25,25,7)},
        "detectCellShapeParameter"     : {"threshold": 10}
    }
    
    # If check is True, then we only run spot detection on middle slices given by size_z. 
    check = False
    if check:
        num_z = len(os.listdir(os.path.split(DataFile)[0]))
        size_z = 50
        DataFileRange['z'] = (num_z // 2 - size_z, num_z // 2 + size_z)
        detectSpotsParameter["removeBackgroundParameter"]["save"] = idir('bg\\[0-9]{4}.tif')
        detectSpotsParameter["findExtendedMaximaParameter"]["save"] = idir('exm\\[0-9]{4}.tif')
        detectSpotsParameter["detectCellShapeParameter"]["save"] = idir('shape\\[0-9]{4}.tif')
        
    spotdetect = True
    if spotdetect: 
        StackProcessingParameter = {
            "source": DataFile,
            "sink": (idir('spots.npy'), idir('spot_intensities.npy')),
            "processes": processes,
            "chunkSizeMax": 64,
            "chunkSizeMin": 32,
            "chunkOverlap": 8,
            "function": detectSpots,
            "verbose": True,
            "detectSpotsParameter": detectSpotsParameter
        }
        StackProcessingParameter = joinParameter(StackProcessingParameter, DataFileRange)
        
        parallelProcessStack(**StackProcessingParameter)
        
        points, intensities = io.readPoints(StackProcessingParameter["sink"])
        points, intensities = fusePoints(points, intensities, distance=5)
        
        
        # Filter out ones that are too small 
        cellShape = intensities[:,1]
        locs = numpy.argwhere(cellShape > shapeThresh)
        points = points[locs]
        points = numpy.reshape(points, (points.shape[0], 3))
        print("Number of spots: ", points.shape)
        intensities = intensities[locs,:]
        intensities = numpy.squeeze(intensities, axis=1)
        io.writePoints((idir('spots.npy'), idir('spot_intensities.npy')), (points, intensities))

    
    # Check cell detection on middle 50 slices 
    checkcells = False
    if checkcells:
        num_z = len(os.listdir(os.path.split(DataFile)[0]))
        size_z = 50
        checkRange = DataFileRange.copy()
        checkRange['z'] = (num_z // 2 - size_z, num_z // 2 + size_z)
        data = plt.overlayPoints(DataFile, points, pointColor = None, **checkRange)
        io.writeData(bdir('cells_check.tif'), data)
        
    
    # Check cell detection on whole data set 
    visualizeAll = False
    if visualizeAll:
        dataSizeSink = list(io.dataSize(DataFile))
        dataSizeSink = tuple(dataSizeSink) 
        
        voxelizeParameter = {
            "method" : 'Spherical',
            "size" : (2,2,2),
            "dataSize": dataSizeSink,
            "weights" : None
        }
        cells = voxelize(points,**voxelizeParameter)
        print(cells.shape)
        image = io.readData(DataFile, **DataFileRange)
        io.writeData(bdir('cells_check_full.tif'), numpy.stack((cells,image),axis=3).astype('uint16'))
    