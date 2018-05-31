import sys
import os
import math
import numpy

import multiprocessing  
import tempfile

import shutil
import cv2

import clarity.IO.IO as io
import clarity.IO.FileList as fl

from clarity.Utils.ProcessWriter import ProcessWriter


def fixOrientation(orientation):
    if orientation is None:
        return None
        
    #fix named representations
    if orientation == 'Left':
        orientation = (1,2,3)
    if orientation == 'Right':
        orientation = (-1,2,3)
    
    return orientation


def inverseOrientation(orientation):
    if orientation is None:
        return None
    
    n = len(orientation)
    iper = list(orientation)
    
    #permutation is defined as permuting the axes and then axis inversion
    for i in range(n):
        if orientation[i] < 0:
            iper[int(abs(orientation[i])-1)] = -(i + 1)
        else:
            iper[int(abs(orientation[i])-1)] = (i + 1)
    
    return tuple(iper)


def orientationToPermuation(orientation):
    orientation = fixOrientation(orientation)
    if orientation is None:
        return (0,1,2)
    else:
        return tuple(int(abs(i))-1 for i in orientation)


def orientResolution(resolution, orientation):
    if resolution is None:
        return None
    
    per = orientationToPermuation(orientation)
    return tuple(resolution[i] for i in per)
    

def orientResolutionInverse(resolution, orientation):
    if resolution is None:
        return None
    
    per = orientationToPermuation(inverseOrientation(orientation))
    return tuple(resolution[i] for i in per)

 
def orientDataSize(dataSize, orientation):
    return orientResolution(dataSize, orientation)
 
def orientDataSizeInverse(dataSize, orientation):
    return orientResolutionInverse(dataSize, orientation)
 
 
def resampleDataSize(dataSizeSource, dataSizeSink = None, resolutionSource = None, resolutionSink = None, orientation = None):
    orientation = fixOrientation(orientation)
    
    #determine data sizes if not specified
    if dataSizeSink is None:
        if resolutionSource is None or resolutionSink is None:
            raise RuntimeError('resampleDataSize: data size and resolutions not defined!')
        
        #orient resolution of source to resolution of sink to get sink data size
        resolutionSourceO = orientResolution(resolutionSource, orientation)
        dataSizeSourceO = orientDataSize(dataSizeSource, orientation)
        
        #calculate scaling factor
        dataSizeSink = tuple([int(math.ceil(dataSizeSourceO[i] *  resolutionSourceO[i]/resolutionSink[i])) for i in range(len(dataSizeSource))])
        
    
    if dataSizeSource is None:
        if resolutionSource is None or resolutionSink is None:
            raise RuntimeError('resampleDataSize: data size and resolutions not defined!')
        
        #orient resolution of source to resolution of sink to get sink data size
        resolutionSourceO = orientResolution(resolutionSource, orientation)
        
        #calculate source data size
        dataSizeSource = tuple([int(math.ceil(dataSizeSink[i] *  resolutionSink[i]/resolutionSourceO[i])) for i in range(len(dataSizeSink))])
        dataSizeSource = orientDataSizeInverse(dataSizeSource)
        
        
    #calculate effecive resolutions
    if resolutionSource is None:
        if resolutionSink is None:
            resolutionSource = (1,1,1)
        else:
            dataSizeSourceO = orientDataSize(dataSizeSource, orientation)
            resolutionSource = tuple(float(dataSizeSink[i]) / dataSizeSourceO[i] * resolutionSink[i] for i in range(len(dataSizeSource)))
            resolutionSource = orientResolutionInverse(resolutionSource, orientation)
    
    dataSizeSourceO = orientDataSize(dataSizeSource, orientation)
    
    
    resolutionSourceO = orientResolution(resolutionSource, orientation)
    resolutionSink = tuple(float(dataSizeSourceO[i]) / float(dataSizeSink[i]) * resolutionSourceO[i] for i in range(len(dataSizeSource)))
    
    
    return dataSizeSource, dataSizeSink, resolutionSource, resolutionSink  




def fixInterpolation(interpolation):
    
    if interpolation == 'nn' or interpolation is None or interpolation == cv2.INTER_NEAREST:
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
        
    return interpolation
        


def resampleXY(source, dataSizeSink, sink = None, interpolation = 'linear', out = sys.stdout, verbose = True):
    data = io.readData(source)
    dataSize = data.shape
    
    
    if data.ndim != 2:
        raise RuntimeError('resampleXY: expects 2d image source, found %dd' % data.ndim)
    
    #dataSizeSink = tuple([int(math.ceil(dataSize[i] *  resolutionSource[i]/resolutionSink[i])) for i in range(2)])
    if verbose:
        out.write(("resampleData: Imagesize: %d, %d " % (dataSize[0], dataSize[1])) + ("Resampled Imagesize: %d, %d" % (dataSizeSink[0], dataSizeSink[1])))
        #out.write(("resampleData: Imagesize: %d, %d " % dataSize) + ("Resampled Imagesize: %d, %d" % (outputSize[1], outputSize[0])))
    
    # note: cv2.resize reverses x-Y axes
    interpolation = fixInterpolation(interpolation)
    sinkData = cv2.resize(data,  (dataSizeSink[1], dataSizeSink[0]), interpolation = interpolation)
    #sinkData = cv2.resize(data,  outputSize)
    #sinkData = scipy.misc.imresize(sagittalImage, outputImageSize, interp = 'bilinear'); #normalizes images -> not usefull for stacks !
    
    #out.write("resampleData: resized Image size: %d, %d " % sinkData.shape)
    
    return io.writeData(sink, sinkData)


def _resampleXYParallel(arg):
    fileSource = arg[0]
    fileSink = arg[1]
    dataSizeSink = arg[2]
    interpolation = arg[3]
    ii = arg[4]
    nn = arg[5]
    verbose = arg[6]
    
    pw = ProcessWriter(ii)
    if verbose:
        pw.write("resampleData: resampling in XY: image %d / %d" % (ii, nn))
    
    data = numpy.squeeze(io.readData(fileSource, z = ii))
    resampleXY(data, sink = fileSink, dataSizeSink = dataSizeSink, interpolation = interpolation, out = pw, verbose = verbose)


def reorientData(source, sink = None,  orientation = None, 
                 processes = 1, verbose = True, **args):
    orientation = fixOrientation(orientation)
    

    #orient actual resolutions onto reference resolution    
    dataSizeSource = io.dataSize(source)
    
        
    nZ = dataSizeSource[2]
    data = io.readData(source, z = 0)
    resampledData = numpy.zeros(dataSizeSource, dtype = data.dtype)
    for i in range(nZ):
        if verbose and i % 10 == 0:
            print( "resampleData: reading %d/%d" % (i, nZ))
        resampledData[:,:, i] = numpy.squeeze(io.readData(source, z = i))


    #account for using (z,y,x) array representation -> (y,x,z)
    #resampledData = resampledData.transpose([1,2,0])
    #resampledData = resampledData.transpose([2,1,0])
    
    if not orientation is None:
        
        #reorient
        per = orientationToPermuation(orientation)
        resampledData = resampledData.transpose(per)
    
        #reverse orientation after permuting e.g. (-2,1) brings axis 2 to first axis and we can reorder there
        if orientation[0] < 0:
            resampledData = resampledData[::-1, :, :]
        if orientation[1] < 0:
            resampledData = resampledData[:, ::-1, :]
        if orientation[2] < 0:
            resampledData = resampledData[:, :, ::-1]
        
        #bring back from y,x,z to z,y,x
        #resampledImage = resampledImage.transpose([2,0,1])
    if verbose:
        print("resampleData: resampled data size: " + str(resampledData.shape))
    
    if sink == []:
        if io.isFileExpression(source):
            sink = os.path.split(source)
            sink = os.path.join(sink[0], 'resample_\d{4}.tif')
        elif isinstance(source, str):
            sink = source + '_resample.tif'
        else:
            raise RuntimeError('resampleData: automatic sink naming not supported for non string source!')
    
    return io.writeData(sink, resampledData)
    
        


def resampleData(source, sink = None,  orientation = None, dataSizeSink = None, resolutionSource = (4.0625, 4.0625, 3), resolutionSink = (25, 25, 25), 
                 processingDirectory = None, processes = 1, cleanup = True, verbose = True, interpolation = 'linear', **args):
    orientation = fixOrientation(orientation)
    
    if isinstance(dataSizeSink, str):
        dataSizeSink = io.dataSize(dataSizeSink)

    #orient actual resolutions onto reference resolution    
    dataSizeSource = io.dataSize(source)
        
    dataSizeSource, dataSizeSink, resolutionSource, resolutionSink = resampleDataSize(dataSizeSource = dataSizeSource, dataSizeSink = dataSizeSink, 
                                                                                      resolutionSource = resolutionSource, resolutionSink = resolutionSink, orientation = orientation)
    
    dataSizeSinkI = orientDataSizeInverse(dataSizeSink, orientation)
    
    
     
    #rescale in x y in parallel
    if processingDirectory == None:
        processingDirectory = tempfile.mkdtemp()
        
    interpolation = fixInterpolation(interpolation)
     
    nZ = dataSizeSource[2]
    pool = multiprocessing.Pool(processes=processes)
    argdata = []
    for i in range(nZ):
        argdata.append( (source, os.path.join(processingDirectory, 'resample_%04d.tif' % i), dataSizeSinkI, interpolation, i, nZ, verbose) )
    pool.map(_resampleXYParallel, argdata)
    
    #rescale in z
    fn = os.path.join(processingDirectory, 'resample_%04d.tif' % 0)
    data = io.readData(fn)
    zImage = numpy.zeros((dataSizeSinkI[0], dataSizeSinkI[1], nZ), dtype = data.dtype)
    for i in range(nZ):
        if verbose and i % 10 == 0:
            print( "resampleData: reading %d/%d" % (i, nZ))
        fn = os.path.join(processingDirectory, 'resample_%04d.tif' % i)
        zImage[:,:, i] = io.readData(fn)

    
    resampledData = numpy.zeros(dataSizeSinkI, dtype = zImage.dtype)

    for i in range(dataSizeSinkI[0]):
        if verbose and i % 25 == 0:
            print("resampleData: processing %d/%d" % (i, dataSizeSinkI[0]))
        #resampledImage[:, iImage ,:] =  scipy.misc.imresize(zImage[:,iImage,:], [resizedZAxisSize, sagittalImageSize[1]] , interp = 'bilinear')
        #cv2.resize takes reverse order of sizes !
        resampledData[i ,:, :] =  cv2.resize(zImage[i,:,:], (dataSizeSinkI[2], dataSizeSinkI[1]), interpolation = interpolation)
        #resampledData[i ,:, :] =  cv2.resize(zImage[i,:, :], (dataSize[1], resizedZSize))
    

    #account for using (z,y,x) array representation -> (y,x,z)
    #resampledData = resampledData.transpose([1,2,0])
    #resampledData = resampledData.transpose([2,1,0])
    
    if cleanup:
        shutil.rmtree(processingDirectory)

    if not orientation is None:
        
        #reorient
        per = orientationToPermuation(orientation)
        resampledData = resampledData.transpose(per)
    
        #reverse orientation after permuting e.g. (-2,1) brings axis 2 to first axis and we can reorder there
        if orientation[0] < 0:
            resampledData = resampledData[::-1, :, :]
        if orientation[1] < 0:
            resampledData = resampledData[:, ::-1, :]
        if orientation[2] < 0:
            resampledData = resampledData[:, :, ::-1]
        
        #bring back from y,x,z to z,y,x
        #resampledImage = resampledImage.transpose([2,0,1])
    if verbose:
        print("resampleData: resampled data size: " + str(resampledData.shape))
    
    if sink == []:
        if io.isFileExpression(source):
            sink = os.path.split(source)
            sink = os.path.join(sink[0], 'resample_\d{4}.tif')
        elif isinstance(source, str):
            sink = source + '_resample.tif'
        else:
            raise RuntimeError('resampleData: automatic sink naming not supported for non string source!')
    
    return io.writeData(sink, resampledData)
    


def resampleDataInverse(sink, source = None, dataSizeSource = None, orientation = None, resolutionSource = (4.0625, 4.0625, 3), resolutionSink = (25, 25, 25), 
                        processingDirectory = None, processes = 1, cleanup = True, verbose = True, interpolation = 'linear', **args):
    orientation = fixOrientation(orientation)
    
    #assume we can read data fully into memory
    resampledData = io.readData(sink)

    dataSizeSink = resampledData.shape
    
    if isinstance(dataSizeSource, str):
        dataSizeSource = io.dataSize(dataSizeSource)

    dataSizeSource, dataSizeSink, resolutionSource, resolutionSink = resampleDataSize(dataSizeSource = dataSizeSource, dataSizeSink = dataSizeSink, 
                                                                                      resolutionSource = resolutionSource, resolutionSink = resolutionSink, orientation = orientation)
    dataSizeSinkI = orientDataSizeInverse(dataSizeSink, orientation)
    
    
    #flip axes back and permute inversely
    if not orientation is None:
        if orientation[0] < 0:
            resampledData = resampledData[::-1, :, :]
        if orientation[1] < 0:
            resampledData = resampledData[:, ::-1, :]
        if orientation[2] < 0:
            resampledData = resampledData[:, :, ::-1]

        
        #reorient
        peri = inverseOrientation(orientation)
        peri = orientationToPermuation(peri)
        resampledData = resampledData.transpose(peri)
    
    # upscale in z
    interpolation = fixInterpolation(interpolation)
    
    resampledDataXY = numpy.zeros((dataSizeSinkI[0], dataSizeSinkI[1], dataSizeSource[2]), dtype = resampledData.dtype)
    fileExperssionToFileName
    for i in range(dataSizeSinkI[0]):
        if verbose and i % 25 == 0:
            print("resampleDataInverse: processing %d/%d" % (i, dataSizeSinkI[0]))

        #cv2.resize takes reverse order of sizes !
        resampledDataXY[i ,:, :] =  cv2.resize(resampledData[i,:,:], (dataSizeSource[2], dataSizeSinkI[1]), interpolation = interpolation)

    # upscale x, y in parallel
    
    if io.isFileExpression(source):
        files = source
    else:
        if processingDirectory == None:
            processingDirectory = tempfile.mkdtemp()
        files = os.path.join(sink[0], 'resample_\d{4}.tif')
    
    io.writeData(files, resampledDataXY)
    
    nZ = dataSizeSource[2]
    pool = multiprocessing.Pool(processes=processes)
    argdata = []
    for i in range(nZ):
        argdata.append( (source, fl.fileExpressionToFileName(files, i), dataSizeSource, interpolation, i, nZ) )
    pool.map(_resampleXYParallel, argdata)
    
    if io.isFileExpression(source):
        return source
    else:
        data = io.convertData(files, source)
        
        if cleanup:
            shutil.rmtree(processingDirectory)
        
        return data
    



def resamplePoints(pointSource, pointSink = None, dataSizeSource = None, dataSizeSink = None, orientation = None, resolutionSource = (4.0625, 4.0625, 3), resolutionSink = (25, 25, 25), **args):

    
    #fix (y,x,z) image array representation
    #resolutionSource, resolutionSink = self.fixResolutions(resolutionSource, resolutionSink)
    
    orientation = fixOrientation(orientation)

    #datasize of data source
    if isinstance(dataSizeSource, str):
        dataSizeSource = io.dataSize(dataSizeSource)
    
    dataSizeSource, dataSizeSink, resolutionSource, resolutionSink = resampleDataSize(dataSizeSource = dataSizeSource, dataSizeSink = dataSizeSink, 
                                                                                      resolutionSource = resolutionSource, resolutionSink = resolutionSink, orientation = orientation)

    points = io.readPoints(pointSource)

    dataSizeSinkI = orientDataSizeInverse(dataSizeSink, orientation)
    #resolutionSinkI = orientResolutionInverse(resolutionSink, orientation)
        
    #scaling factors
    scale = [float(dataSizeSource[i]) / float(dataSizeSinkI[i]) for i in range(3)]
    
    repoints = points.copy()
    for i in range(3):    
        repoints[:,i] = repoints[:,i] / scale[i]
               
    #permute for non trivial orientation
    if not orientation is None:
        per = orientationToPermuation(orientation)
        repoints = repoints[:,per]
        
        for i in range(3):
            if orientation[i] < 0:
                repoints[:,i] = dataSizeSink[i] - repoints[:,i]
      
    return io.writePoints(pointSink, repoints)


     
def resamplePointsInverse(pointSource, pointSink = None, dataSizeSource = None, dataSizeSink = None, orientation = None, resolutionSource = (4.0625, 4.0625, 3), resolutionSink = (25, 25, 25), **args):

    orientation = fixOrientation(orientation)
    
    #datasize of data source
    if isinstance(dataSizeSource, str):
        dataSizeSource = io.dataSize(dataSizeSource)
    
    dataSizeSource, dataSizeSink, resolutionSource, resolutionSink = resampleDataSize(dataSizeSource = dataSizeSource, dataSizeSink = dataSizeSink, 
                                                                                      resolutionSource = resolutionSource, resolutionSink = resolutionSink, orientation = orientation)
            
    points = io.readPoints(pointSource)
    
    dataSizeSinkI = orientDataSizeInverse(dataSizeSink, orientation)
    #resolutionSinkI = orientResolutionInverse(resolutionSink, orientation)
        
    #scaling factors
    scale = [float(dataSizeSource[i]) / float(dataSizeSinkI[i]) for i in range(3)]

    rpoints = points.copy()
    
    #invert axis inversion and permutations    
    if not orientation is None:
        #invert permuation
        iorientation = inverseOrientation(orientation)
        per = orientationToPermuation(iorientation)
        rpoints = rpoints[:,per]
        
        for i in range(3):
            if iorientation[i] < 0:
                rpoints[:,i] = dataSizeSink[i] - rpoints[:,i]
    
    #scale points
    for i in range(3):   
        rpoints[:,i] = rpoints[:,i] * scale[i]
    
    return io.writePoints(pointSink, rpoints)



def sagittalToCoronalData(source, sink = None):

    source = io.readData(source)
    d = source.ndim
    if d < 3:
        raise RuntimeError('sagittalToCoronalData: 3d image required!')
    
    tp = range(d)
    tp[0:3] = [2,0,1]
    source = source.transpose(tp)
    source = source[::-1]
    #source = source[::-1,:,:]
    return io.writeData(sink, source)


