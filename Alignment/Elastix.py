import os
import tempfile
import shutil
import numpy
import re

import clarity.Settings as settings
import clarity.IO as io

ElastixBinary = None
ElastixLib = None
TransformixBinary = None
Initialized = False
    
def printSettings():
    global ElastixBinary, ElastixLib, TransformixBinary, Initialized
    
    if Initialized:
        print( "ElastixBinary     = %s" % ElastixBinary )
        print( "ElastixLib        = %s" % ElastixLib )
        print( "TransformixBinary = %s" % TransformixBinary )
    else:
        print( "Elastix not initialized" )


def setElastixLibraryPath(path = None): 
    if path is None:
        path = settings.ElastixPath
    
    if 'LD_LIBRARY_PATH' in os.environ:
        lp = os.environ['LD_LIBRARY_PATH']
        if not path in lp.split(':'):
            os.environ['LD_LIBRARY_PATH'] = lp + ':' + path
    else:
        os.environ['LD_LIBRARY_PATH'] = path


def initializeElastix(path = None):
    global ElastixBinary, ElastixLib, TransformixBinary, Initialized
    
    if path is None:
        path = settings.ElastixPath
    
    #search for elastix binary
    elastixbin = os.path.join(path, 'elastix.exe')
    if os.path.exists(elastixbin):
        ElastixBinary = elastixbin
    else:
        raise RuntimeError("Cannot find elastix binary %s, set path in Settings.py accordingly!" % elastixbin)
    
    #search for transformix binarx
    transformixbin = os.path.join(path, 'transformix.exe')
    if os.path.exists(transformixbin):
        TransformixBinary = transformixbin
    else:
        raise RuntimeError("Cannot find transformix binary %s set path in Settings.py accordingly!" % transformixbin)
    
    #search for elastix libs
    elastixlib = os.path.join(path, 'ANNlib.dll')
    if os.path.exists(elastixlib):
        ElastixLib = elastixlib
    else:
        raise RuntimeError("Cannot find elastix libs in %s  set path in Settings.py accordingly!" % elastixlib)
    
    #set path
    setElastixLibraryPath(elastixlib)
        
    Initialized = True
    
    
    #print( "Elastix sucessfully initialized from path: %s" % path" )
    
    return path



initializeElastix()


def checkElastixInitialized():
    global Initialized
    
    if not Initialized:
        raise RuntimeError("Elastix not initialized: run initializeElastix(path) with proper path to elastix first")

    return True


def getTransformParameterFile(resultdir):  
    files = os.listdir(resultdir)
    files = [x for x in files if re.match('TransformParameters.\d.txt', x)]
    files.sort()
    
    if files == []:
        raise RuntimeError('Cannot find a valid transformation file in ' + resultdir)
    
    return os.path.join(resultdir, files[-1])


def setPathTransformParameterFiles(resultdir):
    files = os.listdir(resultdir)
    files = [x for x in files if re.match('TransformParameters.\d.txt', x)]
    files.sort()
    
    if files == []:
        raise RuntimeError('Cannot find a valid transformation file in ' + resultdir)
    
    rec = re.compile("\(InitialTransformParametersFileName \"(?P<parname>.*)\"\)")
    
    for f in files:
        fh, tmpfn = tempfile.mkstemp()
        ff = os.path.join(resultdir, f)
        
        with open(tmpfn, 'w') as newfile:
            with open(ff) as parfile:
                for line in parfile:
                    m = rec.match(line)
                    if m != None:
                        pn = m.group('parname')
                        if pn != 'NoInitialTransform':
                            pathn, filen = os.path.split(pn)
                            filen = os.path.join(resultdir, filen)
                            newfile.write(line.replace(pn, filen))
                        else:
                            newfile.write(line)
                    else:
                        newfile.write(line)
                            
        os.close(fh)
        os.remove(ff)
        shutil.move(tmpfn, ff)


def parseElastixOutputPoints(filename, indices = True):
    with open(filename) as f:
        lines = f.readlines()
        f.close()
    
    np = len(lines)
    
    if np == 0:
        return numpy.zeros((0,3))
    
    points = numpy.zeros((np, 3))
    k = 0
    for line in lines:
        ls = line.split()
        if indices:
            for i in range(0,3):
                points[k,i] = float(ls[i+22])
        else:
            for i in range(0,3):
                points[k,i] = float(ls[i+30])
        
        k += 1
    
    return points
          
         
def getTransformFileSizeAndSpacing(transformfile):
    resi = re.compile("\(Size (?P<size>.*)\)")
    resp = re.compile("\(Spacing (?P<spacing>.*)\)")
    
    si = None
    sp = None
    with open(transformfile) as parfile:
        for line in parfile:
            m = resi.match(line)
            if m != None:
                pn = m.group('size')
                si = pn.split()
                
            m = resp.match(line)
            if m != None:
                pn = m.group('spacing')
                sp = pn.split()
    
        parfile.close()

    si = [float(x) for x in si]
    sp = [float(x) for x in sp]
    
    return si, sp


def getResultDataFile(resultdir):
    files = os.listdir(resultdir)
    files = [x for x in files if re.match('.*.(mhd|nii)', x)]
    files.sort()
    
    if files == []:
        raise RuntimeError('Cannot find a valid result data file in ' + resultdir)
    
    return os.path.join(resultdir, files[0])


    
def setTransformFileSizeAndSpacing(transformfile, size, spacing):
    resi = re.compile("\(Size (?P<size>.*)\)")
    resp = re.compile("\(Spacing (?P<spacing>.*)\)")
    
    fh, tmpfn = tempfile.mkstemp()
    
    si = tuple([int(x) for x in size])
    spacing = tuple(spacing)
    with open(transformfile) as parfile:        
        with open(tmpfn, 'w') as newfile:
            for line in parfile:
                
                m = resi.match(line)
                if m != None:
                    newfile.write("(Size %d %d %d)\n" % si)
                else:
                    m = resp.match(line)
                    if m != None:
                        newfile.write("(Spacing %f %f %f)\n" % spacing)
                    else:
                        newfile.write(line)
            
            newfile.close()
            parfile.close()
            
    os.remove(transformfile)
    shutil.copy(tmpfn, transformfile)
        


def rescaleSizeAndSpacing(size, spacing, scale):
    si = [int(x * s) for x,s in zip(size,scale)]
    sp = [float(x)/float(s) for x,s in zip(spacing,scale)]
    
    return si, sp



##############################################################################
# Elastix Runs
##############################################################################

def alignData(fixedImage, movingImage, affineParameterFile, bSplineParameterFile = None, resultDirectory = None):
    checkElastixInitialized()
    global ElastixBinary
    
    if resultDirectory == None:
        resultDirectory = tempfile.gettempdir()
    
    if not os.path.exists(resultDirectory):
        os.mkdir(resultDirectory)
    
    
    if bSplineParameterFile is None:
        cmd = '"' + ElastixBinary + '"' + ' -threads 16 -m ' + movingImage + ' -f ' + fixedImage + ' -p ' + affineParameterFile + ' -out ' + resultDirectory
    elif affineParameterFile is None:
        cmd = '"' + ElastixBinary + '"' + ' -threads 16 -m ' + movingImage + ' -f ' + fixedImage + ' -p ' + bSplineParameterFile + ' -out ' + resultDirectory
    else:
        cmd = '"' + ElastixBinary + '"' + ' -threads 16 -m ' + movingImage + ' -f ' + fixedImage + ' -p ' + affineParameterFile + ' -p ' + bSplineParameterFile + ' -out ' + resultDirectory
        #$ELASTIX -threads 16 -m $MOVINGIMAGE -f $FIXEDIMAGE -fMask $FIXEDIMAGE_MASK -p  $AFFINEPARFILE -p $BSPLINEPARFILE -out $ELASTIX_OUTPUT_DIR
    
    res = os.system(cmd)
    
    if res != 0:
        raise RuntimeError('alignData: failed executing: ' + cmd)
    
    return resultDirectory


def transformData(source, sink = [], transformParameterFile = None, transformDirectory = None, resultDirectory = None):
    global TransformixBinary
    
    if isinstance(source, numpy.ndarray):
        imgname = os.path.join(tempfile.gettempdir(), 'elastix_input.tif')
        io.writeData(source, imgname)
    elif isinstance(source, str):
        if io.dataFileNameToType(source) == "TIF" or io.dataFileNameToType(source) == "RAW":
            imgname = source
        else:
            imgname = os.path.join(tempfile.gettempdir(), 'elastix_input.tif')
            io.convertData(source, imgname)
    else:
        raise RuntimeError('transformData: source not a string or array')

    if resultDirectory == None:
        resultdirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
    else:
        resultdirname = resultDirectory
        
    if not os.path.exists(resultdirname):
        os.makedirs(resultdirname)
        
    
    if transformParameterFile == None:
        if transformDirectory == None:
            raise RuntimeError('neither alignment directory and transformation parameter file specified!')
        transformparameterdir = transformDirectory
        transformParameterFile = getTransformParameterFile(transformparameterdir)
    else:
        transformparameterdir = os.path.split(transformParameterFile)
        transformparameterdir = transformparameterdir[0]
    
    #transform
    #make path in parameterfiles absolute
    setPathTransformParameterFiles(transformparameterdir)
   
    #transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txt
    cmd = '"' + TransformixBinary + '"' + ' -in ' + imgname + ' -out ' + resultdirname + ' -tp ' + transformParameterFile
    
    res = os.system(cmd)
    
    if res != 0:
        raise RuntimeError('transformData: failed executing: ' + cmd)
    
    
    if not isinstance(source, str):
        os.remove(imgname)

    if sink == []:
        return getResultDataFile(resultdirname)
    elif sink is None:
        resultfile = getResultDataFile(resultdirname)
        return io.readData(resultfile)
    elif isinstance(sink, str):
        resultfile = getResultDataFile(resultdirname)
        return io.convertData(resultfile, sink)
    else:
        raise RuntimeError('transformData: sink not valid!')


def deformationField(sink = [], transformParameterFile = None, transformDirectory = None, resultDirectory = None):
    global TransformixBinary
    
    if resultDirectory == None:
        resultdirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
    else:
        resultdirname = resultDirectory
        
    if not os.path.exists(resultdirname):
        os.makedirs(resultdirname)
        
    if transformParameterFile == None:
        if transformDirectory == None:
            raise RuntimeError('neither alignment directory and transformation parameter file specified!')
        transformparameterdir = transformDirectory
        transformParameterFile = getTransformParameterFile(transformparameterdir)
    else:
        transformparameterdir = os.path.split(transformParameterFile)
        transformparameterdir = transformparameterdir[0]
    
    #transform
    #make path in parameterfiles absolute
    setPathTransformParameterFiles(transformparameterdir)
   
    #transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txt
    cmd = '"' + TransformixBinary + '"' + ' -def all -out ' + resultdirname + ' -tp ' + transformParameterFile
    
    res = os.system(cmd)
    
    if res != 0:
        raise RuntimeError('deformationField: failed executing: ' + cmd)
    
    
    if sink == []:
        return getResultDataFile(resultdirname)
    elif sink is None:
        resultfile = getResultDataFile(resultdirname)
        data = io.readData(resultfile)
        if resultDirectory is None:
            shutil.rmtree(resultdirname)
        return data
    elif isinstance(sink, str):
        resultfile = getResultDataFile(resultdirname)
        data = io.convertData(resultfile, sink)
        if resultDirectory is None:
            shutil.rmtree(resultdirname)
        return data
    else:
        raise RuntimeError('deformationField: sink not valid!')


def deformationDistance(deformationField, sink = None, scale = None):
    deformationField = io.readData(deformationField)
    
    df = numpy.square(deformationField)
    if not scale is None:
        for i in range(3):
            df[:,:,:,i] = df[:,:,:,i] * (scale[i] * scale[i])
            
    return io.writeData(sink, numpy.sqrt(numpy.sum(df, axis = 3)))
    

def writePoints(filename, points, indices = True):
    points = io.readPoints(points).astype('float32')
  
    with open(filename, 'wb') as pointfile:
        if indices:
            pointfile.write(b'index\n')
        else:
            pointfile.write(b'point\n')
    
        pointfile.write(str(points.shape[0]).encode('utf-8') + b'\n')
        numpy.savetxt(pointfile, points, delimiter = ' ', newline = '\n', fmt = '%.5e')
        pointfile.close()
    
    return filename



def transformPoints(source, sink = None, transformParameterFile = None, transformDirectory = None, indices = True, resultDirectory = None, tmpFile = None):
    global TransformixBinary
    
    checkElastixInitialized()
    global ElastixSettings

    if tmpFile == None:
        tmpFile = os.path.join(tempfile.gettempdir(), 'elastix_input.txt')

    # write text file
    if isinstance(source, str):
        
        #check if we have elastix signature                 
        with open(source) as f:
            line = f.readline()
            f.close()
            
            if line[:5] == 'point' or line[:5] != 'index':
                txtfile = source
            else:                
                points = io.readPoints(source)
                #points = points[:,[1,0,2]]
                txtfile = tmpFile
                writePoints(txtfile, points)
    
    elif isinstance(source, numpy.ndarray):
        txtfile = tmpFile
        #points = source[:,[1,0,2]]
        writePoints(txtfile, source)
        
    else:
        raise RuntimeError('transformPoints: source not string or array!')
    
    
    if resultDirectory == None:
        outdirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
    else:
        outdirname = resultDirectory
        
    if not os.path.exists(outdirname):
        os.makedirs(outdirname)
        
    
    if transformParameterFile == None:
        if transformDirectory == None:
            RuntimeError('neither alignment directory and transformation parameter file specified!')
        transformparameterdir = transformDirectory
        transformparameterfile = getTransformParameterFile(transformparameterdir)
    else:
        transformparameterdir = os.path.split(transformParameterFile)
        transformparameterdir  = transformparameterdir[0]
        transformparameterfile = transformParameterFile
    
    #transform
    #make path in parameterfiles absolute
    setPathTransformParameterFiles(transformparameterdir)
    
    #run transformix   
    cmd = '"' + TransformixBinary + '"' + ' -def ' + txtfile + ' -out ' + outdirname + ' -tp ' + transformparameterfile
    res = os.system(cmd)
    
    if res != 0:
        raise RuntimeError('failed executing ' + cmd)
    
    
    #read data / file 
    if sink == []:
        return os.path.join(outdirname, 'outputpoints.txt')
    
    else:
        #read coordinates
        transpoints = parseElastixOutputPoints(os.path.join(outdirname, 'outputpoints.txt'), indices = indices)

        #correct x,y,z to y,x,z
        #transpoints = transpoints[:,[1,0,2]]
        
        #cleanup
        for f in os.listdir(outdirname):
            os.remove(os.path.join(outdirname, f))
        os.rmdir(outdirname)
        
        return io.writePoints(sink, transpoints)
