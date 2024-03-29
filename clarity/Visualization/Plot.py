import math
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

import clarity.IO as io
import clarity.Analysis.Voxelization as vox


def plotTiling(dataSource, tiling = "automatic", maxtiles = 20, x = all, y = all, z = all, inverse = False): 
    """Plot 3d image as 2d tiles
    
    Arguments:
        dataSouce (str or array): volumetric image data
        tiling (str or tuple): tiling specification
        maxtiles: maximalnumber of tiles
        x, y, z (all or tuple): sub-range specification
        inverse (bool):invert image
    
    Returns:
        (object): figure handle
    """
    
    image = io.readData(dataSource, x = x, y = y, z = z)
    dim = image.ndim
    
    if dim < 2 or dim > 4:
        raise StandardError('plotTiling: image dimension must be 2 to 4')   
    
    if dim == 2:
        image = image.reshape(image.shape + (1,))
        dim = 3

    if image.ndim == 3: 
        if image.shape[2] == 3:  # 2d color image
            ntiles = 1
            cmap = None
            image = image.reshape((image.shape[0], image.shape[1], 1, 3))
        else:                   # 3d gray image
            ntiles = image.shape[2]
            cmap = plt.cm.gray
            image = image.reshape(image.shape + (1,))
    else:
        ntiles = image.shape[2]# 3d color = 4d
        cmap = None
    
    if ntiles > maxtiles:
        print("plotTiling: number of tiles %d very big! Clipping at %d!" % (ntiles, maxtiles))
        ntiles = maxtiles
    
    if tiling == "automatic":
        nx = math.floor(math.sqrt(ntiles))
        ny = int(math.ceil(ntiles / nx))
        nx = int(nx)
    else:
        nx = int(tiling[0])
        ny = int(tiling[1])
        
    fig, axarr = plt.subplots(nx, ny, sharex = True, sharey = True)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    axarr = numpy.array(axarr)
    axarr = axarr.flatten()
    
    imin = image.min()
    imax = image.max()
    
    if inverse:
        (imin, imax) = (-float(imax), -float(imin))
    
    for i in range(0, ntiles): 
        a = axarr[i]
        imgpl = image[:,:,i,:].copy()
        imgpl = imgpl.transpose([1,0,2]) 
        
        if imgpl.shape[2] == 1:
            imgpl = imgpl.reshape((imgpl.shape[0], imgpl.shape[1]))      
            
        if inverse:
            imgpl = -imgpl.astype('float')
        
        a.imshow(imgpl, interpolation='none', cmap = cmap, vmin = imin, vmax = imax)
    
    
    return fig


def overlayLabel(dataSource, labelSource, sink = None,  alpha = False, labelColorMap = 'jet', x = all, y = all, z = all):
    """Overlay a gray scale image with colored labeled image
    
    Arguments:
        dataSouce (str or array): volumetric image data
        labelSource (str or array): labeled image to be overlayed on the image data
        sink (str or None): destination for the overlayed image
        alpha (float or False): transparency
        labelColorMap (str or object): color map for the labels
        x, y, z (all or tuple): sub-range specification
    
    Returns:
        (array or str): figure handle
        
    See Also:
        :func:`overlayPoints`
    """ 
    
    label = io.readData(labelSource, x= x, y = y, z = z)
    image = io.readData(dataSource, x= x, y = y, z = z)
    
    lmax = labelSource.max()
    
    if lmax <= 1:
        carray = numpy.array([[1,0,0,1]])
    else:
        cm = mpl.cm.get_cmap(labelColorMap)
        cNorm  = mpl.colors.Normalize(vmin=1, vmax = int(lmax))
        carray = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)
        carray = carray.to_rgba(numpy.arange(1, int(lmax + 1)))

    if alpha == False:
        carray = numpy.concatenate(([[0,0,0,1]], carray), axis = 0)
    else:
        carray = numpy.concatenate(([[1,1,1,1]], carray), axis = 0)
        
    cm = mpl.colors.ListedColormap(carray)
    carray = cm(label)
    carray = carray.take([0,1,2], axis = -1)

    if alpha == False:
        cimage = (label == 0) * image
        cimage = numpy.repeat(cimage, 3)
        cimage = cimage.reshape(image.shape + (3,))
        cimage = cimage.astype(carray.dtype)
        cimage += carray
    else:
        cimage = numpy.repeat(image, 3)
        cimage = cimage.reshape(image.shape + (3,))
        cimage = cimage.astype(carray.dtype)
        cimage *= carray

    return io.writeData(sink, cimage)
    
    
def plotOverlayLabel(dataSource, labelSource, alpha = False, labelColorMap = 'jet',  x = all, y = all, z = all, tiling = "automatic", maxtiles = 20):
    """Plot gray scale image overlayed with labeled image
    
    Arguments:
        dataSouce (str or array): volumetric image data
        labelSource (str or array): labeled image to be overlayed on the image data
        alpha (float or False): transparency
        labelColorMap (str or object): color map for the labels
        x, y, z (all or tuple): sub-range specification
        tiling (str or tuple): tiling specification
        maxtiles: maximalnumber of tiles
    
    Returns:
        (object): figure handle
        
    See Also:
        :func:`overlayLabel`
    """    
    
    ov = overlayLabel(dataSource, labelSource, alpha = alpha, labelColorMap = labelColorMap, x = x, y = y, z = z)
    return plotTiling(ov, tiling = tiling, maxtiles = maxtiles)



def overlayPoints(dataSource, pointSource, sink = None, pointColor = [1,0,0], x = all, y = all, z = all):
    """Overlay points on 3D data and return as color image
    
    Arguments:
        dataSouce (str or array): volumetric image data
        pointSource (str or array): point data to be overlayed on the image data
        pointColor (array): RGB color for the overlayed points
        x, y, z (all or tuple): sub-range specification
    
    Returns:
        (str or array): image overlayed with points
        
    See Also:
        :func:`overlayLabel`
    """
    data = io.readData(dataSource, x = x, y = y, z = z)
    points = io.readPoints(pointSource, x = x, y = y, z = z, shift = True)
    
    if not pointColor is None:
        dmax = data.max()
        dmin = data.min()
        if dmin == dmax:
            dmax = dmin + 1
        cimage = numpy.repeat( (data - dmin) / (dmax - dmin), 3)
        cimage = cimage.reshape(data.shape + (3,))   
    
        if data.ndim == 2:
            for p in points: # faster version using voxelize ?
                cimage[p[0], p[1], :] = pointColor
        elif data.ndim == 3:
            for p in points: # faster version using voxelize ?
                cimage[p[0], p[1], p[2], :] = pointColor
        else:
            raise RuntimeError('overlayPoints: data dimension %d not suported' % data.ndim)
    
    else:
        ## WG edit: 1/4/18 - voxelize so it's bigger and we can visualize it 
        # cimage = vox.voxelize(points, data.shape, method = 'Pixel')
        cimage = vox.voxelize(points, data.shape, method = 'Rectangular', size=(3,3,1))
        cimage = cimage.astype(data.dtype) * data.max()
        data.shape = data.shape + (1,)
        cimage.shape =  cimage.shape + (1,)
        cimage = numpy.concatenate((data, cimage), axis  = 3)
    
    return io.writeData(sink, cimage)  


def plotOverlayPoints(dataSource, pointSource, pointColor = [1,0,0], x = all, y = all, z = all):
    """Plot points overlayed on gray scale 3d image as tiles.
    
    Arguments:
        dataSouce (str or array): volumetric image data
        pointSource (str or array): point data to be overlayed on the image data
        pointColor (array): RGB color for the overlayed points
        x, y, z (all or tuple): sub-range specification
    
    Returns:
        (object): figure handle
        
    See Also:
        :func:`plotTiling`
    """
    cimg = overlayPoints(dataSource, pointSource, pointColor = pointColor, x = x, y = y, z = z)
    return plotTiling(cimg)
        



    
    
    
    
