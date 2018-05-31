import sys
self = sys.modules[__name__]

import numpy
from scipy import stats

import clarity.IO as io
import clarity.Analysis.Label as lbl

def readDataGroup(filenames, combine = True, **args):
    """Turn a list of filenames for data into a numpy stack"""
    
    #check if stack already:
    if isinstance(filenames, numpy.ndarray):
        return filenames
    
    #read the individual files
    group = []
    for f in filenames:
        data = io.readData(f, **args)
        data = numpy.reshape(data, (1,) + data.shape)
        group.append(data)
    
    if combine:
        return numpy.vstack(group)
    else:
        return group


def readPointsGroup(filenames, **args):
    """Turn a list of filenames for points into a numpy stack"""
    
    #check if stack already:
    if isinstance(filenames, numpy.ndarray):
        return filenames
    
    #read the individual files
    group = []
    for f in filenames:
        data = io.readPoints(f, **args)
        #data = numpy.reshape(data, (1,) + data.shape)
        group.append(data)
    
    return group
    #return numpy.vstack(group)


def tTestVoxelization(group1, group2, signed = False, removeNaN = True, pcutoff = None):
    """t-Test on differences between the individual voxels in group1 and group2, group is a array of voxelizations"""
    
    g1 = self.readDataGroup(group1)
    g2 = self.readDataGroup(group2)
    
    tvals, pvals = stats.ttest_ind(g1, g2, axis = 0, equal_var = True)

    #remove nans
    if removeNaN: 
        pi = numpy.isnan(pvals)
        pvals[pi] = 1.0
        tvals[pi] = 0

    pvals = self.cutoffPValues(pvals, pcutoff = pcutoff)

    #return
    if signed:
        return pvals, numpy.sign(tvals)
    else:
        return pvals

        
def cutoffPValues(pvals, pcutoff = 0.05):
    if pcutoff is None:
        return pvals
    
    pvals2 = pvals.copy()
    pvals2[pvals2 > pcutoff]  = pcutoff
    return pvals2
    

def colorPValues(pvals, psign, positive = [1,0], negative = [0,1], pcutoff = None, positivetrend = [0,0,1,0], negativetrend = [0,0,0,1], pmax = None):
    
    pvalsinv = pvals.copy()
    if pmax is None:
        pmax = pvals.max()
    pvalsinv = pmax - pvalsinv
    
    if pcutoff is None:  # color given p values
        
        d = len(positive)
        ds = pvals.shape + (d,)
        pvc = numpy.zeros(ds)
    
        #color
        ids = psign > 0
        pvalsi = pvalsinv[ids]
        for i in range(d):
            pvc[ids, i] = pvalsi * positive[i]
    
        ids = psign < 0
        pvalsi = pvalsinv[ids]
        for i in range(d):
            pvc[ids, i] = pvalsi * negative[i]
        
        return pvc
        
    else:  # split pvalues according to cutoff
    
        d = len(positivetrend)
        
        if d != len(positive) or  d != len(negative) or  d != len(negativetrend) :
            raise RuntimeError('colorPValues: postive, negative, postivetrend and negativetrend option must be equal length!')
        
        ds = pvals.shape + (d,)
        pvc = numpy.zeros(ds)
    
        idc = pvals < pcutoff
        ids = psign > 0

        ##color 
        # significant postive
        ii = numpy.logical_and(ids, idc)
        pvalsi = pvalsinv[ii]
        w = positive
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i]
    
        #non significant postive
        ii = numpy.logical_and(ids, numpy.negative(idc))
        pvalsi = pvalsinv[ii]
        w = positivetrend
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i]
            
         # significant negative
        ii = numpy.logical_and(numpy.negative(ids), idc)
        pvalsi = pvalsinv[ii]
        w = negative
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i]
    
        #non significant postive
        ii = numpy.logical_and(numpy.negative(ids), numpy.negative(idc))
        pvalsi = pvalsinv[ii]
        w = negativetrend
        for i in range(d):
            pvc[ii, i] = pvalsi * w[i]
        
        return pvc
    

    
    
def mean(group, **args):
    g = self.readGroup(group, **args)
    return g.mean(axis = 0)


def std(group, **args):
    g = self.readGroup(group, **args)
    return g.std(axis = 0)

   
def var(group, **args):
    g = self.readGroup(group, **args)
    return g.var(axis = 0)
    



    
def thresholdPoints(points, intensities, threshold = 0, row = 0):
    """Threshold points by intensities"""
    
    points, intensities = io.readPoints((points, intensities))
            
    if not isinstance(threshold, tuple):
        threshold = (threshold, all)
    
    if not isinstance(row, tuple):
        row = (row, row)
    
    
    if intensities.ndim > 1:
        i = intensities[:,row[0]]
    else:
        i = intensities
    
    iids = numpy.ones(i.shape, dtype = 'bool')
    if not threshold[0] is all:
        iids = numpy.logical_and(iids, i >= threshold[0])
        
    if intensities.ndim > 1:
        i = intensities[:,row[1]]
    
    if not threshold[1] is all:
        iids = numpy.logical_and(iids, i <= threshold[1])
    
    return (points[iids, ...], intensities[iids, ...])




def weightsFromPrecentiles(intensities, percentiles = [25,50,75,100]):
    perc = numpy.percentiles(intensities, percentiles)
    weights = numpy.zeros(intensities.shape)
    for p in perc:
        ii = intensities > p
        weights[ii] = weights[ii] + 1
    
    return weights
        

def countPointsGroupInRegions(pointGroup, labeledImage = lbl.DefaultLabeledImageFile, intensityGroup = None, intensityRow = 0, returnIds = True, returnCounts = False, collapse = None):
     """Generates a table of counts for the various point datasets in pointGroup"""
 
     if intensityGroup is None: 
         counts = [lbl.countPointsInRegions(pointGroup[i], labeledImage = labeledImage, sort = True, allIds = True, returnIds = False, returnCounts = returnCounts, intensities = None, collapse = collapse) for i in range(len(pointGroup))]
     else:
         counts = [lbl.countPointsInRegions(pointGroup[i], labeledImage = labeledImage, sort = True, allIds = True, returnIds = False, returnCounts = returnCounts,
                                            intensities = intensityGroup[i], intensityRow = intensityRow, collapse = collapse) for i in range(len(pointGroup))]
     
     if returnCounts and not intensityGroup is None:
         countsi = (c[1] for c in counts)
         counts  = (c[0] for c in counts)
     else:
         countsi = None
         
     counts = numpy.vstack((c for c in counts)).T
     if not countsi is None:
         countsi =  numpy.vstack((c for c in countsi)).T
     
     if returnIds:
         ids = numpy.sort(lbl.Label.ids)
         #ids.shape = (1,) + ids.shape
         
         #return numpy.concatenate((ids.T,counts), axis = 1
         if countsi is None:
             return ids, counts
         else:
             return ids, counts, countsi      
     else:
         if countsi is None:
             return counts
         else:
             return counts, countsi
         
