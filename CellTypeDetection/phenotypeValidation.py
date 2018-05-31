import numpy as np
import clarity.IO as io 
import clarity.Visualization.Plot as plt
import multiprocessing as mp 
from functools import partial
import os, time

## Compute accuracy on the manual training set 

def distanceBetweenPoints(DataResolution, pointOne, pointTwo):
    return np.sqrt(np.sum(np.square(DataResolution*(pointOne-pointTwo))))

def computeAccuracyKernel(CorrectThreshold, DataResolution, groundTruth, expressionPoint):
    for center in groundTruth:
        if distanceBetweenPoints(DataResolution, expressionPoint, center) <= CorrectThreshold:
            return 1
            break 
    return 0

def computeAccuracy(CorrectThreshold, DataResolution, groundTruth, expressionVector, num_workers, nucleiVector=None):
    # Parallel kernel for doing this 
    start = time.time()
    g = mp.Pool(num_workers)
    func = partial(computeAccuracyKernel, CorrectThreshold, DataResolution, groundTruth)
    correct = g.map(func, expressionVector)
    g.close()
    g.join()
    print('Time elapsed (check accuracy):',time.time()-start) 
    TP = np.sum(correct)
    print("True positives:",TP)
    # F1 score = 2*precision*recall/(recall + precision)
    recall = TP/groundTruth.shape[0] # recall = TP/(TP+FN)
    precision = TP/expressionVector.shape[0] # precision = TP/(TP + FP)
    if nucleiVector is not None:
        TN = nucleiVector.shape[0] - expressionVector.shape[0]
        accuracy = (TP + TN) / (TN + expressionVector.shape[0] + groundTruth.shape[0] - TP)
        return 2*recall*precision / (recall + precision), precision, recall, TP, accuracy 
    else:
        return 2*recall*precision / (recall + precision), precision, recall, TP # just return F1, precision, recall, true positive 
    
def unParallelAccuracy(CorrectThreshold, DataResolution, groundTruth, expressionVector):
    accurate = 0 
    for point in expressionVector:
        for center in groundTruth:
            if distanceBetweenPoints(DataResolution, point, center) <= CorrectThreshold:
                accurate += 1
    print(accurate)
    return accurate / (groundTruth.shape[0] + expressionVector.shape[0]-accurate)

def loadPointsInRange(FilePath, ValidationRange):
    expressionVector = np.array([[0,0,0]])
    for i in range(len(ValidationRange)):
        expressionVector = np.concatenate((expressionVector,io.readPoints(FilePath, **ValidationRange[i])), axis=0)
    return expressionVector[1:,:]

def validate(ResultDirectory, DataFile, ValidationRange, CorrectThreshold, DataResolution, num_workers, visualizeErrors, expressedPoints=None, considerUnaccountedFor=True):
    ''' Main script for cell phenotype validation. '''
    
    bdir = lambda f: os.path.join(ResultDirectory, f)
    # Load all points within the range
    if expressedPoints is None:    
        expressionVector = loadPointsInRange(bdir('positive_cells.npy'), ValidationRange)
    else:
        expressionVector = loadPointsInRange(expressedPoints, ValidationRange)
    if os.path.isfile(bdir('ground_truth.npy')):
        GroundTruth = np.loadtxt(bdir('Log.txt'))
        np.save(bdir('ground_truth.npy'),GroundTruth)
        GroundTruth = loadPointsInRange(bdir('ground_truth.npy'),ValidationRange)
    else:
        GroundTruth = loadPointsInRange(bdir('ground_truth.npy'),ValidationRange)
    # Total nuclei detected 
    nucleiVector = loadPointsInRange(bdir('spots_filtered.npy'),ValidationRange)
    print("Number of total nuclei detected:",nucleiVector.shape[0])
    print("Number of cells in ground truth:",GroundTruth.shape[0])
    print("Number of detected cells:",expressionVector.shape[0])
    
    if num_workers <= 1:
        # Do the serial version
        start = time.time()
        accuracy = unParallelAccuracy(CorrectThreshold, DataResolution, GroundTruth, expressionVector)
        print("Serial time elapsed:",time.time()-start)
    else:
        F1, precision, recall, TP, accuracy = computeAccuracy(CorrectThreshold, DataResolution, GroundTruth, expressionVector, num_workers, nucleiVector)
    print("F1 score:",F1)
    print("Precision (TP/(TP + FP)):",precision)
    print("Recall (TP/(TP + FN)):",recall)
    #print("Accuracy (w.r.t detected nuclei):",accuracy)
    
    if considerUnaccountedFor:
        # Also want to check how many of the ground truth are not detected by Syto to begin with 
        if not os.path.isfile(bdir('nuclei_counted.npy')):
            accountedFor = []
            for i in range(GroundTruth.shape[0]):
                found = False; j = 0 
                while j < len(nucleiVector) and found == False:
                    if distanceBetweenPoints(DataResolution, GroundTruth[i], nucleiVector[j]) <= CorrectThreshold:
                        found = True 
                        accountedFor.append(nucleiVector[j])
                    j += 1    
            np.save(bdir('nuclei_counted.npy'),np.asarray(accountedFor))
            numUnaccounted = GroundTruth.shape[0] - len(accountedFor)
            accountedFor = np.asarray(accountedFor)
            print("Number of unaccounted for:",numUnaccounted)
            
            
        else:
            accountedFor = loadPointsInRange(bdir('nuclei_counted.npy'), ValidationRange)
            
        F1_mod = 2*TP/(expressionVector.shape[0] + len(accountedFor))
        recall_mod = TP/len(accountedFor)
        print("Modified recall:",recall_mod) 
        print("Modified F1 score:",F1_mod)
    else:
        F1_mod = F1; recall_mod = recall
    
    if visualizeErrors:
        # Make 3 channel image: 1st channel is cell type marker; 2nd channel: ground truth; 3rd channel: detected cells 
        for i in range(len(ValidationRange)):
            Rnge = ValidationRange[i] 
            overlay = plt.overlayPoints(DataFile, accountedFor, pointColor = None, **Rnge)
            overlay2 = plt.overlayPoints(DataFile, expressionVector, pointColor = None, **Rnge)
            overlay = np.concatenate((overlay, overlay2[:,:,:,1:]), axis=3)
            #print(np.sum(overlay[:,:,:,1]>0), np.sum(overlay[:,:,:,2]>0))
            io.writeData(bdir('groundtruth_overlaid_%d.tif'%i),overlay)
        
    return F1_mod, precision, recall_mod, recall
    
