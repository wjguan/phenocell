import sys
sys.path.append(r'phenotyping')
from activelearning import phenotype
import os, time 

def main():
    # Main script to run cell phenotyping using a convolutional neural network and active learning.
    
    # Parameters needed for input 
    loadModel = False # if we have an existing model that we want to load in and continue training with 
    loadImages = True # if we want to manually annotate at all, then make this True. If we just want to train and predict based on 
                      # already done annotations, then make this False. 
    
    # Parameters regarding the active learning structure 
    num_test_set = 500 # number of elements in a test set used to validate
    initial_num_labeling = 1000
    num_candidates = 1000 # first pass: number of most uncertain samples to choose from 
    max_training_examples = 4500 # if none, then set max training examples to be all training examples    
    random_fraction = 0.5 # the fraction of the total annotation suggestions that should be randomly selected (only use if doing random_uncertain method)
    num_annotation_suggestions = num_candidates # keep it equal to num_candidates to not use representativeness as metric
    
    # File name and directory inputs 
    model_file = r'GFAPr2_test_results\models\GFAPr2.hdf5'
    ResultDirectory = r'GFAPr2_test_results\results' # where all the results are lcoated 
    ProcessedDirectory = r'GFAPr2_test_results\processed' # where our X.bc and y.bc files are located 
    X_filename = r'X_total.bc' # The numpy array or bcolz file containing entire processed data set 
    SpotFile = r'GFAPr2_test_results\inputs\spots.npy'
    NucleusFile = r'GFAPr2_test_results\inputs\Syto.tif'# if we want to display the nucleus channel
    # NucleusFile = None # Use "None" if we don't want to display the nucleus channel in classification 
    DataFile = r'GFAPr2_test_results\inputs\GFAP.tif'
    DataFileRange = {'x':all, 'y':all, 'z':all}
    
    # Parameters regarding neural network model 
    BOUND_SIZE = 32
    max_epochs = 50
    batch_size = 32
    modelType = 'allconv' # What kind of model: allconv, or triplanar: triplanar doesn't work yet 
    
     
    
    
    
    
    Parameters = {'loadModel': loadModel, 'loadImages': loadImages,
                  'num_test_set':num_test_set, 'initial_num_labeling':initial_num_labeling, 'num_candidates':num_candidates, 'num_annotation_suggestions':num_annotation_suggestions,
                  'random_fraction':random_fraction, 'max_training_examples':max_training_examples,
                  'model_file':model_file, 'ResultDirectory':ResultDirectory, 'ProcessedDirectory':ProcessedDirectory, 'X_filename':X_filename, 'NucleusFile':NucleusFile,
                  'SpotFile':SpotFile, 'DataFile':DataFile, 'DataFileRange':DataFileRange,'BOUND_SIZE':BOUND_SIZE, 
                  'modelType':modelType, 'batch_size':batch_size, 'max_epochs':max_epochs}
    phenotype(**Parameters)
    
if __name__ == '__main__':
    main()
    

