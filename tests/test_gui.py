import time
import phenotyping.clarity.IO as io 
from phenotyping.annotationgui import manualValidate 

def main():
    # Test the GUI
    
    ## Define the inputs 
    # Range of images we want to load 
    DataFileRange = {'x':[0,1000],'y':[0,1000],'z':[0,10]}
    # Raw cell type marker image files
    DataFile = r'D:\analysis\data\raw\NPY647_Syto16488_12192017\NPY647\NPY647_Z[0-9]{4}.tif'
    # Detected nucleus centers file
    CellCentersFile = r'D:\analysis\results\20171219_NPY647_Syto16488\NPY647\positive_cells.npy'
    # Raw nucleus image files
    NucleusImageFile = r'D:\analysis\data\raw\NPY647_Syto16488_12192017\Syto16\Syto16_Z[0-9]{4}.tif'
    # File to save the results of the annotation 
    SaveDirectory = r'D:\analysis\results\20171219_NPY647_Syto16488\NPY647\test_gui.npy'
    
    
    start = time.time()
    img = io.readData(DataFile,**DataFileRange)
    print("Time elapsed to load image:",(time.time()-start)/60, "minutes")
    cell_centers =  io.readPoints(CellCentersFile,**DataFileRange)
    start = time.time()
    nucleusImage = io.readData(NucleusImageFile,**DataFileRange)
    print("Time elapsed to load nucleus image:",(time.time()-start)/60, "minutes")
    manualValidate(cell_centers, img, SaveDirectory, nucleusImage)
    
if __name__ == '__main__':
    main()