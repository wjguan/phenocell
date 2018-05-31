README: clarity\CellTypeDetection

Prequisites: 
1. "spots_filtered.npy" file containing all of the nuclei centers detected using spot detection 
IF DOING QUANTIFICATION / VALIDATION
2. "Log.txt" file generated from ImageJ containing all of the cell-type positive nuclei centers (MAKE SURE THAT THEY ARE THE CORRECT COORDINATES)
3. "activate webster" works in all accounts, but "activate tensorflow" should only be used in Clarity 
4. "activate webster" or "activate tensorflow" should work for cell phenotyping, but only tensorflow works for nucleus detection 
PROCESS:
1. Run spot detection for nucleus detection (spotdetection_Sytotest.py); change parameters as needed 
    a) check parameters on small sections first before running on entire data set 
    b) make sure to "activate tensorflow" for this. 
2. If we want to validate quantitatively:
    a) generate "cells_check.tif" (two channel image of the Syto16 and detected cell centers) 
    b) split image into channels, then merge with the cell type marker channel 
    c) click away using the custom click coordinates plugin (IMPORTANT: DOUBLE CLICK THE ICON AND MAKE SURE TO NOT CLICK "SCALED COORDINATES" AND ADJUST THE NUMBER OF CHANNELS, and make sure it's 16 bit)
    d) save the Log.txt as Log.txt in the appropriate folder with everything else 
    e) make sure you note what the coordinates of the region(s) you labeled were 

Relevant script files: 
1. "phenotypeCellsScript.py": script that allows the user to either - 
    a) run phenotypeCellsExp.py once with a given set of parameters
        i) check the quantatitive accuracies on a given section that has been validated (additionally; visualize the errors)
        ii) visualize the performance of the detection on a given section (eye test)
        iii) once parameters are decided, you can just run it like this on the entire data set 
    b) run phenotypeCellsExp.py over a grid search (choose which parameters to test in grid search)
        i) shuold only be used if you want to visualize the performance of many parameter sets 
        ii) if you have the quantitative validation capability (i.e. you manually annotated some of it) then you can quantitatively view the results.
        iii) produces a file called "gridSearchResults_combined.csv" - you can look at which parameters correspond to which images and F1 scores based on this file. 
2. "phenotypeValScript.py": 
    a) run if you are validating the detected cells that have already been saved (requires (1) "positive_cells.npy", and (2) "ground_truth.npy" or "Log.txt" in the appropriate directory). 
    
    
COMMON ERRORS:
During spot detection...
"axes don't match array" or any kind of error that is not obvious what's going on from reading the error message:
1. Check that the image you've input has 3 dimensions (x,y,z): sometimes, especially with Leica images, it for some reason has 3 channels (red, green, blue), and thus has 
   four dimensions. You can check this by doing the following in the command prompt (each as a separate line) 
    i) activate tensorflow
    ii) python 
    iii) import clarity.IO as io 
    iv) import numpy as np 
    v) g = io.readData(r'<insert path of file here>')
    vi) g.shape 
   Now, if there are four dimensions in the shape, then you need to take whatever channel actually has all the information you want. 
   vii) print(np.amax(g[:,:,:,0]), np.amax(g[:,:,:,1]), np.amax(g[:,:,:,2]))
   Now, you take whichever channel has a non-zero max value printed out on to the screen. FOR EXAMPLE, if it's channel 0, then:
   viii) io.writeData(r'<insert path of file here>', g[:,:,:,0])
2. Now, after step (vi) if the image does indeed have 3 channels and it still throws an error, then do the following (simply resave the image):
    vii) io.writeData(r'<insert path of file here>', g)
   That should solve all your problems with this. 
