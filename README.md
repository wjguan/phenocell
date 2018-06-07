This package uses detected nuclei centers to phenotype the cells as positive or negative for a given marker. 

FILES NEEDED
1. positive_cells.npy (or spots_filtered.npy): array of every cell center, unshuffled; shape (num_positive_cells, 3)
2. Cell type marker TIFF files 
3. Nucleus channel TIFF files

FILES GENERATED
ProcessedDirectory (D:\analysis\data\processed\~)
1. X_total.bc: bcolz file containing all of the unshuffled image sections; shape (num_positive_cells, 32, 32, 2)
2. X_test.bc: bcolz file (shuffled) containing test data (image sections); shape (num_test_set, 32, 32, 2)
3. y_test.bc: after annotation, the binary labels; shape (num_test_set,)
4. X_train.bc: bcolz file (shuffled) containing train data (image sections); shape (num_train, 32, 32, 2)
5. y_train.bc: bcolz file (shuffled) containing train labels; shape (num_train) 
6. X_unannotated.bc: bcolz file (shuffled) containing all of the unannotated image sections; shape (num_positive_cells - num_test_set - num_train, 32, 32, 2)

ResultDirectory (D:\analysis\results\~)
2. cell_centers_all.npy: array of every cell center, shuffled;  shape (num_positive_cells, 3)
3. cell_centers_test.npy: array of test cell centers, shuffled; shape (num_test_set, 3)
4. cell_centers_annotated.npy: array of annotated cell centers, shuffled; shape (num_train, 3)
5. cell_centers_unannotated.npy: array of remaining unannotated cell centers; shape (num_positive_cells - num_train - num_test, 3)
6. indices.npy: array of the shuffled indices; shape (num_positive_cells, 3)
7. y_current_annotation.npy: saved array of all of the annotations of the current iteration; shape (num_annotated_this_iteration, 2)
8. train_dices.npy, train_accuracies.npy: DICE and  accuracies for the training set 
9. test_dices.npy, test_accuracies.npy: DICE and accuracies for the test set

model_file (D:\analysis\models\...) i.e. where models are stored 
1. model_file 