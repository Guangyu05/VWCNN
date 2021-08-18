# VWCNN

This is the code of the paper 'Variable Weights Algorithm for Convolutional Neural Networks and its Application to Classification in Epilepsy'.

The TUSZ dataset is processed as follows

1. Download the dataset through the rsync command ```rsync -auxvL nedc@www.isip.piconepress.com:data/tuh_eeg_seizure/v1.4.1/ .``` or follow the instructions on https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)

2. As we used the benchmark dataset provided by IBM Features For Seizure Detection (IBMFT), the raw data are processed with the procedures on https://github.com/IBM/seizure-type-classification-tuh/tree/master/data_preparation (Note that the file 'generate_fft_images.py' on https://github.com/IBM/seizure-type-classification-tuh/tree/master/data_preparation) has minor errors, readers can use the file with the same name in this respository to get IBM features.

3. Run the file 'data_processing_train_val_test.py' to obtain the training, validation, and testing datasets used in this paper.
