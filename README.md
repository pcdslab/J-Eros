# J-Eros

The input files from KKI, OHSU and NYU are provided from :
https://www.nitrc.org/plugins/mwiki/index.php/neurobureau:AthenaPipeline#Extracted_Time_Courses 
for training data
and 
https://www.nitrc.org/plugins/mwiki/index.php/neurobureau:AthenaPipeline#Test_Dataset
for test data. Data from cc200 time courses is used for this study.

Please note: The first part of the code for reading input data is designed based of the structure of the text files containing time series values of different regions from the links above. If you want to use the code for other datasets make sure to change the first part of the code to be consistent with the input files you want to use. 
The data should be 

The output displays the best value of k picked by J-Eros algorithm and the accuracy of performing k nearest neighbor on test set, and also achieved sensitive and specificity.

To run:
Python J-Eros.py

Specification of system and libraries which used for testing the code:
Ubuntu version 14.04.2
Python version 3.5
numpy version 1.13.1
scipy version 0.19.1
scikit-learn version 0.19.0
