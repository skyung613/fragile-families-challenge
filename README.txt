----------------------------------------------
COS 424: Fragile Families Challenge (Spring 2018)
----------------------------------------------

Packages to install:
numpy
scipy
scikitlearn
fancyimpute

This directory comprises the following files

- impute_data.py, called as follows: 
	python ./impute_data.py 

The script looks in current directory and given background.csv and train.csv, generates two output files in the working directory: 
	- imputed_background_*.csv 
	- imputed_train_*.csv
	- imputing data for each of six labels is hardcoded in the script - can easily disable/enable comments to impute background and train datasets for one of six labels.

- classifier.py, called as follows:
	python classifier.py

The script looks in current directory and given train_eviction.csv and background_eviction.csv, generates violin plot of eviction distributions on selected features and saves the resulting images in the working directory, , performs chi-squared feature selection, uses grid search and randomized search for hyper parameter tuning, and trains three classification models.

- regression.py, called as follows:
	python regression.py

The script looks in current directory and given train_*.csv and background_*.csv, generates linear regression plots for two selected features and saves the resulting images in the working directory, performs f_regression feature selection, uses grid search and randomized search for hyper parameter tuning, and trains three regression models.

- ffc_predict.py, called as follows:
	python ffc_predict.py

The script takes all training sets and background sets imputed for the six labels and trains RFR for continuous labels and RFC for binary labels. The resulting predictions are saved as prediction.csv.



