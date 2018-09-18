from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, f_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def import_data():

    #import data for gpa                               
    data = pd.read_csv("imputed_background_gpa.csv", low_memory = False)
    y = pd.read_csv("imputed_train_gpa.csv")
    y_gpa = y[['challengeID','gpa']]    
    X_gpa = y_gpa.join(data.set_index('challengeID'), on='challengeID')

    data = pd.read_csv("imputed_background_eviction.csv", low_memory = False)
    y = pd.read_csv("imputed_train_eviction.csv")
    y_eviction = y[['challengeID','eviction']]
    X_eviction = y_eviction.join(data.set_index('challengeID'), on='challengeID')

    data = pd.read_csv("imputed_background_grit.csv", low_memory = False)
    y = pd.read_csv("imputed_train_grit.csv")
    y_grit = y[['challengeID','grit']]
    X_grit = y_grit.join(data.set_index('challengeID'), on='challengeID')
    
    data = pd.read_csv("imputed_background_materialHardship.csv", low_memory = False)
    y = pd.read_csv("imputed_train_materialHardship.csv")
    y_mh = y[['challengeID','materialHardship']]
    X_mh = y_mh.join(data.set_index('challengeID'), on='challengeID')

    data = pd.read_csv("imputed_background_layoff.csv", low_memory = False)
    y = pd.read_csv("imputed_train_layoff.csv")
    y_layoff = y[['challengeID','layoff']]
    X_layoff = y_layoff.join(data.set_index('challengeID'), on='challengeID')
    
    data = pd.read_csv("imputed_background_jobTraining.csv", low_memory = False)
    y = pd.read_csv("imputed_train_jobTraining.csv")
    y_jobTraining = y[['challengeID','jobTraining']]
    X_jobTraining = y_jobTraining.join(data.set_index('challengeID'), on='challengeID')

    return X_gpa, X_eviction, X_grit, X_mh, X_layoff, X_jobTraining

def train(X_gpa, X_eviction, X_grit, X_mh, X_layoff, X_jobTraining):

    y = X_gpa.iloc[:, X_gpa.columns == 'gpa']
    X = X_gpa.iloc[:, X_gpa.columns != 'gpa']
    X = X.iloc[:, X.columns != 'challengeID']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)

    rfr = RandomForestRegressor(n_estimators=1000, bootstrap=True, min_samples_leaf=2, 
                                max_features='sqrt', min_samples_split=10, max_depth=10)
    rfr.fit(X_train, y_train.values.ravel()) 
    y_gpa = rfr.predict(X)
    
    y = X_grit.iloc[:, X_grit.columns == 'grit']
    X = X_grit.iloc[:, X_grit.columns != 'grit']
    X = X.iloc[:, X.columns != 'challengeID']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)
    rfr = RandomForestRegressor(n_estimators=1000, bootstrap=True, min_samples_leaf=2,
                                max_features='sqrt', min_samples_split=10, max_depth=10)
    rfr.fit(X_train, y_train.values.ravel())
    y_grit = rfr.predict(X)
    
    y = X_mh.iloc[:, X_mh.columns == 'materialHardship']
    X = X_mh.iloc[:, X_mh.columns != 'materialHardship']
    X = X.iloc[:, X.columns != 'challengeID']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)
    rfr = RandomForestRegressor(n_estimators=1000, bootstrap=True, min_samples_leaf=2,
                                max_features='sqrt', min_samples_split=10, max_depth=10)
    rfr.fit(X_train, y_train.values.ravel())
    y_mh = rfr.predict(X)

    y = X_eviction.iloc[:, X_eviction.columns == 'eviction']
    X = X_eviction.iloc[:, X_eviction.columns != 'eviction']
    X = X.iloc[:, X.columns != 'challengeID']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)

    rfc = RandomForestClassifier(n_estimators=20)
    rfc.fit(X_train, y_train.values.ravel())
    y_evic = rfc.predict_proba(X)
    
    y = X_layoff.iloc[:, X_layoff.columns == 'layoff']
    X = X_layoff.iloc[:, X_layoff.columns != 'layoff']
    X = X.iloc[:, X.columns != 'challengeID']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)
                                               
    rfc = RandomForestClassifier(n_estimators=20)
    rfc.fit(X_train, y_train.values.ravel())
    y_layoff = rfc.predict_proba(X)

    y = X_jobTraining.iloc[:, X_jobTraining.columns == 'jobTraining']
    X = X_jobTraining.iloc[:, X_jobTraining.columns != 'jobTraining']
    X = X.iloc[:, X.columns != 'challengeID']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)

    rfc = RandomForestClassifier(n_estimators=20)
    rfc.fit(X_train, y_train.values.ravel())
    y_jobTraining = rfc.predict_proba(X)

    # prediction.csv using the best performance regression
    pred = pd.read_csv('prediction.csv')
    pred.loc[X_gpa['challengeID'], 'gpa'] = y_gpa
    pred.loc[X_grit['challengeID'], 'grit'] = y_grit
    pred.loc[X_mh['challengeID'], 'materialHardship'] = y_mh
    pred.loc[X_eviction['challengeID'], 'eviction'] = y_evic[:, 1]
    pred.loc[X_layoff['challengeID'], 'layoff'] = y_layoff[:, 1]
    pred.loc[X_jobTraining['challengeID'], 'jobTraining'] = y_jobTraining[:, 1]
    pred.to_csv('prediction.csv', index=False)
    
def main():
    X_gpa, X_evic, X_grit, X_mh, X_layoff, X_jobTraining = import_data()
    train(X_gpa, X_evic, X_grit, X_mh, X_layoff, X_jobTraining)
    
if __name__ == "__main__":
  main()
  

