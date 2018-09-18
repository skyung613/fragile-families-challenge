from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
import csv
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, f_regression, SelectKBest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

"""                                                  
URLs for referencing different classifiers:                           
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

"""

def import_data():

    #import data                                                          
    data = pd.read_csv("imputed_background_gpa.csv", low_memory = False)
    y = pd.read_csv("imputed_train_gpa.csv")

    y_gpa = y[['challengeID','gpa']]    
    X = y_gpa.join(data.set_index('challengeID'), on='challengeID')
    X = X.iloc[:, X.columns != 'challengeID']

    return X

#linear relationship between features and gpa
def plot_feature_selection(X):
    plt.figure(1)
    sb.regplot(y=X['gpa'], x=X['cm1edu'], fit_reg = True)
    plt.savefig('cm1edu.png')
    
    plt.figure(2)
    sb.regplot(y=X['gpa'], x=X['cf1edu'], fit_reg = True)
    plt.savefig('cf1edu.png')
    
def find_param(X):
    
    X_data = X.iloc[:, X.columns != 'gpa']
    y_data = X.iloc[:, X.columns == 'gpa']

    #Linear Regression with Lasso
    param_grid = {"alpha": [1000, 10000],
                  "fit_intercept": [True],
                  "normalize": [True, False],
                  "tol": [0.01, 0.1, 1.0],
                  "positive": [True, False],
                  "selection": ["random"],
                  "max_iter": [1000]
                  }
    lrl = Lasso()
    lrl_gs = GridSearchCV(lrl,
                         param_grid=param_grid,
                         scoring='r2', 
                         n_jobs=1,
                         verbose=2,
                         cv=3)
    lrl_gs.fit(X_data, y_data)
    print lrl_gs.best_params_

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
    rf = RandomForestRegressor() 
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                   n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    rf_random.fit(X_data, y_data)
    print '*' * 10
    print rf_random.best_params_
    
    n_neighbors = [int(x) for x in np.linspace(start = 10, stop = 900, num = 10)]
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    param_grid = {'n_neighbors': n_neighbors,
                  'algorithm': algorithm}
    knn = KNeighborsRegressor()
    knn_gs = GridSearchCV(estimator = knn, param_grid = param_grid, cv = 5)
    knn_gs.fit(X_data, y_data)
    print knn_gs.best_params_

def train_cont(X):

    y = X.iloc[:, X.columns == 'gpa']
    X = X.iloc[:, X.columns != 'gpa']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)

    #{'bootstrap': True, 'min_samples_leaf': 2, 'n_estimators': 1000, 
    #'max_features': 'sqrt', 'min_samples_split': 10, 'max_depth': 10}
    rfr = RandomForestRegressor(n_estimators=1000, bootstrap=True, min_samples_leaf=2, 
                                max_features='sqrt', min_samples_split=10, max_depth=10)
    start_time = time.time()
    rfr.fit(X_train, y_train.values.ravel()) 
    rfr_y_predicted = rfr.predict(X_test)
    end_time = time.time()
    rfr_time = end_time - start_time
    rfr_mse = mean_squared_error(y_test, rfr_y_predicted) 
    rfr_r2 = r2_score(y_test, rfr_y_predicted)
    rfr_acc = 0

    for i in range(len(rfr_y_predicted)):
        if y_test.iloc[i].values-.25 <= rfr_y_predicted[i] <= y_test.iloc[i].values+.25:
            rfr_acc = rfr_acc + 1

    print "RFR: "
    print "MSE: ", rfr_mse
    print "R^2: ", rfr_r2
    print "Acc: ", rfr_acc/len(rfr_y_predicted)
    print "Time: ", rfr_time
    print '*' * 50

    #{'normalize': True, 'selection': 'random', 'fit_intercept': True, 
    #'positive': True, 'max_iter': 1000, 'tol': 0.01, 'alpha': 1000}
    lnr = Lasso(normalize=True, selection='random', fit_intercept=True,
                positive=True, max_iter=1000, tol=0.01, alpha=1000)
    start_time = time.time()                                                                     
    lnr.fit(X_train, y_train.values.ravel())         
    lnr_y_predicted = lnr.predict(X_test)                                                 
    end_time = time.time()                                                                       
    lnr_time = end_time - start_time  
    lnr_mse = mean_squared_error(y_test, lnr_y_predicted)
    lnr_r2 = r2_score(y_test, lnr_y_predicted)
    lnr_acc = 0
    
    for i in range(len(lnr_y_predicted)):
        if y_test.iloc[i].values-.25 <= lnr_y_predicted[i] <= y_test.iloc[i].values+.25:
            lnr_acc = lnr_acc + 1

    print "LNR: "
    print "MSE: ", lnr_mse
    print "R^2: ", lnr_r2
    print "Acc: ", lnr_acc/len(lnr_y_predicted)
    print "Time: ", lnr_time
    print '*' * 50

    #{'n_neighbors': 108, 'algorithm': 'ball_tree'}
    knr = KNeighborsRegressor(n_neighbors=100)
    start_time = time.time()                                                                     
    knr.fit(X_train, y_train.values.ravel())                                                     
    knr_y_predicted = knr.predict(X_test)                                                        
    end_time = time.time()                                                                       
    knr_time = end_time - start_time                                                              
    knr_mse = mean_squared_error(y_test, knr_y_predicted) 
    knr_r2 = r2_score(y_test, knr_y_predicted)
    knr_acc = 0

    for i in range(len(knr_y_predicted)):
        if y_test.iloc[i].values-.25 <= knr_y_predicted[i] <= y_test.iloc[i].values+.25:
            knr_acc = knr_acc + 1

    print "KNR: "
    print "MSE: ", knr_mse
    print "R^2: ", knr_r2
    print "Acc: ", knr_acc/len(knr_y_predicted)
    print "Time: ", knr_time
    print '*' * 50

def train_cont_feat(X):

    y = X.iloc[:, X.columns == 'gpa']
    X = X.iloc[:, X.columns != 'gpa']

    selector = SelectKBest(f_regression, k=10).fit(X, y.values.ravel())
    top_ten_feat = selector.get_support(indices=True)
    top_ten = X.iloc[:, top_ten_feat]
    print "Top ten features: "
    print top_ten.columns

    print "Training with Feature Selection"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)

    select_feature = SelectKBest(f_regression, k=500).fit(X_train, y_train.values.ravel())
    rfr_X_train = select_feature.transform(X_train)
    rfr_X_test = select_feature.transform(X_test)
    
    rfr = RandomForestRegressor(n_estimators=1000, bootstrap=True, min_samples_leaf=2,
                                max_features='sqrt', min_samples_split=10, max_depth=10)

    start_time = time.time()
    rfr.fit(rfr_X_train, y_train.values.ravel())
    rfr_y_predicted = rfr.predict(rfr_X_test)
    end_time = time.time()
    rfr_time = end_time - start_time
    rfr_mse = mean_squared_error(y_test, rfr_y_predicted)
    rfr_r2 = r2_score(y_test, rfr_y_predicted)
    rfr_acc = 0

    for i in range(len(rfr_y_predicted)):
        if y_test.iloc[i].values-.25 <= rfr_y_predicted[i] <= y_test.iloc[i].values+.25:
            rfr_acc = rfr_acc + 1

    print "RFR: "
    print "MSE: ", rfr_mse
    print "R^2: ", rfr_r2
    print "Acc: ", rfr_acc/len(rfr_y_predicted)
    print "Time: ", rfr_time
    print '*' * 50

    select_feature = SelectKBest(f_regression, k=500).fit(X_train, y_train)
    lnr_X_train = select_feature.transform(X_train)
    lnr_X_test = select_feature.transform(X_test)

    lnr = Lasso(normalize=True, selection='random', fit_intercept=True,
                positive=True, max_iter=1000, tol=0.01, alpha=1000)

    start_time = time.time()
    lnr.fit(lnr_X_train, y_train.values.ravel())
    lnr_y_predicted = lnr.predict(lnr_X_test)
    end_time = time.time()
    lnr_time = end_time - start_time
    lnr_mse = mean_squared_error(y_test, lnr_y_predicted)
    lnr_r2 = r2_score(y_test, lnr_y_predicted)
    lnr_acc = 0

    for i in range(len(lnr_y_predicted)):
        if y_test.iloc[i].values-.25 <= lnr_y_predicted[i] <= y_test.iloc[i].values+.25:
            lnr_acc = lnr_acc + 1

    print "LNR: "
    print "MSE: ", lnr_mse
    print "R^2: ", lnr_r2
    print "Acc: ", lnr_acc/len(lnr_y_predicted)
    print "Time: ", lnr_time
    print '*' * 50

    select_feature = SelectKBest(f_regression, k=500).fit(X_train, y_train)
    knr_X_train = select_feature.transform(X_train)
    knr_X_test = select_feature.transform(X_test)

    knr = KNeighborsRegressor(n_neighbors=100, algorithm='ball_tree')

    start_time = time.time()
    knr.fit(knr_X_train, y_train.values.ravel())
    knr_y_predicted = knr.predict(knr_X_test)
    end_time = time.time()
    knr_time = end_time - start_time
    knr_mse = mean_squared_error(y_test, knr_y_predicted)
    knr_r2 = r2_score(y_test, knr_y_predicted)
    knr_acc = 0

    for i in range(len(knr_y_predicted)):
        if y_test.iloc[i].values-.25 <= knr_y_predicted[i] <= y_test.iloc[i].values+.25:
            knr_acc = knr_acc + 1

    print "KNR: "
    print "MSE: ", knr_mse
    print "R^2: ", knr_r2
    print "Acc: ", knr_acc/len(knr_y_predicted)
    print "Time: ", knr_time
    print '*' * 50
    
def main():
    X = import_data()
    #find_param(X)
    train_cont(X)
    train_cont_feat(X)
    plot_feature_selection(X)
    
if __name__ == "__main__":
  main()
  

