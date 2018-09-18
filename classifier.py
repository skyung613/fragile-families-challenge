from __future__ import division
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import seaborn as sb
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy import interp
import time

"""                                                                                                                 
URLs for referencing different classifiers:                                                               
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html                          
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html                              
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html                           
https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74    
https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization
"""

def import_data():

    #import data                                                                                                         
    data = pd.read_csv("imputed_background_eviction.csv", low_memory = False)    
    y = pd.read_csv("imputed_train_eviction.csv")

    y_eviction = y[['challengeID','eviction']]
    X = y_eviction.join(data.set_index('challengeID'), on='challengeID')

    return X

def feature_selection(X):

    data = X[['cm1ethrace', 'cf1ethrace', 'cm1edu', 'cf1edu',                 
              'cm1age', 'cf1age', 'cf2cohm', 'cm5adult', 'cf5adult', 'cm4kids',                    
              'cf4kids', 'm3c41', 'f3c41', 'm2c33', 'cm1povca', 'cf1povca',                                  
              'cm5povca', 'cf5povca', 'm1j1a', 'm1h3', 'f1h3']]     
    y = X[['eviction']]
    data_n_2 = (data - data.mean()) / (data.std())
    plt.figure(1)
    data = pd.concat([y, data_n_2], axis=1)
    data = pd.melt(data,id_vars="eviction",
                    var_name="features",
                    value_name='value')
    plt.figure(figsize=(20,10))
    sb.violinplot(x="features", y="value", hue="eviction", data=data,split=True, inner="quart")
    plt.xticks(rotation=90)
    plt.savefig('eviction_features.png')

def find_param(X):
    
    y = X.iloc[:, X.columns == 'eviction']
    X = X.iloc[:, X.columns != 'eviction']
    
    rfc = RandomForestClassifier()
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 900, num = 10)]
    param_grid = {'n_estimators': n_estimators}
    clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    clf.fit(X, y.values.ravel())
    print clf.best_params_

    bnb = BernoulliNB()
    alpha = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]
    param_grid = {'alpha': alpha}
    clf = GridSearchCV(estimator=bnb, param_grid=param_grid, cv=5)
    clf.fit(X, y.values.ravel())
    print clf.best_params_

    knn = KNeighborsClassifier()
    n_neighbors = [int(x) for x in np.linspace(start = 10, stop = 900, num = 10)]
    param_grid = {'n_neighbors': n_neighbors}
    clf = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
    clf.fit(X, y.values.ravel())
    print clf.best_params_

def train_cat(X):
    y = X.iloc[:, X.columns == 'eviction']
    X = X.iloc[:, X.columns != 'eviction']

    print "Training without Feature Selection"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)

    rfc = RandomForestClassifier(n_estimators=20)
    start_time = time.time()
    rfc.fit(X_train, y_train.values.ravel())
    rfc_y_predicted = rfc.predict(X_test)
    end_time = time.time()
    rfc_time = end_time - start_time
    rfc_acc = accuracy_score(y_test, rfc_y_predicted)
    print "RFR: "
    print "Acc: ", rfc_acc
    print "Time: ", rfc_time
    print "precision recall fbeta support"
    print precision_recall_fscore_support(y_test, rfc_y_predicted, average='weighted', labels=np.unique(rfc_y_predicted))
    print '*' * 50

    bnb = BernoulliNB(alpha=0.41)
    start_time = time.time()
    bnb.fit(X_train, y_train.values.ravel())
    bnb_y_predicted = bnb.predict(X_test)
    end_time = time.time()
    bnb_time = end_time - start_time
    bnb_acc = accuracy_score(y_test, bnb_y_predicted)
    print "BNB: "
    print "Acc: ", bnb_acc
    print "Time: ", bnb_time
    print "precision recall fbeta support"
    print precision_recall_fscore_support(y_test, bnb_y_predicted, average='weighted', labels=np.unique(bnb_y_predicted))
    print '*' * 50
    
    knn = KNeighborsClassifier(n_neighbors=45)
    start_time = time.time()
    knn.fit(X_train, y_train.values.ravel())
    knn_y_predicted = knn.predict(X_test)
    end_time = time.time()
    knn_time = end_time - start_time
    knn_acc = accuracy_score(y_test, knn_y_predicted)
    print "KNN: "
    print "Acc: ", knn_acc
    print "Time: ", knn_time
    print "precision recall fbeta support"
    print precision_recall_fscore_support(y_test, knn_y_predicted, average='weighted', labels=np.unique(knn_y_predicted))
    print '*' * 50

def train_cat_feat(X):
    
    y = X.iloc[:, X.columns == 'eviction']
    X = X.iloc[:, X.columns != 'eviction']

    selector = SelectKBest(chi2, k=10).fit(X, y.values.ravel())
    top_ten_feat = selector.get_support(indices=True)
    top_ten = X.iloc[:, top_ten_feat]
    print "Top ten features: "
    print top_ten.columns
    
    print "Training with Feature Selection"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 5)

    select_feature = SelectKBest(chi2, k=100).fit(X_train, y_train)
    rfc_X_train = select_feature.transform(X_train)
    rfc_X_test = select_feature.transform(X_test)
    rfc = RandomForestClassifier(n_estimators=20)                                                                        
    start_time = time.time()                                                                                              
    rfc.fit(rfc_X_train, y_train.values.ravel())                                                                          
    rfc_y_predicted = rfc.predict(rfc_X_test)                                                                           
    end_time = time.time()                                                                                                
    rfc_time = end_time - start_time                                                                                      
    rfc_acc = accuracy_score(y_test, rfc_y_predicted)  
    print "RFR: "                                                                                                         
    print "Acc: ", rfc_acc                                                                                                
    print "Time: ", rfc_time                                                                                              
    print "precision recall fbeta support"                                                                                
    print precision_recall_fscore_support(y_test, rfc_y_predicted, average='weighted', labels=np.unique(rfc_y_predicted))
    print '*' * 50    

    select_feature = SelectKBest(chi2, k=100).fit(X_train, y_train)
    bnb_X_train = select_feature.transform(X_train)
    bnb_X_test = select_feature.transform(X_test)
    bnb = BernoulliNB(alpha=0.41)
    start_time = time.time()
    bnb.fit(bnb_X_train, y_train.values.ravel())
    bnb_y_predicted = bnb.predict(bnb_X_test)
    end_time = time.time()
    bnb_time = end_time - start_time
    bnb_acc = accuracy_score(y_test, bnb_y_predicted)
    print "BNB: "
    print "Acc: ", bnb_acc
    print "Time: ", bnb_time
    print "precision recall fbeta support"
    print precision_recall_fscore_support(y_test, bnb_y_predicted, average='weighted', labels=np.unique(bnb_y_predicted))
    print '*' * 50

    select_feature = SelectKBest(chi2, k=100).fit(X_train, y_train)
    knn_X_train = select_feature.transform(X_train)
    knn_X_test = select_feature.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=45)
    start_time = time.time()
    knn.fit(knn_X_train, y_train.values.ravel())
    knn_y_predicted = knn.predict(knn_X_test)
    end_time = time.time()
    knn_time = end_time - start_time
    knn_acc = accuracy_score(y_test, knn_y_predicted)
    print "KNN: "
    print "Acc: ", knn_acc
    print "Time: ", knn_time
    print "precision recall fbeta support"
    print precision_recall_fscore_support(y_test, knn_y_predicted, average='weighted', labels=np.unique(knn_y_predicted))
    print '*' * 50

def main():
    X = import_data()
    #feature_selection(X)                                                                                             
    find_param(X)                                                                                                     
    #train_cat(X)
    #train_cat_feat(X)

if __name__ == "__main__":
  main()
