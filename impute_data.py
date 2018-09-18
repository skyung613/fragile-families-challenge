from __future__ import division
import pandas as pd
import numpy as np
import sys
import missingno as msno
import matplotlib.pyplot as plt
from fancyimpute import KNN

def import_data():
    background = pd.read_csv('background.csv', low_memory=False)
    train = pd.read_csv('train.csv', low_memory=False)

    # drop all rows where gpa is NaN in train.csv
    #train = train[train['gpa'].notnull()]

    # drop all rows where eviction is NaN in train.csv
    #train = train[train['eviction'].notnull()]

    # drop all rows where grit is NaN in train.csv
    #train = train[train['grit'].notnull()]

    # drop all rows where material hardship is NaN in train.csv
    # train = train[train['materialHardship'].notnull()]
    
    # drop all rows where material hardship is NaN in train.csv
    #train = train[train['layoff'].notnull()]

    # drop all rows where material hardship is NaN in train.csv
    train = train[train['jobTraining'].notnull()]
    index = train['challengeID']
    
    # drop rows not present in train.csv
    background = background.loc[background['challengeID'].isin(index)]

    train.to_csv('imputed_train_jobTraining.csv', sep=',', index=False)

    return background

def replace_missingness(df):
    
    # replace all negative values with NaN
    print "Replacing missing values with NaN"
    df[df < 0] = np.nan
    
    # replace Missing or Other with NaN
    df[df == 'Missing'] = np.nan
    df[df == 'Other'] = np.nan

    # drop columns with over 75% missingness
    print "Dropping columns with over 75% missingness"
    include_col = []
    theta = .75
    dropped_col = 0
    for col in df.columns:
        missingness = df[col].isnull().sum() / len(df[col])
        if theta >= missingness:
            include_col.append(col)
        else :
            dropped_col += 1
    df = df[include_col]
    print "Dropped ", dropped_col, " columns"

    # find and drop columns that do not contain numeric values
    print "Dropping columns that do not contain numeric values"
    df_subset = df.select_dtypes(exclude=[np.number])
    print "Columns dropped: ", df_subset.columns
    df = df.drop(df_subset.columns, axis=1)    

    # using knn imputation
    print "Running Knn imputation"
    df_imputed_columns = df.columns
    k = 109
    df_imputed = KNN(k=k).complete(df)
    df_imputed = pd.DataFrame(df_imputed)
    #print df_imputed
    df_imputed.columns = df_imputed_columns
    
    # save result to imputed_background.csv 
    df_imputed.to_csv('imputed_background_jobTraining.csv', sep=',', index=False)
    
def main():    
    
    background = import_data()
    replace_missingness(background)

if __name__ == "__main__":
  main()

