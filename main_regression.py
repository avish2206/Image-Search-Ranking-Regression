## Solution.py
# Author: Aditya Vishwanathan (aditya_22@hotmail.com)
# Linear regression model for image search scores.
# Run using:
# python3 solution.py <path_to_data.csv>

# Packages to import
import os
import argparse
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# calcScore: calculates _score_ which is the clicks scaled by the min-max of that particular query
# input:  df - must contain 'query' and 'clicks'
# output: df - returns with additional _score_ column
def calcScore(df):
    groups_min = df.groupby('query').min().to_dict()
    groups_max = df.groupby('query').max().to_dict()
    min_vals = groups_min['clicks']
    max_vals = groups_max['clicks']
    df['_score_'] = (df['clicks'] - df['query'].map(min_vals)) / (df['query'].map(max_vals) - df['query'].map(min_vals))
    return df

# EVD: Performs Eigenvalue Decomposition of matrix to identify collinearity
# inputs: features   - must be np.matrix
#         eig_thresh - threshold to identify "small" eigenvalues
# output: to_remove - set of collinear features to remove 
def EVD(features,eig_thresh):
    # calculate correlation matrix and eigenvalues/eigenvectors
    corr = np.corrcoef(features,rowvar=0)
    w, v = np.linalg.eig(corr)
    
    # find small eigenvalues (<1e-3) and remove features which possess high coefficients in corresponding eigenvectors
    # note: this indicates collinearity, and so we delete all but one feature from the set of linearly dependent features.
    to_remove = set([])
    for idx, eigval in enumerate(w):
        if eigval < eig_thresh:
            linearities = np.where(abs(v[:,idx]) > eig_thresh)
            for i in range(1,len(linearities[0])):
                to_remove.add(linearities[0][i]) 
    return to_remove


# getKBest: Finds K-best features using chi-squared univariate analysis 
# inputs: X - training data
#         y - output data
#         K - number of features to output (<= X.shape[1])
# Output: list of K best features
def getKBest(X,y,K):
    if K=='all' or K>X.shape[1]:
        return list(X.columns)
    bestfeatures = SelectKBest(score_func=chi2, k=K)   
    fit = bestfeatures.fit(X,y)
    scores = pd.DataFrame(fit.scores_)
    columns = pd.DataFrame(X.columns)
    df = pd.concat([columns,scores],axis=1)
    df.columns = ['Specs','Score']  
    return list(df.nlargest(K,'Score').Specs.values) 


# processData: processes original data and obtains final set of features and scores 
# inputs: df - original data frame
#         eig_thresh - eigenvalue threshold for EVD 
#         K - number of features to extract using chi-sq analysis
# Output: final data frame
def processData(df,eig_thresh,K):
    # remove media_id column
    df.drop('media_id',axis=1,inplace=True)

    # remove features with NANs (how='all' should be used, but doesn't drop correctly.)
    df.dropna(axis=1,inplace=True,how='any')

    # transform boolean variables into 0/1 and remove features with std=0
    for column in df:
        if df[column].dtype == 'bool':
            df[column] = df[column]*1
        if df[column].dtype != 'object' and df[column].values.std() == 0:
            df = df.drop(column,axis=1)

    # calculate _score_ and set as output (y)
    y = calcScore(df.loc[:,['query','clicks']]).loc[:,'_score_']
    
    # define features matrix and implement min-max scaling
    X = df.iloc[:,:-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    X.iloc[:,1:] = min_max_scaler.fit_transform(X.iloc[:,1:])
    features = np.array(X.iloc[:,1:].values)

    # perform Eigenvalue Decomposition to remove collinear features
    to_remove = EVD(features,eig_thresh)
    features = np.delete(features,list(to_remove),axis=1)
    X.drop(X.columns[[x+1 for x in list(to_remove)]],axis=1,inplace=True)

    # extract best K features (NOTE: unknown data type error if I use y(float64), so use clicks(int) instead)
    Kbest = getKBest(X.iloc[:,1:],df['clicks'],K)
    Kbest.insert(0,'query')
    X = X.loc[:,Kbest]
    
    # concatente to return final dataframe
    return pd.concat([X, y], axis=1)
    

# runRegression: runs linear regression and prints results to stdout
# inputs: X - features 
#         y - eigenvalue threshold for EVD 
#         test_size - proportion of data to save for testing
#         ksplits - number splits for k-fold cross validation
def runRegression(X,y,test_size,ksplits):
    # split data into test, training
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)

    # get dummy variables for query
    X_train_dummies = pd.get_dummies(X_train)
    X_test_dummies  = pd.get_dummies(X_test)

    # K-fold cross validation on the training set
    regr = linear_model.LinearRegression()
    cv = KFold(n_splits=ksplits, shuffle=True)
    scores = []
    for train_index, test_index in cv.split(X_train_dummies):
        X_tr,y_tr,X_te,y_te = X_train_dummies.iloc[train_index,:], y_train.iloc[train_index], X_train_dummies.iloc[test_index,:], y_train.iloc[test_index]
        regr.fit(X_tr, y_tr)
        scores.append(regr.score(X_te,y_te)) # track accuracy of model across splits
      
    # calculate mean accuracy (r-squared)
    accuracy = np.mean(scores)

    # predict test set (ensure inclusive [0,1] range in data)
    y_pred = regr.predict(X_test_dummies)
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > 1] = 1

    # print results to stdout
    count = 0
    for idx,row in X_test.iterrows():
        print(row['query']+'\t%f\t%f' %(y_test.get(idx), y_pred[count]))
        count+=1


# main function
def main():
    # parse input argument to extract data file
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs=1, help="Path to data in .csv format")
    args = parser.parse_args()
    path = args.path[0]
    if not os.path.exists(path):
        parser.error(f"Invalid path {path}")
        
    # import data
    df = pd.read_csv(path)
    
    # process data and extract final set of scaled features(X) and scores(y) 
    # note eig_thresh is a threshold value used in EVD, K is the number 
    # of features selected in getKBest ('all' for no filtering)
    eig_thresh = 1e-3
    K = 10 # K=10 was found to be the most accurate through trial & error
    df = processData(df,eig_thresh,K)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    
    # _features_ and _scores_ defined:
    # _features_ = ['f9', 'f11', 'f29', 'f19', 'f14', 'f10', 'f4', 'f28', 'f15', 'f1'] for K=10
    # _scores_ is clicks which are min-max scaled with respect to each qury
    _features_ = list(X.iloc[:,1:].columns)
    _scores_   = y
    
    # run learning algorithm
    # note test_size is data saved for final testing and output, ksplits is for kfold cross-validation
    test_size = 0.1
    ksplits = 10
    runRegression(X,y,test_size,ksplits)


if __name__ == "__main__":
    main()
