import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv

data = pd.read_csv('seeds_dataset.txt', delimiter = '\t')

data.columns = ['1', '2', '3', '4', '5', '6', '7', '8']

labels = data['8']
dimensions = data.ix[:,['1', '2', '3', '4', '5', '6', '7']]

def distance(p1,p2):
    return np.sum((p2-p1)**2)

def model(dimensions, labels, new_example):
    comp = []
    pred = []

    for e in range(len(new_example)):
        for i in range(len(dimensions)):
            comp.append(distance(dimensions[i], new_example[e]))
            
        smallest = (np.array(comp)).argmin()

        pred.append(labels[smallest])

        comp = []

    return pred

def folds(labels, dimensions):
    #stratified k-folds CV
    skf = cv.StratifiedKFold(labels, n_folds=10, shuffle = True)
    score = []
    
    for train_index, test_index in skf:
        X_train, X_test = dimensions.ix[train_index], dimensions.ix[test_index]
        Y_train, Y_test = labels.ix[train_index], labels.ix[test_index]
        X_train, X_test = X_train.as_matrix(), X_test.as_matrix()
        Y_train, Y_test = Y_train.as_matrix(), Y_test.as_matrix()
        
        prediction = model(X_train, Y_train, X_test)

        result = [prediction==Y_test]

        score.append(np.sum(result) / len(Y_test))

    score = np.array(score)

    print (score)

    print (np.mean(score))
        
def main():
    folds(labels, dimensions)
    
    
