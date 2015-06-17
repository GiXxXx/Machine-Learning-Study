from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

def model(features, label):
    plength = features[:, 2]
    # use numpy operations to get setosa features
    is_setosa = (label == 0)
    # This is the important step:
    # Give the maximum Petal Length of an Iris Setosa
    max_setosa =plength[is_setosa].max()
    #Give the minimum Length of an Iris Setosa
    min_non_setosa = plength[~is_setosa].min()

    features = features[~is_setosa]
    label = label[~is_setosa]
    is_virginica = (label == 2)

    #initial accuracy, make it any negative number
    best_acc = -1.0

    #if an array has n rows and m columns, array.shape = (n, m) => array.shape(0/1) = n/m
    for fi in range(features.shape[1]):
        #generate thresh for all possible features ie. Pedal length, sepal length
        thresh = features[:, fi].copy()
        thresh.sort()
        #Now test all threshold, if they are not the thresholds, update the value
        for t in thresh:
            pred = (features[:,fi] > t)
            acc = (pred == is_virginica).mean()
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
    result = [best_fi, best_t, best_acc]
    return result

def main():
    # We load the data with load_iris from sklearn
    data = load_iris()
    features = data['data']
    feature_names = data['feature_names']
    target = data['target']
    for t,marker,c in zip(range(3),">ox","rgb"):
    # We plot each class on its own to get different colored markers
        plt.scatter(features[target == t,0],
        features[target == t,1],
        marker=marker,
        c=c)

    label = data['target']

    virginica = features[label == 2]

    full = data['data']

    case = full[label > 0]

    #Leave-One-Out-Cross-Validation
    for i in range (len(features)):
        error = 0.0
        training = np.ones(len(features), bool)
        training[i] = False
        testing = ~training
        result = model(features[training], label[training])

        for e in features[testing]:
            if (e[result[0]] > result[1] and label[testing] != 2):
                error += 1

        print(error)

    return error

    #k-fold CV: for i in range k.   length/k as range->test case, remaining as training  k times
    #leave-p-out CV: require nCp times

    
                
                
        
        
        

