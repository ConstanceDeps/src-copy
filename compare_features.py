__author__ = 'constancedeperrois'


import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


def main():

    # My data

    data_x = np.load('features.npy')
    data_y = np.load('label.npy')

    my_data = (data_x, np.array(data_y))

    X, y = my_data[0], np.array(my_data[1])

    #Compare the importance of various features in discriminating true and false examples

    n_features = X.shape[1]

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(10):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    #return(n_features, indices)

    # Plot the feature importances of the forest
    plt.figure(figsize=(27, 9))

    plt.title("Feature importances")
    plt.bar(range(n_features), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(n_features), indices)
    plt.xlim([-1, n_features])
    plt.show()

if __name__ == "__main__":
    main()