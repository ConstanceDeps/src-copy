__author__ = 'constancedeperrois'


import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def printcheck(X_train, X_test, y_train, y_test):
    print X_test.shape, y_test.shape
    print X_train.shape, y_train.shape
    print X_test, y_test
    print X_train, y_train

def selectlastgraph(y):
    shape = y.shape[0]
    n = 2      # for index in liste, remove last element
    while n < shape:
        if y[-n] == 1:
            break
        else:
            n += 1
    print "index before last graph", n
    return n - 1


def numberofgraphs(y, test):
    shape = y.shape[0]
    number_graphs = 0
    n = 1
    while n < shape:
        if y[-n] == 1:
            number_graphs += 1
        n += 1
    if test:
        print "number of graphs found in label set", number_graphs
    return number_graphs


def selectkthgraph(k, y):
    num = numberofgraphs(y, False)
    if k > num:
        print "Error, not enough graphs"
        return -1
    shape = y.shape[0]
    alreadyseen = -1
    lastindex = 0
    n = 1
    while n <= shape:
        if y[-n] == 1:
            alreadyseen += 1
            if alreadyseen == k:
                break
            else:
                lastindex = n
                n +=1
        else:
            n += 1
    print "graph", k, "between", n-1, lastindex-1, "back"
    return n-1, lastindex-1

def predict_one_edge(X_train, X_test, y_train, y_test, names, classifiers):

    # iterate over classifiers
    for name, clf in zip(names, classifiers):

        # Fit the classifier
        clf.fit(X_train, y_train)

        # score = clf.score(X_test, y_test)     What is classifier score??
        #print score

        # Predict the last one
        prediction = clf.predict(X_test)

        print "Classifier", name, "one edge"
        print "Prediction", prediction, ", result should be", y_test


def predict_one_graph(X_train, X_test, y_train, y_test, names, classifiers):
    # iterate over classifiers
    rez = []
    for name, clf in zip(names, classifiers):

        # Fit the classifier
        clf.fit(X_train, y_train)

        # Predict the last one
        prediction = clf.predict(X_test)

        score = f1_score(y_test, prediction)
        rez.append(score)

        if np.array_equal(prediction, y_test):
            print "Classifier", name, "(most likely edge in last graph using training set)"
            print "Prediction", prediction
            print "Real res..", y_test
            print "WOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"

        print "Classifier", name, "(most likely edge in last graph using training set)"
        print "Prediction", prediction
        print "Real res..", y_test
    print "f1", rez
    return rez


def main():

    # My data
    data_x = np.load('features.npy')
    data_y = np.load('label.npy')
    my_data = (data_x, np.array(data_y))
    datasets = [
                my_data
                ]
    #print data_y.shape

    choice = 2   # 0 for one edge, 1 for one graph, 2 for cross val.

    """
    Compare various classifiers
    """
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA", "Logistic Regression"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear"),
        SVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA(),
        LogisticRegression()]


    # iterate over datasets
    for ds in datasets:

        # pre-process dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)

        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1)

        if choice == 0:
            X_train, X_test, y_train, y_test = X[:-1], X[-1], y[:-1], y[-1]
            predict_one_edge(X_train, X_test, y_train, y_test, names, classifiers)

        # Predict for an entire graph now.
        if choice == 1:
            n = selectlastgraph(data_y)
            X_train, X_test, y_train, y_test = X[:-n], X[-n:], y[:-n], y[-n:]
            #print X_test.shape, y_test.shape

            rez = predict_one_graph(X_train, X_test, y_train, y_test, names, classifiers)
            print rez

        if choice == 2:
            n = numberofgraphs(y, True)
            shape = y.shape[0]
            file = open('resulats_learn' + str(n) + '.txt', 'w')
            file.write('Shape of feature is' + str(shape) + '\n')

            mes_scores = []

            for i in range(0, n):
                a, b = selectkthgraph(i+1, y)
                X_test, y_test = X[shape-a:shape-b], y[shape-a:shape-b]
                X_1 = X[:shape-a]
                X_2 = X[shape-b:]
                y_1 = y[:shape-a]
                y_2 = y[shape-b:]
                X_train, y_train = np.concatenate([X_1, X_2], axis=0), np.concatenate([y_1, y_2], axis=0)

                #printcheck(X_train, X_test, y_train, y_test)
                rez = predict_one_graph(X_train, X_test, y_train, y_test, names, classifiers)
                print "graph was", i
                file.write('graph number' + str(i) + 'from back end \n')
                file.write(str(rez))
                file.write('\n')

                if i == 0:
                    mes_scores = rez
                else:
                    for j in range(len(mes_scores)):
                        mes_scores[j] = mes_scores[j] + rez[j]

            for i in range(len(mes_scores)):
                mes_scores[i] = mes_scores[i]/n
                file.write(str(mes_scores[i]) + " ")

            print "final scores", mes_scores
            file.close()
            return mes_scores



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))



# Build a classification task

"""
X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

X, y = make_classification(n_features=30, n_redundant=10, n_informative=20,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

"""

# Basic usage of a classifier to predict the value of a test

"""

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA", "Logistic Regression"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LDA(),
        QDA(),
        LogisticRegression()]

"""

