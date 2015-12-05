__author__ = 'constancedeperrois'



import numpy as np
import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc



def printcheck(X_train, X_test, y_train, y_test):
    print X_test.shape, y_test.shape
    print X_train.shape, y_train.shape
    print X_test, y_test
    print X_train, y_train



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



def classify(datasets, name, clf):

    lendata = len(datasets)
    done = 0
    number_perfect = 0

    #  iterate over datasets
    for ds in datasets:

        # pre-process dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)

        n = numberofgraphs(y, True)
        print "in learnalgo, number of graphs is ", n
        shape = y.shape[0]

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        all_auc = []

        for i in range(0, n):

            all_auc.append(0)

            a, b = selectkthgraph(i+1, y)
            X_test, y_test = X[shape-a:shape-b], y[shape-a:shape-b]
            X_1 = X[:shape-a]
            X_2 = X[shape-b:]
            y_1 = y[:shape-a]
            y_2 = y[shape-b:]
            X_train, y_train = np.concatenate([X_1, X_2], axis=0), np.concatenate([y_1, y_2], axis=0)

            #printcheck(X_train, X_test, y_train, y_test)

            # Fit the classifier
            clf.fit(X_train, y_train)

            # Predict the last one
            prediction = clf.predict(X_test)

            probas_ = clf.fit(X_train, y_train).predict_proba(X_test)

            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0

            roc_auc = auc(fpr, tpr)

            all_auc[i] = (roc_auc, fpr, tpr)

            print "Classifier", name, "(most likely edge in last graph using training set)"
            print "Prediction", prediction
            print "Real res..", y_test

            if np.array_equal(prediction, y_test):
                print "WOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"
                number_perfect +=1

        done += 1

    if done == lendata:
        print "all done all data found"
        print "number perfect", number_perfect
    else:
        print "something's wrong"

    mean_tpr /= n
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, label='Mean ROC ' + str(name) + ' (area = %0.2f)' % mean_auc, lw=1)

    return all_auc, number_perfect


def filecount(where_to_look):
    return len([f for f in os.listdir(where_to_look) if os.path.isfile(os.path.join(where_to_look, f))])


def get_data(number_ga, lots):

    if number_ga > 0:
        direct_name = 'final' + str(number_ga) +'00/data' + str(number_ga) + '00'
        done = 0
        if lots == 'y':
            total = filecount(direct_name)/2
        else:
            total = 1
        print "total files", total
        rez_name = direct_name + '/resvizzz' + '_' + str(number_ga)

        while done < total:
            name_f = 'features' + str(number_ga) + '_' + str(done) + '.npy'
            name_l = 'label' + str(number_ga) + '_' + str(done) + '.npy'
            data_x = np.load(direct_name + '/' + name_f)
            data_y = np.load(direct_name + '/' + name_l)
            my_data = (data_x, np.array(data_y))
            print name_f, "found"

            if done == 0:
                datasets = [my_data]
            else:
                datasets.append(my_data)
            done += 1

        if total == len(datasets):
            print "all loaded in datasets"
        else:
            print "missing data"

    else:
        data_x = np.load('features.npy')
        data_y = np.load('label.npy')
        my_data = (data_x, np.array(data_y))
        rez_name = 'scoref1' + '_' + str(number_g) + '.txt'

        datasets = [
                    my_data
                    ]

    print len(datasets), "len datasets"
    return datasets, rez_name



def main(number_gg):

    datasets, rez_name = get_data(number_gg, testee)


    #  Compare various classifiers
    names = ["Decision Tree",
             "Random Forest", "AdaBoost"
             ]
    classifiers = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier()
    ]


    for name, clf in zip(names, classifiers):
        auc, perf = classify(datasets, name, clf)

        myfile = open('rocpred' + str(name) + 'samples'+ str(number_g) + '.txt', 'w')
        myfile.write('Algorithm is ' + str(name) + '\n')
        myfile.write('Number of subset is ' + str(number_g) + '\n')
        myfile.write('auc' + str(auc) + '\n' + str(perf))
        myfile.close()

        print "auc", auc
        auc.sort(key=lambda tup: tup[0])
        print auc

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()




if __name__ == "__main__":
    number_g = int(raw_input("How many subsets in sample? "))
    testee = raw_input("lots of tests? y / n ")
    start_time = time.time()
    main(number_g)
    print("--- %s seconds ---" % (time.time() - start_time))











