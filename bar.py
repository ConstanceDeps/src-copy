__author__ = 'constancedeperrois'

import numpy as np
import time
import os

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt




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

    ratios = [0, 0, 0, 0, 0]

    #  iterate over datasets
    for ds in datasets:

        # pre-process dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)

        n = numberofgraphs(y, True)
        print "in learnalgo, number of graphs is ", n
        shape = y.shape[0]

        for i in range(0, n):

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

            leng = len(y_test)
            ratio = 0.0
            for j in range(0, leng):
                if y_test[j] == 0:
                    if prediction[j] == y_test[j]:
                        ratio += 1.0
                else:
                    if prediction[j] == y_test[j]:
                        ratio += 50.0
            leng += 50.0
            ratio = ratio / leng
            print ratio

            if ratio < 0.2:
                ratios[0] += 1
            elif ratio < 0.4:
                ratios[1] += 1
            elif ratio < 0.6:
                ratios[2] += 1
            elif ratio < 0.8:
                ratios[3] += 1
            else:
                ratios[4] +=1

            #print "Classifier", name, "(most likely edge in last graph using training set)"
            #print "Prediction", prediction
            #print "Real res..", y_test

            if np.array_equal(prediction, y_test):
                print "WOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO"
                number_perfect +=1

        done += 1

    if done == lendata:
        print "all done all data found"
        print "number perfect", number_perfect
    else:
        print "something's wrong"

    return ratios, number_perfect


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
    names = [#"Decision Tree" #,
            #Random Forest" ,
              "AdaBoost"
             ]
    classifiers = [
        #DecisionTreeClassifier() #,
       # RandomForestClassifier() #,
        AdaBoostClassifier()
    ]

    alphab = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    pos = np.arange(len(alphab))
    width = 0.3     # gives histogram aspect to the bar diagram

    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(alphab)

    for name, clf in zip(names, classifiers):
        rat, perf = classify(datasets, name, clf)

        frequencies = rat
        if name == "Decision Tree":
            plt.bar(pos, frequencies, width, color='r')
        elif name == "Random Forest":
            plt.bar(pos, frequencies, width, color='g')
        else:
            plt.bar(pos, frequencies, width, color='b')

    plt.xlim(pos.min(), pos.max()+width)
    plt.show()


if __name__ == "__main__":
    number_g = int(raw_input("How many subsets in sample? "))
    testee = raw_input("lots of tests? y / n ")
    start_time = time.time()
    main(number_g)
    print("--- %s seconds ---" % (time.time() - start_time))




