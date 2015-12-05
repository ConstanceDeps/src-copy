__author__ = 'constancedeperrois'

"""

Functions to create the data array.

This returns an array of input data with shape `(n_samples, n_features)`
    and an array of `n_samples` target labels.

"""

import numpy as np
import os

def getmyname_baby(nn, seed_family, seed_simul, baby_number):
    name = 'results' + str(nn) + str(seed_family) + 'baby' + str(baby_number) + '.txtsimul' + str(seed_simul) + '.txt'
    return name

def getmyname_parent(nn, seed_family, seed_simul):
    name = 'results' + str(nn) + str(seed_family) + '.txtsimul' + str(seed_simul) + '.txt'
    return name

def get_number_babies(nn, seed_family, seed_simul, n_simul):
    babies = 0

    start_name = 'results' + str(nn) + str(seed_family) + 'baby'
    wrong_end = 'baby.txtsimul' + str(seed_simul) + '.txt'

    for filename in os.listdir('simres/'):
            if filename.startswith(start_name):
                if not(filename.endswith(wrong_end)):
                    babies += 1
    #print babies
    #print n_simul
    babies = babies / n_simul
    return babies

def get_res_to_list_one_simul(nn, seed_family, seed_simul, baby_number, n1, babytest):
    res = []
    if babytest:
        name = getmyname_baby(nn, seed_family, seed_simul, baby_number)
    else:
            name = getmyname_parent(nn, seed_family, seed_simul)
    myfile = open('simres/' + name, 'r')
    # print myfile
    lines = myfile.readlines()
    length = len(lines)
    for i in range(2, length):
        current_line = lines[i].split()
        test = current_line[0]
        if test.isdigit():
            for n in range(n1):
                number = int(current_line[2 + n])
                res.append(number)
    myfile.close()
    return res

def get_res_to_array(nn, seed_family, seed_simul_learn, baby_number, n1, babytest, n_simul):
    results = []
    for i in range(n_simul):
        #print "current test famille", seed_family, "simul", seed_simul_learn + i, "bb", baby_number
        res = get_res_to_list_one_simul(nn, seed_family, seed_simul_learn + i, baby_number, n1, babytest)
        if i == 0:
            results = np.array([res])
        else:
            results = np.concatenate((results, np.array([res])), axis=0)
            results = np.array([np.sum(results, axis=0)])

    if babytest:
        label = 0
    else:
        label = 1

    results = np.divide(results, n_simul + 0.0)

    #print results
    #print results.shape
    return [results, label]


# THE REFERENCE IS THE DADDY SIMULATION NUMBER 1

def get_feature_one_sibling(nn, seed_family, seed_simul, baby_number, n1, n_simul):
    data_baby = get_res_to_array(nn, seed_family, seed_simul, baby_number, n1, True, n_simul)
    data_daddy = get_res_to_array(nn, seed_family, seed_simul, 0, n1, False, n_simul)

    baby = data_baby[0]
    daddy = data_daddy[0]

    label = data_baby[1]

    feature = np.subtract(baby, daddy)

    #print feature
    #print feature.shape
    return [feature, label]

def get_feature_parent(nn, seed_family, seed_simul, n1, n_simul):
    data_baby = get_res_to_array(nn, seed_family, seed_simul, 0, n1, False, n_simul)
    data_daddy = get_res_to_array(nn, seed_family, seed_simul, 0, n1, False, 1)

    baby = data_baby[0]
    daddy = data_daddy[0]

    label = data_baby[1]

    feature = np.subtract(baby, daddy)
    feature = np.absolute(feature)

    #print feature
    #print feature.shape
    return [feature, label]

def get_feature_one_family(nn, seed_family, seed_simul, n1, n_simul):
    babies = get_number_babies(nn, seed_family, seed_simul, n_simul)
    print "babies", babies
    label = []
    for i in range(1, babies):
        new_feat = get_feature_one_sibling(nn, seed_family, seed_simul, i, n1, n_simul)
        if i == 1:
            feature = new_feat[0]
            #print feature.shape
        else:
            feature = np.concatenate((feature, new_feat[0]), axis=0)
        label.append(new_feat[1])

    parent = get_feature_parent(nn, seed_family, seed_simul, n1, n_simul)
    feature = np.concatenate((feature, parent[0]), axis=0)
    label.append(parent[1])

    #print feature
    #print feature.shape
    #print label
    return [feature, label]

def get_feature_all_families(nn, seed_family_init, seed_simul_init, n1, n_simul, n_graph):
    label = []
    left = n_graph
    current_seed = seed_family_init
    while left > 0:
        #print left, current_seed
        try:
            res = get_feature_one_family(nn, current_seed, seed_simul_init, n1, n_simul)
            label = label + res[1]
            left -= 1
            if left == n_graph -1 :
                features = res[0]
            else:
                features = np.concatenate((features, res[0]), axis=0)
            print "learning graphs last", current_seed, "was found", left, "is left"
        except IOError:
            print "this seed was not found", current_seed
        current_seed += 1

    #print features[0]
    #print "data shape", features.shape
    #print features[0]
    #print label
    return [features, label]

def check_identical_rows(feature):
    for i in range(len(feature)): #generate pairs
        for j in range(i+1, len(feature)):
            if np.array_equal(feature[i], feature[j]): #compare rows
                print (i, j)
    print "done"

def store_data_print_size(feature, label, name_x, name_y):
    print "Data Shape", feature.shape
    if feature.shape[0] == len(label) :
        print "Number of samples", len(label), ", number of features", feature.shape[1]
    else:
        print "Error in data, number of samples does not match"

    np.save(name_x, feature)
    np.save(name_y, label)
    print "Data saved under name", name_x, name_y

def check_right_number(nn, number, seed_init, seed_finale):
    number_families = 0
    current_family = seed_init

    while number_families < number:
        start_name = 'results' + str(nn) + str(current_family)
        found = False
        res = False

        for filename in os.listdir('simres/'):
            if filename.startswith(start_name):
                found = True

        if found:
            number_families += 1
            #print current_family, number_families

        if number_families == number and current_family == seed_finale:
            print "all good !"
            res = True
            break

        current_family += 1

        if current_family > seed_finale:
            print "error"
            break

    print number_families, number, current_family
    return res