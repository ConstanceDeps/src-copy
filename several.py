__author__ = 'constancedeperrois'

import numpy as np
import time
import itertools
import os

n = 1   # number of 100 packages in sample size

n_possible = 12    # total number of 100 samples to choose from

def which_items(n_graph, n_poss):
    liste_1 = []
    for i in range(1, n_poss+1):
        liste_1.append(i)
    liste = list(itertools.combinations(liste_1, n_graph))
    return liste


def main():

    rez = which_items(n, n_possible)
    li = len(rez)
    name = 'data' + str(n) + '00'

    try:
        os.stat(name)
    except:
        os.mkdir(name)

    for j in range(0, li):
        liste_direct = rez[j]
        print "list used", liste_direct
        done = 0

        for i in range(0, n):
            current_direct_num = liste_direct[i]

            name_directory = 'res_final/' + str(current_direct_num) + '/'

            if done == 0:
                data_x = np.load(name_directory+ 'features.npy')
                data_y = np.array(np.load(name_directory + 'label.npy'))
                done = 1

            else:
                new_f = np.load(name_directory+ 'features.npy')
                data_x = np.concatenate((new_f, data_x), axis=0)
                new_l = np.array(np.load(name_directory+ 'label.npy'))
                data_y = np.concatenate((new_l, data_y), axis=0)
                done +=1

        print "has been done", done, "subset", j

        np.save(name + '/' +'features' + str(n) + '_' + str(j), data_x)
        np.save(name + '/' + 'label' + str(n) + '_' + str(j), data_y)
        print "Data saved (lots)", n

        print "Data Shape", data_x.shape
        if data_x.shape[0] == len(data_y) :
            print "Number of samples", len(data_y), ", number of features", data_x.shape[1]
        else:
            print "Error in data, number of samples does not match"


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))