__author__ = 'constancedeperrois'

import time
import numpy as np


np.set_printoptions(suppress=True)

n_al = 10

names_algo = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "LDA", "QDA", "Logistic Regression"]


def get_values_array(name1, n_algo, n_networks):

    name = name1 + '.txt'
    current_file = open(name, 'r')

    lines = current_file.readlines()
    length = len(lines)

    graph_found = 0
    graph_used = 0

    values = np.zeros((n_algo, n_networks))

    for i in range(1, length):

        current_line = lines[i].split()
        test = current_line[0]
        if test == "graph":
            graph_found += 1
        else:
            if graph_used < n_networks:
                if len(current_line) == n_algo:
                    print current_line
                    for j in range(0, n_algo):
                        if j == 0:
                            interim = current_line[j]
                            interim1 = interim[1:]
                            interim2 = interim1[:-1]
                        elif j == n_algo-1:
                            interim = current_line[j]
                            interim1 = interim[:-1]
                            interim2 = interim1[:-1]
                        else:
                            interim = current_line[j]
                            interim2 = interim[:-1]
                        values[j, graph_found-1] = float(interim2)
                        print interim2
                    graph_used += 1
                else:
                    print "Incorrect data", i
                    break
    if graph_found == graph_used:
        if graph_found == n_networks:
            print "ok for loading ", name
    else:
        print "needs check up", graph_found, graph_used, n_networks

    np.save(name1, values)
    current_file.close()


def get_kpis(namee, n_algo, n_networks):
    name1 = namee + '.npy'
    values = np.load(name1)

    kpi_values = np.zeros((n_al, 4))    #    kpi values are mean range and standard deviation

    for i in range(0, n_algo):
        current = kpi_values[i, :]
        current_values = values[i, :]
        if len(current_values) == n_networks:
            current[0] = np.mean(current_values)
            current[1] = np.var(current_values)
            current[2] = np.min(current_values)
            current[3] = np.max(current_values)
        else:
            print "error in data"
            break
    kpi_values = np.around(kpi_values, 4)
    print kpi_values

    name2 = namee + 'kpi'
    np.save(name2, kpi_values)


def av_all_kpis(namee, n_al, n_f):

    kpi_values_av = np.zeros((n_al, 4))

    kpi_values_mean = np.zeros((n_al, n_f))
    kpi_values_var = np.zeros((n_al, n_f))
    kpi_values_min = np.zeros((n_al, n_f))
    kpi_values_max = np.zeros((n_al, n_f))

    for i in range(0, n_f):
        name1 = namee + str(i) + 'kpi.npy'
        kpi_values = np.load(name1)
        kpi_values_mean[:, i] = kpi_values[:, 0]
        kpi_values_var[:, i] = kpi_values[:, 1]
        kpi_values_max[:, i] = kpi_values[:, 2]
        kpi_values_min[:, i] = kpi_values[:, 3]

    for i in range (0, n_al):
        kpi_values_av[i, 0] = np.mean(kpi_values_mean[i, :])
        kpi_values_av[i, 1] = np.mean(kpi_values_var[i, :])
        kpi_values_av[i, 2] = np.mean(kpi_values_max[i, :])
        kpi_values_av[i, 3] = np.mean(kpi_values_min[i, :])

    kpi_values_av = np.around(kpi_values_av, 4)

    print "average over ", n_f
    print kpi_values_av
    np.savetxt(namee + 'txt', kpi_values_av)


def main(n_netwww, n_files):

    n_net = n_netwww
    directory = 'final' + str(n_net)

    for i in range(0, n_files):
        name = directory + '/' + 'resultats_learn' + str(n_net/100) + '_' + str(i)
        get_values_array(name, n_al, n_net)
        get_kpis(name, n_al, n_net)

    av_all_kpis(directory + '/' + 'resultats_learn' + str(n_net/100) + '_', n_al, n_files)
    print names_algo


if __name__ == "__main__":
    n_netww = int(raw_input("N_Networks is? "))
    number_files = int(raw_input("Number of files in calculation? "))
    start_time = time.time()
    main(n_netww, number_files)
    print("--- %s seconds ---" % (time.time() - start_time))
