__author__ = 'constancedeperrois'

import Graph_Plot
import my_code_simul
import start_my_code
import time
import learning
import learn_algo
import compare_features

def myfunctionsuperkey(test, graph, simul, learn):

    #  Commands for the fit of the nodes
    if test == -1:
        Graph_Plot.check_right_number_graphs(start_my_code.NUMBER_OF_NODES, start_my_code.NUMBER_OF_GRAPHS, start_my_code.INIT_SEED, 1483)

    if test == 1:
        Graph_Plot.calculatemystats(start_my_code.NUMBER_OF_NODES, start_my_code.NUMBER_OF_TIER_1, start_my_code.LENGTH, start_my_code.SIG_FACT, 500)
        Graph_Plot.findmyfactor(start_my_code.NUMBER_OF_NODES, start_my_code.NUMBER_OF_TIER_1, start_my_code.LENGTH, 500)

    #  Create a new random graph
    if graph == 1:
        G = Graph_Plot.build_my_random_net(start_my_code.NUMBER_OF_NODES, start_my_code.NUMBER_OF_TIER_1, start_my_code.LENGTH, start_my_code.SIG_FACT, start_my_code.GAMMA, start_my_code.FACTOR,  True)
        G.writemygraph(Graph_Plot.getnamenodes(start_my_code.INIT_SEED), Graph_Plot.getnameedges(start_my_code.INIT_SEED), start_my_code.INIT_SEED)
        Graph_Plot.plotpretty(G.graph, G.nodelayer,  Graph_Plot.getmygraphname(start_my_code.INIT_SEED))

    #  Load an existing graph (Careful with existing disruptions!, txt file should be topological)
    if graph == 2:
        G = Graph_Plot.addmygraph(Graph_Plot.getnamenodes(start_my_code.INIT_SEED), Graph_Plot.getnameedges(start_my_code.INIT_SEED))
        #Graph_Plot.plotpretty(G.graph, G.nodelayer,  Graph_Plot.getmygraphname(start_my_code.INIT_SEED))

    #  Start a simulation
    if simul == 1:
        my_code_simul.StartMySimul(start_my_code.TIME_SIMUL, G, True, True, start_my_code.INIT_SEED_SIMUL)

    #  Start a lot of simul
    if simul == 2:
        my_code_simul.startall(start_my_code.TIME_SIMUL, G, True, True, start_my_code.INIT_SEED_SIMUL, start_my_code.NUMBER_OF_SIMUL)

    #  Compute statistics with a lot of simulations
    if test == 2:
        my_code_simul.analysemytrack("500trackdis1234.txt")
        my_code_simul.stats_lot(start_my_code.NUMBER_OF_SIMUL)

    #  Generate a lot of graphs for the learning and check for duplicates
    if graph == 3:
        Graph_Plot.generatealotofgraphs(start_my_code.NUMBER_OF_GRAPHS, True, start_my_code.INIT_SEED, 0)

    #  Get the babies
    if graph == 4:
        Graph_Plot.getmybabies(start_my_code.NUMBER_OF_GRAPHS, start_my_code.INIT_SEED)
        Graph_Plot.getallsiblings(start_my_code.NUMBER_OF_GRAPHS, start_my_code.INIT_SEED)

    #  Start simulations for all the graphs in the directory graphdata. Stores all in.
    if simul == 3:
        left = start_my_code.NUMBER_OF_GRAPHS
        current_seed_simul3 = start_my_code.INIT_SEED
        while left > 0:
            for i in range(start_my_code.NUMBER_OF_SIMUL):
                print "current simulation", i
                test = my_code_simul.simulforallbabies(start_my_code.TIME_SIMUL, start_my_code.INIT_SEED_SIMUL + i, start_my_code.NUMBER_OF_NODES, 1, current_seed_simul3)
            if test == 0:
                current_seed_simul3 += 1
            if test == 1:
                current_seed_simul3 +=1
                left -= 1
            if test == -10:
                print "Erreur de data graphs"
                break

    #   Learning, preprocessing of data
    if learn == 1:

        #learning.check_right_number(start_my_code.NUMBER_OF_NODES, start_my_code.NUMBER_OF_GRAPHS, start_my_code.INIT_SEED, 1483)
        #learning.get_feature_one_family(start_my_code.NUMBER_OF_NODES, start_my_code.INIT_SEED, start_my_code.INIT_SEED_SIMUL, start_my_code.NUMBER_OF_TIER_1, start_my_code.NUMBER_OF_SIMUL)
        data = learning.get_feature_all_families(start_my_code.NUMBER_OF_NODES, start_my_code.INIT_SEED, start_my_code.INIT_SEED_SIMUL, start_my_code.NUMBER_OF_TIER_1, start_my_code.NUMBER_OF_SIMUL, start_my_code.NUMBER_OF_GRAPHS)
        learning.check_identical_rows(data[0])
        learning.store_data_print_size(data[0], data[1], 'features', 'label')

    if learn == 2:

        learn_algo.main(1)

    if learn == 3:

        compare_features.main()

    print "done", test, graph, simul, learn

    """
    Tests
    """
    #Graph_Plot.howmanyedges(start_my_code.GAMMA, start_my_code.FACTOR, start_my_code.NUMBER_OF_NODES)

    #print G.graph.edges(12)
    #print G.myedges[2].origin


    #my_code_simul.computeavproba()
    #my_code_simul.computeavtime()

    #print Graph_Plot.aretheseedgesthesame(Graph_Plot.getnameedges(1234), Graph_Plot.getnameedges(1235))

    #G = Graph_Plot.addmygraph(Graph_Plot.getnamenodes(start_my_code.INIT_SEED), Graph_Plot.getnameedges(start_my_code.INIT_SEED))
    #Graph_Plot.removeoneedge(G)

    #learning.get_res_to_array(start_my_code.NUMBER_OF_NODES, 1235, 5678, 1, start_my_code.NUMBER_OF_TIER_1, True, start_my_code.NUMBER_OF_SIMUL)

    #learning.get_feature_one_sibling(start_my_code.NUMBER_OF_NODES, 1234, 5678, 1, start_my_code.NUMBER_OF_TIER_1, start_my_code.NUMBER_OF_SIMUL)

    #learning.get_feature_parent(start_my_code.NUMBER_OF_NODES, 1234, 5678, start_my_code.NUMBER_OF_TIER_1, start_my_code.NUMBER_OF_SIMUL)

    #learning.get_number_babies(start_my_code.NUMBER_OF_NODES, 1234, 5678)



if __name__ == "__main__":
    start_time = time.time()
    myfunctionsuperkey()
    print("--- %s seconds ---" % (time.time() - start_time))
