__author__ = 'constancedeperrois'

import matplotlib.pyplot as plt
import networkx as nx
import start_my_code
import scipy
from scipy import special
import numpy
from graphviz import Digraph
import filecmp
import os
import copy

class Supplier(object):
    supplierCount = 0

    def __init__(self, label, state, rating, lead_time):
        self.rating = rating
        self.label = label
        self.lead_time = lead_time  # order time, average time between dispatch of products.
        Supplier.supplierCount += 1
        self.state = state  # true means it is disrupted
        self.delay = 0  # value of the delay at this supplier

    def displayCount(self):
        print "Total Suppliers", Supplier.supplierCount

    def displaySupplier(self):
        print "Label", self.label, "Rating", self.rating

    def createProblem(self, delay):
        self.state = True
        self.delay = delay

    def randomProblem(self):
        duration = numpy.random.poisson(start_my_code.LAMBDA)
        #  print "duration", duration
        self.createProblem(duration)
        return max(duration, 1)

    def writemynode(self, name):
        file_node = open('graphdata/' + name, 'a')
        lines_text = ["Supplier\n", str(self.label), "\n", str(self.state), "\n", str(self.rating), "\n",
                      str(self.lead_time), "\n"]
        file_node.writelines(lines_text)
        file_node.close()


class Manufacturer(Supplier):
    def __init__(self):
        Supplier.__init__(self, 0, False, 0, 0)

class Edge:
    def __init__(self, node1, node2, part):
        self.origin = node1
        self.destination = node2
        self.part = part

class SupplyNetwork(object):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.mynodes = []
        self.myedges = []
        self.nodelayer = []
        self.seed = 0

    # Functions to add items to a graph

    def addSupplier(self, supplier):
        self.mynodes.append(supplier)
        self.graph.add_node(supplier.label)
        self.mynodes.sort(key=lambda item: item.label)

    def addEdge(self, edge):
        self.graph.add_edge(edge.origin.label, edge.destination.label)
        self.myedges.append(edge)

    # Function to write an existing graph in a txt file.

    def writemygraph(self, namenode, nameedge, seed):
        file = open('graphdata/' + namenode, 'w')
        file.write(str(self.nodelayer))
        file.write('\n')
        file.close()
        file_e = open('graphdata/' + nameedge, 'w')
        file_e.write("edges\n")
        file_e.write(str(seed) + '\n')
        file_e.close()
        nodes = self.mynodes
        for i in range(0, len(nodes)):
            node = nodes[i]
            node.writemynode(namenode)
        for i in range(0, len(self.myedges)):
            edge = self.myedges[i]
            file_edge = open('graphdata/' + nameedge, 'a')
            lines_text = ["Edge\n", str(edge.origin.label), "\n", str(edge.destination.label), "\n", str(edge.part),
                          "\n"]
            file_edge.writelines(lines_text)
            file_edge.close()

    # Functions to create a graph from a txt file

    def addmynodes(self, namenode):
        file_n = open('graphdata/' + namenode, 'r')
        lines = file_n.readlines()
        length = len(lines)
        for i in range(0, length):
            if lines[i].startswith("Supplier"):
                if lines[i + 2] == "True":
                    boolean = True
                else:
                    boolean = False
                supplier = Supplier(int(lines[i + 1]), boolean, int(lines[i + 3]), int(lines[i + 4]))
                self.addSupplier(supplier)
            if lines[i].startswith("Manufacturer"):
                manuf = Manufacturer()
                self.addSupplier(manuf)
        file_n.close()

    def addmyedges(self, nameedges):
        file_n = open('graphdata/' + nameedges, 'r')
        lines = file_n.readlines()
        length = len(lines)
        for i in range(0, length):
            if lines[i].startswith("edges"):
                self.seed = int(lines[i + 1])
            if lines[i].startswith("Edge"):
                origin = int(lines[i + 1])
                dest = int(lines[i + 2])
                part = int(lines[i + 3])
                for i in range(0, len(self.mynodes)):
                    node = self.mynodes[i]
                    if node.label == dest:
                        node_d = node
                    if node.label == origin:
                        node_o = node
                edge = Edge(node_o, node_d, part=part)
                self.addEdge(edge)
        file_n.close()

    # Function to add a disruption to a graph

    def createMyDis(self, index, delay):
        nodes = self.mynodes
        for n in nodes:
            if n.label == index:
                n.createProblem(delay)

    def createRandomDis(self, number, test):
        nnodes = len(self.mynodes)
        index = -1
        pb = -1
        for i in range(0, number):
            index = numpy.random.randint(1, nnodes)
            if test:
                print index, "has a spontaneous dis"
            for n in self.mynodes:
                if n.label == index:
                    pb = n.randomProblem()
        return [index, pb]

    def copymynetw(self):

        netw_baby = SupplyNetwork()

        graph_baby = self.graph.copy()
        netw_baby.graph = graph_baby

        netw_baby.myedges = copy.deepcopy(self.myedges)
        netw_baby.mynodes = copy.deepcopy(self.mynodes)
        netw_baby.nodelayer = copy.deepcopy(self.nodelayer)
        netw_baby.seed = copy.deepcopy(self.seed)

        return netw_baby

    def removeoneedge(self):
        netw_baby = self.copymynetw()
        leng = len(netw_baby.myedges)

        int = numpy.random.randint(0, leng)
        edge = netw_baby.myedges[int]
        while edge.destination.label == 0:
            int = numpy.random.randint(0, leng)
            edge = netw_baby.myedges[int]

        netw_baby.myedges.pop(int)
        netw_baby.graph.remove_edge(edge.origin.label, edge.destination.label)

        plotpretty(self.graph, self.nodelayer, 'essaib' + str(self.seed))
        plotpretty(netw_baby.graph, netw_baby.nodelayer, 'essaib' + str(netw_baby.seed)+"bb")
        print "edge removed", edge.origin.label, edge.destination.label

        namenode = getnamenodesbaby(netw_baby.seed)
        nameedge = getnameedgesbaby(netw_baby.seed)
        netw_baby.writemygraph(namenode, nameedge, netw_baby.seed)

        return netw_baby

    def writesiblings(self):
        siblings = 0
        baby_graph = self.graph
        seed = self.seed
        for n in baby_graph.nodes_iter():
            for m in baby_graph.nodes_iter():
                if not(baby_graph.has_edge(n, m)) and m*n > 0 and n != m:
                    #print "new edge", n, m
                    bro = self.copymynetw()
                    siblings += 1
                    noden = returnmynode(n, self.mynodes)
                    nodem = returnmynode(m, self.mynodes)
                    edge = random_edge(noden, nodem)
                    bro.addEdge(edge)
                    namenode = getnamenodesbro(seed, siblings)
                    nameedge = getnameedgesbro(seed, siblings)
                    bro.writemygraph(namenode, nameedge, seed)
        print "siblings", siblings

# Function to create a new graph from txt files (one for edges and one for nodes).

def addmygraph(namenode, nameedge):
    g = SupplyNetwork()
    g.addmynodes(namenode)
    g.addmyedges(nameedge)
    g.nodelayer = []
    file = open('graphdata/' + namenode, 'r')
    file.read(1)
    char = file.read(1)
    while char != 'STOP':
        if char == ',' or char == ' ':
            char = file.read(1)
        elif char == ']':
            char = 'STOP'
        else:
            g.nodelayer.append(int(float(char)))
            char = file.read(1)
    file.close()
    return g

# Functions to create a random graph

# Generate a random supplier and a random edge

def random_supplier(label):
    a = numpy.random.random()
    if a < start_my_code.RATING_1:
        rating = 1
    elif a < start_my_code.RATING_1 + start_my_code.RATING_2:
        rating = 2
    else:
        rating = 3
    lead_time = int(numpy.around(numpy.random.triangular(start_my_code.LOW, start_my_code.MEAN, start_my_code.HIGH)))
    supplier = Supplier(label, False, rating, lead_time)  # Default random suppliers are not disrupted.
    return supplier

def random_edge(origin, destination):
    a = numpy.random.random()
    if a < start_my_code.CRITIC_1:
        critic = 1
    elif a < start_my_code.CRITIC_1 + start_my_code.CRITIC_2:
        critic = 2
    else:
        critic = 3
    edge = Edge(origin, destination, critic)
    return edge

# Start building the graph

def build_skeleton(N_nodes):  # Add ann the nodes to the graph but no edges.
    netw = SupplyNetwork()
    manuf = Manufacturer()
    netw.mynodes.append(manuf)
    for i in range(1, N_nodes):
        supplier = random_supplier(i)
        netw.addSupplier(supplier)
    netw.mynodes.sort(key=lambda item: item.label)
    return netw

def build_first_tier(netw, N_first_tier):  # Add edged between the manufacturer and its tier-1 suppliers.
    for i in range(0, N_first_tier):
        edge = random_edge(netw.mynodes[i + 1], netw.mynodes[0])
        netw.addEdge(edge)

    return netw

# Functions to calculate sigma and sort the nodes in the layers.

def find_sigma(N_nodes, N_first_tier, mu):
    # Find the sigma of the gaussian distribution for nodes in the network. Do two different estimate and take the mean.

    # Estimate with the first tier
    proba_middle = (N_nodes - 2.0 * N_first_tier - 2.0) / N_nodes
    erffactor2 = scipy.special.erfinv(proba_middle)
    n_factor = numpy.sqrt(2) * erffactor2
    sigma = (mu - 1.5) / n_factor
    if sigma <0 :
        print "ERROR with values entered (sigma calculation)"
    else:
        return sigma


def sort_my_nodes_per_layer(N_nodes, N_first_tier, max_length, sig_fact, check):
    aa = []
    nodes_per_layer = []
    for i in range(0, max_length):
        nodes_per_layer.append(1)  # No layer can be empty.
    nodes_per_layer[1] = N_first_tier

    mu = sig_fact * max_length / 2.0
    sigma = find_sigma(N_nodes, N_first_tier, mu)
    print "mu", mu, "sigma", sigma

    # Distribute all the remaining nodes in the table. If a node falls in layer 0 or 1 just drew it again.
    for i in range(0,
                   N_nodes - N_first_tier - max_length + 1):  # Number of nodes left to add to the graph (manuf is a node)
        a = numpy.random.normal(mu, sigma)
        aa.append(a)
        while a < 1.5 or a > max_length - 0.5:  # If layer 0 or 1 then draw again.
            a = numpy.random.normal(mu, sigma)
        myint = int(numpy.around(a))
        nodes_per_layer[myint] += 1

    if check:
        print "sigma", sigma
        print "nodes per layer", nodes_per_layer
        for i in nodes_per_layer:
            print "% of node in layer i", (i + 0.0) / N_nodes
        #plt.hist(aa, bins=max_length)
        #plt.plot(nodes_per_layer)
        # plt.show()

    return nodes_per_layer

# Functions to fit the graph with the right parameters, and find s.

def calculatemystats(N_nodes, N_first_tier, max_length, sig_fact, occur):
    test = []
    for i in range(max_length - 2):
        test.append(0)
    for i in range(occur):
        res = sort_my_nodes_per_layer(N_nodes, N_first_tier, max_length, sig_fact, False)
        for j in range(max_length - 2):
            test[j] += res[j + 2]
    for i in range(max_length - 2):
        test[i] = (test[i] + 0.0) / (occur * (N_nodes - N_first_tier - 1))
    print test
    return test

def findmyfactor(N_nodes, N_first_tier, max_length, occur):
    res = []
    val = []
    good_factor = 0
    good_val = 1000
    for i in range(1, 50):
        item = 1.0 + i / 100.0
        # print item
        t = calculatemystats(N_nodes, N_first_tier, max_length, item, occur)
        val.append(t)
        valll = max(abs(t[0] - start_my_code.TIER2), abs(t[1] - start_my_code.TIER3), abs(t[2] - start_my_code.TIER4))  # Alexandra's frequency per tier)
        res.append(valll)
        if valll < good_val:
            good_val = valll
            good_factor = item
    print good_factor, good_val

# Build the rest of the graphs with the edges

def howmanyedges(gamma, factor):  # The number returned has to be above 1

    # Max degree is 40. Frequencies is the table of the cumulated distribution function.
    frequencies = [1]
    freq = 0
    for i in range(2, 41):
        fact = factor * numpy.power(i+0.0, -gamma+0.0)
        freq += fact
        frequencies.append(fact)  #  Because data layout from Alexandra's paper.
    for i in range(len(frequencies)):
        frequencies[i] = frequencies[i]
    a = numpy.random.random()
    num = 0
    print frequencies
    for i in range(0, 40):
        if a < frequencies[i]:
            num += 1
    # TODO power law per tier
    # print frequencies
    return max(num, 1)  # Because of the regression there is a risk that is less than 1.

def addmyedges(nodes_per_layer, netw, max_length, gamma, factor, N_nodes, check):
    nodes_below = 1
    nodes_just_below = nodes_per_layer[0]
    total_edges = nodes_per_layer[1]

    for l in range(1, max_length):   # l is the tier
        #print "layer", l
        #print max_length
        current_nodes = nodes_per_layer[l]  # Nodes in the layer we are working on
        for n in range(0, current_nodes):  # n is the n-th node of the current layer.
            label = nodes_below + n  # label of the node we are currently working on
            # How many edges going out
            n_edges = howmanyedges(gamma, factor)
            #print "ed", n_edges

            # The first edge is to someone from the lower layer.
            dest_int = numpy.random.randint(1, nodes_just_below+1)
            dest = nodes_below - nodes_just_below + dest_int - 1

            test = netw.graph.has_edge(label, dest)
            if test is False:
                edge = random_edge(netw.mynodes[label], netw.mynodes[dest])
                netw.addEdge(edge)
                total_edges += 1

            # If there are more edges, then they may be connected to a supplier from : same tier or tier+1 or tier+2
            for e in range(1, n_edges):
                # Choose the layer.
                if (n == 0 and l !=(max_length-1)) or l == 1:  # First node of the layer or first cannot be same layer.
                    layer = 1
                    #print "aaa"
                elif l == (max_length-1):
                    #print "bbbb"
                    layer = -1  # no additional edges
                else:
                    #print "ccc"
                    layer = numpy.random.randint(0, 2)
                #print "layerrrrr", layer

                if layer == 0:  # Edge connected to the same layer, n possibilities.
                    dest_int = numpy.random.randint(0, n)
                    dest = nodes_below + dest_int  # Label of the destination
                    if dest >= N_nodes:
                        print "erreur destination 1", label

                if layer == 1:  # Edge connected to the layer above (if tier n, connected n+1)
                    if nodes_per_layer[l+1] > 1:
                        dest_int = numpy.random.randint(1, nodes_per_layer[l+1])
                    else:
                        dest_int = 1
                    dest = nodes_below + current_nodes + dest_int - 1  # -1 pour le premier
                    if dest >= N_nodes:
                        print "erreur destination 2e", label, dest, nodes_below, current_nodes, dest_int

                test = netw.graph.has_edge(label, dest)
                if test is False:
                    edge = random_edge(netw.mynodes[label], netw.mynodes[dest])
                    netw.addEdge(edge)
                    total_edges += 1

        nodes_below += current_nodes
        nodes_just_below = current_nodes

    if check:
        print "Total number of edges", total_edges
    return netw

def build_my_random_net(N_nodes, N_first, max_length, sig_fact, gamma, factor, check):
    netw = build_skeleton(N_nodes)
    build_first_tier(netw, N_first)

    list = sort_my_nodes_per_layer(N_nodes, N_first, max_length, sig_fact, check)

    net = addmyedges(list, netw, max_length, gamma, factor, N_nodes, check)
    net.nodelayer = list
    return net

# Plot the graph in pdf

def plotpretty(graph, liste, name):
    dot = Digraph(comment=name)
    counter = 0
    layer = 0
    if len(liste) > 0:
        for n in graph.nodes():
            if counter < liste[layer]:
                counter += 1
            else:
                counter = 1
                layer += 1
            dot.node(str(n), str(n), style="filled", colorscheme="paired12", color=str(layer + 1))
    else:
        for n in graph.nodes():
            dot.node(str(n), str(n))
    for e in graph.edges():
        dot.edge(str(e[0]), str(e[1]))
        #print str(e[0]), str(e[1])
    # print dot.source
    dot.render('graph_output/' + name, view=False)

def plotprettydis(graph, delays, name):
    dot = Digraph(comment=name)
    for n in graph.nodes():
        if delays[n] > 0:
            color = min(delays[n], 9)
            dot.node(str(n), str(n), style="filled", colorscheme="reds9", color=str(color))
    for e in graph.edges():
        dot.edge(str(e[0]), str(e[1]))
        #print str(e[0]), str(e[1])
    dot.render('graph_output_disrupted/' + name, view=False)

# TODO compute clustering measures and do power law per tier for degree distribution
# TODO Issue : TOO many edges in some tiers and not enough in others.


# Get my names

def getnameedges(seed):
    return "edges" + str(start_my_code.NUMBER_OF_NODES) + '_' + str(seed) + ".txt"

def getnamenodes(seed):
    return "nodes" + str(start_my_code.NUMBER_OF_NODES) + '_' + str(seed) + ".txt"

def getmygraphname(seed):
    return 'graph'+str(start_my_code.NUMBER_OF_NODES) + '_' +str(seed)

def getnameedgesbaby(seed):
    return "edges" + str(start_my_code.NUMBER_OF_NODES) + '_' + str(seed) + "baby.txt"

def getnamenodesbaby(seed):
    return "nodes" + str(start_my_code.NUMBER_OF_NODES) + '_' + str(seed) + "baby.txt"

def getmygraphnamebaby(seed):
    return 'graph'+str(start_my_code.NUMBER_OF_NODES) + '_' +str(seed) +'baby'

def getnameedgesbro(seed, n):
    return "edges" + str(start_my_code.NUMBER_OF_NODES) + '_' + str(seed) + "baby" + str(n) + ".txt"

def getnamenodesbro(seed, n):
    return "nodes" + str(start_my_code.NUMBER_OF_NODES) + '_' + str(seed) + "baby" + str(n) + ".txt"

def getmygraphnamebro(seed, n):
    return 'graph'+str(start_my_code.NUMBER_OF_NODES) + '_' +str(seed) + "baby" + str(n) + ".txt"


# Generate a lot of graphs

def generatealotofgraphs(n, plot, initseed, iter):
    duplicate = 0
    built = 0
    for i in range(0, n):
        seed = initseed + i
        numpy.random.seed(seed)
        namenode = getnamenodes(seed)
        nameedge = getnameedges(seed)
        namegraph = getmygraphname(seed)

        graph = build_my_random_net(start_my_code.NUMBER_OF_NODES, start_my_code.NUMBER_OF_TIER_1, start_my_code.LENGTH, start_my_code.SIG_FACT, start_my_code.GAMMA, start_my_code.FACTOR,  True)
        graph.writemygraph(namenode, nameedge, seed)

        if doesthatgraphexistedges(nameedge):
            print "one duplicate of edges topo and critic found", seed
            os.remove('graphdata/' + nameedge)
            os.remove('graphdata/' + namenode)
            duplicate += 1

        else:
            built += 1
            print "graph built", built
            if plot:
                plotpretty(graph.graph, graph.nodelayer, namegraph)

    print "number of duplicate found and not plotted and deleted", duplicate

    if duplicate > 0 and iter < 100:
        iter += 1
        generatealotofgraphs(duplicate, plot, initseed+n, iter)

def aretheseedgesthesame(nameedge, otheredge):
    if filecmp.cmp('graphdata/' + nameedge, 'graphdata/' + otheredge):
        return True
    else:
        return False

def doesthatgraphexistedges(nameedge):
    path = 'graphdata/'
    listing = os.listdir(path)
    test = 0
    res = False
    for infile in listing:
        if aretheseedgesthesame(nameedge, infile):
            if test == 1:
                res = True
            else:
                test = 1
    return res

#  Generate baby graphs

def getmybabies(n, seed):
    total = n
    initt = seed
    left = n
    current_seed = seed

    while left > 0:
        file1 = 'nodes' + str(start_my_code.NUMBER_OF_NODES) + '_' + str(current_seed) + '.txt'
        file2 = 'edges' + str(start_my_code.NUMBER_OF_NODES) + '_' + str(current_seed) + '.txt'
        try:
            netw = addmygraph(file1, file2)
            netw.removeoneedge()
            left -= 1
            print "left", left
        except IOError, e:  ## if failed, report it back to the user ##
            print "try next seed", current_seed
            if current_seed > initt + 1000000*total:
                print "too many seeds, stop."
                err = file.open('erreur_report.txt', 'w')
                err.write("too many seeds stop", current_seed, n, left)
                break
        current_seed +=1

def returnmynode(label, liste):
    try:
        for i in liste:
            if i.label == label:
                node = i
        return node
    except NameError:
        print "node not found"
        raise NameError('StructurePb')

def getallsiblings(n, seed):
    total = n
    initt = seed
    left = n
    current_seed = seed

    while left > 0:
        file1 = 'nodes' + str(start_my_code.NUMBER_OF_NODES) + '_' + str(current_seed) + '.txt'
        file2 = 'edges' + str(start_my_code.NUMBER_OF_NODES) + '_' + str(current_seed) + '.txt'
        current_seed +=1
        try:
            netw = addmygraph(file1, file2)
            netw.writesiblings()
            left -= 1
            #print left
        except IOError, e:  ## if failed, report it back to the user ##
            print "try next seed", current_seed
            if current_seed > initt + 1000000*total:
                print "too many seeds, stop."
                err = file.open('erreur_report_sib.txt', 'w')
                err.write("too many seeds stop", current_seed, n, left)
                break


def graph_number_babies(nn, seed_family):
    babies = 0

    start_name = 'edges' + str(nn) + '_' + str(seed_family) + 'baby'

    for filename in os.listdir('graphdata/'):
            if filename.startswith(start_name):
                babies +=1
    #print babies
    return babies

def check_right_number_graphs(nn, number, seed_init, seed_finale):
    number_families = 0
    current_family = seed_init
    res = False

    while number_families < number:
        start_name = 'edges' + str(nn) + '_' + str(current_family)
        found = False

        for filename in os.listdir('graphdata/'):
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