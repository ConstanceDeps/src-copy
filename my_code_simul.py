__author__ = 'constancedeperrois'

import simpy_mine
import start_my_code
import Graph_Plot
import numpy
import os

class Disruptions(object):
    def __init__(self, netw, filename, env, namepdf, plot, track, seed):
        self.locations = []  # Time since a delay has started at a node. Returns to zero once it's solved.
        self.netw = netw
        self.env = env
        self.action = env.process(self.run)
        self.filename = filename
        self.delays = []
        nodes = self.netw.graph.nodes()
        for i in range(0, len(nodes)):
            self.locations.append(0)
            self.delays.append(0)
        self.file = open(self.filename, 'w')
        self.name = namepdf
        self.plot = plot
        self.track = track
        self.indextrack = []
        self.seed = seed

    @property
    def run(self):

        file_ext = self.file
        file_ext.write("Number of suppliers is " + str(len(self.netw.graph.nodes())-1) + "\n" + "\n" + "Time Manufacturer Suppliers " + "\n" + "t  ")
        for i in range(0, len(self.netw.graph.nodes())):
            if i < 10:
                file_ext.write("   " + str(i))
            elif i < 100:
                file_ext.write("  " + str(i))
            else:
                file_ext.write(" " + str(i))
        file_ext.write("\n")

        #file = open("storytelling.txt", 'w')
        file = 0   # does not mean anything but for not to many files open

        mynodes = self.netw.mynodes

        self.indextrack.append(startmysimul(self.netw, self.track, self.seed, self.plot))

        while True:
            #  Write the beginning of the line
            if env.now < 10:
                file_ext.write(str(env.now) +"  ")
            elif env.now > 99:
                file_ext.write(str(env.now))
            else:
                file_ext.write(str(env.now) +" ")

            for i in range(0, len(mynodes)):

                # Write the delay of the node
                node = mynodes[i]
                if node.delay < 10:
                    file_ext.write("   " + str(node.delay))
                elif node.delay < 100:
                    file_ext.write("  " + str(node.delay))
                else:
                    file_ext.write(" " + str(node.delay))

                if node.label != 0 and node.state:  #  If node disrupted and not manuf
                    newdelay = amplifyDis(node, self.delays)
                    recovery_time = start_my_code.recovery_time(node.rating)
                    time_since_started = self.locations[node.label]

                    if time_since_started >= recovery_time or newdelay == 0:   # If has receovered of if new dealy is zero
                        self.locations[node.label] = 0
                        node.state = False
                        node.delay = 0
                        if self.plot:
                            print "Disruption solved at time", self.env.now, "at location", node.label
                        #file.write("Disruption solved at time " + str(self.env.now) + " at location " + str(node.label)+"\n")

                        if self.track and self.indextrack.count(node.label) > 0:
                            self.indextrack.remove(node.label)

                    else:
                        self.locations[node.label] += 1
                        node.delay = newdelay
                        if time_since_started > 0 and time_since_started % 5 == 0:  #node.lead_time
                            propagateDis(node, self.netw, file, self.delays, newdelay, self.track, self.indextrack, self.seed, self.plot)

            if self.env.now == start_my_code.TIME_SIMUL-1 :
                #print "done, final delays manuf", self.delays[0]
                file_ext.close()
                if self.plot:
                    Graph_Plot.plotprettydis(self.netw.graph, self.delays, self.name)

            yield self.env.timeout(1)
            file_ext.write("\n")
        file_ext.close()
        #file.close()

def amplifyDis(supa, delays):
    rating = supa.rating
    factor = start_my_code.amplificationSupplier(rating)
    newDelay = int(round(factor * supa.delay))
    #print newDelay
    if newDelay > 180:
        newDelay = 180
    delays[supa.label] = newDelay
    return newDelay  # Delay cannot be more than 6 months

def propagateDis(supa, netw, file, delays, newDelay, track, indextrack, seed, plot):
    outof=[]
    for ed in netw.myedges:
        if ed.origin.label == supa.label:
            outof.append(ed)
    length = len(outof)
    if length > 0:
        if plot:
            print "Disruption propagates at time", env.now, "from", supa.label, "to", length, "neighbour(s)"
        #file.write("Disruption propagates at time " + str(env.now) + " from " + str(supa.label) + " to " + str(length) + " neighbour(s) " + "\n")
        for i in range(0, length):
            edge = outof[i]
            dest = edge.destination
            if dest.label == 0:
                dest.delay = newDelay
                delays[0] += newDelay
                if plot:
                    print "Boeing !!!!!! at time", env.now, " from ", supa.label, "with a delay of", newDelay
                #file.write("Disruption reaches Boeing at time " + str(env.now) + " from " + str(supa.label) + " with a delay of " + str(newDelay) + "\n")
                if track and indextrack.count(supa.label) > 0:
                    indextrack.append(dest.label)
                    trackmydis(env.now, 0, newDelay, seed)
            elif doyoutransmit(edge, dest):
                #print "transmits"
                dest.state = True
                dest.delay = newDelay
                delays[dest.label] += newDelay
                if track and indextrack.count(supa.label) > 0:
                    indextrack.append(dest.label)
                    trackmydis(env.now, dest.label, newDelay, seed)
                if plot:
                    print "Disruption affects", dest.label, "at time", env.now, "from", supa.label, "with a delay of", dest.delay
                #file.write("Disruption affects " + str(dest.label) + " at time " + str(env.now) + " from " + str(supa.label) + " with a delay of " + str(dest.delay) + "\n")

def doyoutransmit(edge, dest):
    rating = dest.rating
    part_critic = edge.part
    proba_transmission = start_my_code.proba_transmission(part_critic, rating)
    a = numpy.random.random()
    if a <= proba_transmission:
        return True
    else:
        return False

def startmysimul(netw, track, seed, plot):
    list = start_my_code.DISRUPT_INIT
    for i in list:
        netw.createMyDis(i[0], i[1])
    res = netw.createRandomDis(start_my_code.INIT_RANDOM, track)
    if track:
        try:
            index = res[0]
            value = res[1]
            print "is tracking", index, value
            trackinit(index, value, seed)
            return index
        except IndexError:
            try:
                index = list[0][0]
                value = list[0][1]
                print "is tracking", index, value
                trackinit(index, value, seed)
                return index
            except IndexError:
                print "You cannot track a disruption if there is none"
                print "Disruption was not tracked."

# TODO checkups code

# Functions to get the statistics - To compare simulations

# Track disruptions

def trackinit(index, value, seed):
    filename = start_my_code.DIRECTORY_TRACK + str(start_my_code.NUMBER_OF_NODES) + "trackdis" + str(seed) + ".txt"
    filetrack = open(filename, 'w')
    filetrack.write("The delay being tracked starts at node " + str(index) + "\n")
    filetrack.write("Time Index Value \n")
    filetrack.write("0 " + str(index) + " " + str(value) + "\n")
    filetrack.close()

def trackmydis(time, index, value, seed):
    filename = start_my_code.DIRECTORY_TRACK + str(start_my_code.NUMBER_OF_NODES) + "trackdis" + str(seed) + ".txt"
    filetrack = open(filename, 'a')
    filetrack.write(str(time) + " " + str(index) + " " + str(value) + "\n")
    filetrack.close()

# Gets the value of the disruption at time t
def getmyvalue(linestring):
    i = 0
    value = ''
    l = len(linestring)
    #print linestring, "line", l
    while linestring[i] != ' ':
        i +=1
    i +=1     #miss the first space
    while linestring[i] != ' ':
        i += 1
    j = i+1
    while j<l and linestring[j] != ' ':
        value += linestring[j]
        j +=1
    value = int(value)
    #print i, j, value
    return value

# Analyses only one track
def analysemytrack(filen):
    filename = start_my_code.DIRECTORY_TRACK + filen
    filetrack = open(filename, 'r')
    myList = []
    reaching_B = False
    max_amplification = 0
    init_dis = 1

    for line in filetrack:
        myList.append(line)
        if line[0] == '0':
            i = 2
            while line[i] != ' ':
                i += 1
            if line[i+2] == ' ':
                init_dis = int(line[i+1])
            else:
                init_dis = int(line[i+1] + line[i+2])
    length = len(myList)

    for n in range(2, length):
        value = getmyvalue(myList[n])
        #print "value", value
        max_amplification = max(max_amplification, value)

    #print "max amp", max_amplification
    max_amplification = (max_amplification+0.0) / (init_dis + 0.0)

    final = myList[length-1]
    i = 0
    time = ''
    while final[i] != ' ':
        time += final[i]
        i +=1
    if final[i+1] == '0':
        reaching_B = True
    if reaching_B :
        time_to_B = int(time)
    else:
        time_to_B = -1

    final_dis = getmyvalue(final)

    results = [reaching_B, time_to_B, round(max_amplification, 2), init_dis, final_dis]
    print "[reaching_B, time_to_final, max_amplification, init_dis, final_dis] : ", results
    return results

# Average time to reach the manufacturer for a disruption.

def stats_lot(total):
    freq_reach_b = 0
    time = []
    ampli = []
    init = []
    fin = []

    for i in range(0, total):
        filen = str(start_my_code.NUMBER_OF_NODES) + "trackdis" + str(start_my_code.INIT_SEED + i) + ".txt"
        filename = start_my_code.DIRECTORY_TRACK + filen

        res = analysemytrack(filename)
        if res[0] :
            freq_reach_b += 1
            time.append(res[1])
        ampli.append(res[2])
        init.append(res[3])
        fin.append(res[4])

    freq_reach_b = (freq_reach_b + 0.0) / (total + 0.0)
    av_time_to_b = numpy.mean(time)
    av_max_amplification = numpy.mean(ampli)
    av_init = numpy.mean(init)
    av_fin = numpy.mean(fin)

    results = [freq_reach_b, round(av_time_to_b, 2), round(av_max_amplification, 2), round(av_init, 2), round(av_fin, 2)]
    print "[freq_reach_b, av_time_to_b, av_max_amplification, av_init, av_fin] : ",  results
    return results

def computeavproba():
    av_rating = 1*start_my_code.RATING_1 + 2*start_my_code.RATING_2 + 3*start_my_code.RATING_3
    av_part = 1*start_my_code.CRITIC_1 + 2*start_my_code.CRITIC_2 +3*start_my_code.CRITIC_3
    av_proba = start_my_code.proba_transmission(av_part, av_rating)
    print av_proba
    return av_proba

def computeavtime():
    av_rating = 1*start_my_code.RATING_1 + 2*start_my_code.RATING_2 + 3*start_my_code.RATING_3
    av_rating = 20 * av_rating
    print av_rating
    return av_rating

# Starts a simulation

#env = simpy_mine_mine.Environment()

def StartMySimul(time, G, plot, track, seed):
    global env
    env = simpy_mine.Environment()
    print "env now is ", env.now
    numpy.random.seed(seed)
    namepdf = str(start_my_code.NUMBER_OF_NODES) + 'disrupted.gv' + str(seed)
    name_results = start_my_code.DIRECTORY_SIM + "results" + str(start_my_code.NUMBER_OF_NODES) + str(seed) + ".txt"
    Disruptions(G, name_results, env, namepdf, plot, track, seed)
    env.run(until=time)

def startsim(G, plot, track, seed):
    numpy.random.seed(seed)
    namepdf = str(start_my_code.NUMBER_OF_NODES) + 'disrupted.gv' + str(seed)
    name_results = start_my_code.DIRECTORY_SIM + "results" + str(start_my_code.NUMBER_OF_NODES) + str(seed) + ".txt"
    Disruptions(G, name_results, env, namepdf, plot, track, seed)

def start_sim_no_plot_but_name(G, seed, name_result, namepdf):
    numpy.random.seed(seed)
    Disruptions(G, name_result, env, namepdf, True, True, seed)

def startall(time, G, plot, track, seed, n):
    global env
    env = simpy_mine.Environment()
    for i in range(0, n):
        startsim(G, plot, track, seed + i)
    env.run(until=time)

def simulforallbabies(time, seed_simul, nodes, number_simul, seed):
        global env
        env = simpy_mine.Environment()

        found = -10
        file1 = 'nodes' + str(nodes) + '_' + str(seed)
        file2 = 'edges' + str(nodes) + '_' + str(seed)
        directory = 'graphdata/'
        allgraphs = Graph_Plot.graph_number_babies(nodes, seed) + 1
        print "Famille",seed, "avec", allgraphs
        if allgraphs == 1:
            return 0

        for filename1 in os.listdir(directory):
            if filename1.startswith(file1):
                end = filename1.rsplit('_', 1)[1]
                end2 = end.rsplit('.', 1)[0]
                for filename2 in os.listdir(directory):
                    if filename2.startswith(file2) and filename2.endswith(end):
                        g = Graph_Plot.addmygraph(filename1, filename2)
                        allgraphs -= 1
                        # print allgraphs
                        if allgraphs == 0:
                            found = 1
                        for i in range(number_simul):
                            name_res = start_my_code.DIRECTORY_SIM + "results" + str(nodes) + end + 'simul' + str(seed_simul + i) + ".txt"
                            name_pdf = 'disrupted' + str(nodes) + 'simul' + str(seed_simul + i) + 'famille' + end2
                            start_sim_no_plot_but_name(g, seed_simul + i, name_res, name_pdf)
        env.run(until=time)
        return found



