__author__ = 'constancedeperrois'
import numpy

"""
This is where all parameters / variables are stored.
In addition example graphs are also stored in txt files.
All these variable have to be filled in for certain bits of the code to run.
Parameters and probabilities
"""

"""
Rating of a supplier impact its recovery time.
The recovery time is the time needed to completely solve the disruption.
Once the recovery time has gone by everything functions normally.
Before that, disruptions are transmitted.
Rating is 1, 2 or 3. Suppliers rated 1 are very good and recover faster.
"""

def recovery_time(rating):
    factor = numpy.random.triangular(7, 7*rating, 12*rating)
    time = numpy.around(factor)
    #  print time, rating
    return time

"""
The probability of transmission is the probability that the disruption propagated by a supplier is perceived.
It depends on the criticallity of the part for that customer, and on the customer rating.
(if it is a good rating it is likely that he has inventory.)
part: 3 = very critical, 1 = not critical at all.
"""

def proba_transmission(part, rating):
    #proba = 1.0
    proba = 1 - ((1.0 / (part))*(1.0 / (rating))) + 1/9
    return proba

"""
The amplification factor is the way the supplier affects the delay. Does it amplifies it or mitigates it?
This depends on the supplier's rating. If a supplier is good, then it will deal with the delay.
Above 1 if the supplier amplifies the delay, below if it mitigates it.
"""

def amplificationSupplier(rating):
    factor = numpy.random.triangular(0, 0.8*rating, 1.2*rating)
    # print factor
    return factor

"""
The following parameters are needed to generate a supply network.
Topology of the network
"""

NUMBER_OF_NODES = 9      # 544 in A

NUMBER_OF_TIER_1 = 3     # 91 in Alexandra Less than NUMBER_OF_NODES

# CONDITION : NUMBER_OF_NODES > 2 NUMBER OF TIER 1 + 2

LENGTH = 4  # LENGTH = tiers + 1, and more than 4. Alexandra : 5

SIG_FACT = 1.45  #  Factor by which is multiplied max length/2 to get mu fit distribution with gaussian. Can be calculated with main.

# Parameters for the fit of SIG_FACT
# Theoretical distribution of nodes (A data). Fit can only be done if length is 5.
TIER2 = 0.37
TIER3 = 0.34
TIER4 = 0.28


# AVERAGE_NUMBER_OF_EDGES = 80    # Alexandra's values are 544 nodes and 1657 edges.
# CLUSTERING_OF_NETWORK = 0.314    # 0.314 is Alexandra's value.

# TODO add check for values.

"""
Information about nodes and edges
"""

# Frequencies of supplier ratings, estimates from the manufacturer
RATING_1 = 0.2
RATING_2 = 0.5
RATING_3 = 1-RATING_1-RATING_2

# Frequencies of types of part : critical or not, estimates from the manufacturer
CRITIC_1 = 0.1
CRITIC_2 = 0.2
CRITIC_3 = 1-CRITIC_1-CRITIC_2

# Boundaries and av. time between orders. (for a triangular distribution)
LOW = 2
MEAN = 7
HIGH = 30

# Parameters of the power law for the degree of the graph
GAMMA = 1.2    #  Alexandra 1.911
FACTOR = 1    #  Alexandra: 394.05

"""
Simulation Parameters

List of initial disruptions if they are specified.
Number of random disruptions to be added.
Parameter of the poisson law for the duration of random disruptions () #for thesis: could be refined per tier and so on, other parameters.
Poisson because of independance of events. (assumption)
Create random disruptions does not check for duplicates! there may be less disruptions in the end.

"""

TIME_SIMUL = 30

#DISRUPT_INIT = [[15, 3], [16, 1]]

#DISRUPT_INIT = [[4, 3]]

#DISRUPT_INIT = [[450, 8]]

DISRUPT_INIT = [[8, 5]]

LAMBDA = 15
INIT_RANDOM = 0    #  Number of random initial disruptions.

NUMBER_OF_SIMUL = 10 #200    #  To compute statistics on the propagation of disruptions.

DIRECTORY_SIM = 'simres/'
DIRECTORY_TRACK = 'trackres/'

"""
Data for generating my graphs for the experiment
"""

NUMBER_OF_GRAPHS = 100 #000

"""
V important, seed to start everything
"""

INIT_SEED = 987987   # First seed for graphs

INIT_SEED_SIMUL = 5678
