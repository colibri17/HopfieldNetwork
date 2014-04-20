# Main file. It has to be executed by the user

# I import the numpy library, which allows
# to perform numerical operations quickly
import numpy as np

# I import the Hopfield Network module
import HopfieldNetwork

# I import the matplotlib library in order to
# generate the plots
import matplotlib.pyplot as plt

# Instructions to save the plots in a format which
# can be useful for the latex report
import matplotlib.mlab as mlab
import matplotlib.font_manager
font = {'size'   : 7}
matplotlib.rc('font', **font)
fig = plt.figure(figsize = (6,5))


# I define the initial set of parameters
n = 9
weight_value = 0.5
b = [0.]*n
strictly = 0
# I define the initial points on which test the network
init = [[1, 1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0, 1], [1,1, 0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 1, 0, 1, 1], 
        [1, 0, 0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1, 1, 1], [0, 0, 1, 0, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 0, 0, 1, 0], 
        [0, 1, 0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 1, 1, 0, 1, 1], 
        [1, 0, 1, 1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0, 0], [1, 0, 0, 1, 0, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0, 0, 0, 1], 
        [0, 0, 1, 0, 0, 0, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0, 1, 1, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0]]

# I create the connection weight matrix
# First, I create the matrix having the selected constant weights
# as elements
weight = -weight_value * np.ones([n,n])

# Then, I modify the elements which are different from the constant value.
# To do that, I cycle over the rows and the columns of the matrix and 
# each time I hit a value I want to change, I change it
for i in range(n):
    for j in range(n):
        # Different cases in which I have to modify the matrix element
        if i == j: weight[i,j] = 0
        elif (i==1 and (j==4 or j==7)): weight[i,j] = 1
        elif (i==3 and (j==4 or j==5)): weight[i,j] = 1
        elif (i==4 and (j==1 or j==7)): weight[i,j] = 1
        elif (i==4 and (j==3 or j==5)): weight[i,j] = 1
        elif (i==5 and (j==3 or j==4)): weight[i,j] = 1
        elif (i==7 and (j==1 or j==4)): weight[i,j] = 1
        
# I print the matrix
# print weight
        
# I create a new HopfieldNetwork object.
# n = the number of unit of the network
# weight = the connection matrix
# b = the bias vector
hop = HopfieldNetwork.HopfieldNetwork(n, weight, b)

# (a) I compute the fixed points
steady_points = hop.retrieve_steady_points('sync', weight)
print 'Steady points', steady_points

# (b) I compute the stable points
minimum_points = hop.retrieve_minimum_points(steady_points, weight, strictly)
print 'Stable points', minimum_points

# (c) I run the network for a suitable number of time by using both the scan 
# and the async update rules providing as input the init elements. Also, I generate
# the plots for the desired states, saving them.

# List of states whose plots must be saved in a file
save_list = [[0, 1, 0, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 1]]

# I begin the cycle
for state in init: 
    
    print 'Initial state', state,
    
    # I evolve the network for the selected number of times, also
    # providing the weight matrix and the selected state and the
    # desired updating rule
    # After having run the hopfield function, it returns the list
    # containing all the states which was hit
    evolution_states_scan = hop.hopfield('scan', weight, state, 5)
    # I record the scan energy of all the states
    energy_scan = [hop.energy(weight, x) for x in evolution_states_scan]
    
    # The same as before using the random asynchronous rule instead
    evolution_states_async = hop.hopfield('async', weight, state, 50)
    energy_async = [hop.energy(weight, x) for x in evolution_states_async]
    
    print 'final state', evolution_states_async[-1]
    
    # I save the energy plots
    if state in save_list:
        plt.clf()
        plt.plot(energy_async)
        plt.title('Hopfield energy evolution against the time with a random asynchronous rule')
        plt.ylabel('Energy')
        plt.xlabel('Time')
        plt.grid(True)
        # I save the plot obtained in a file in order to be able to reuse it
        plt.savefig("energy_async_%s_%s.png" % (str(weight_value).replace('.',''),''.join(str(x) for x in state)), bbox_inches='tight', transparent = True, dpi = 600)

        plt.clf()
        plt.plot(energy_scan)
        plt.title('Hopfield energy evolution against the time with a scan asynchronous rule')
        plt.ylabel('Energy')
        plt.xlabel('Time')
        plt.grid(True)
        # I save the plot obtained in a file in order to be able to reuse it
        plt.savefig("energy_scan_%s_%s.png" % (str(weight_value).replace('.',''),''.join(str(x) for x in state)), bbox_inches='tight', transparent = True, dpi = 600)
