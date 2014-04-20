# This file implements the generic methods related to a binary Hopfield network.
# I wanted to mantain (more or less) the same structure as the Mathematica
# files.

# I import the numpy library, which allows
# to perform numerical operations quickly
import numpy as np

# I import the random module
import random

# I import the itertools module
import itertools

# I import the copy module
import copy

# Hopfield class
class HopfieldNetwork:

    # Constructor method
    def __init__(self, n, weight, b):

        # I store the dimension
        self.n = n  
        # I store the b vector, which affects
        # the activate function, generalizing it
        self.b = b
        # I create an empty list to store the states
        # hit during evolution of the network
        # when the hopfield function is called
        self.evolution = []
        # I create an empty list to store the 
        # steady points of the network
        self.steady_points = []
        # Creating an empty list to store the 
        # local minimum points of the network
        self.minimum_points = []
                
        
                
    # It decides whether the unit of the Hopfield neural network has to be 
    # activated. Depending on the i-th unit selected
    # I could have different values of the parameter b
    def activate(self, element, i):
        
        # np.rint allows to round to the nearest integer
        # np.sign returns the sign of the vector
        # The activation function works as it follows:
        # if element > 0 I expect an output of 1. Indeed in this case I obtain round(.5 + .5*1) = 1
        # if element = 0 I expect an output equals 0. Indeed in this case I obtain round(.5 + .5*1) = 0
        # if element > 0 I expect an output of 0. Indeed in this case I obtain round(.5 + .5*(-1)) = 0
        return np.rint(.5 + .5 * np.sign(element + self.b[i]))
    
    
    
    # It updates the network according to the random asynchronous rule
    # state = n element input state vector
    # weight = n*n connection matrix
    def async(self, weight, state):
        
        # I create an array equal to the state variable
        # In doing so, the given state variable is not overwritten
        new_state = copy.deepcopy(state)
        
        # I extract an interger number in [0,n-1]
        # in order to select the unit I want to modify
        a = random.randint(0,self.n-1)
        # print a
        
        # I decide whether activating the unit
        c = self.activate(np.dot(weight, state), a)
        # print c
        
        # IpUpdate the state
        new_state[a] = c[a]
        # print 'New state with async', new_state
        
        # I return the updated state
        return new_state
    
    
    
    # It performs n-updates of the network according to the scan asynchronous rule
    # state = n element input state vector
    # weight = n*n connection matrix
    def scan(self, weight, state): 
        
        # I create an array equal to the state variable
        # In doing so, the state variable is not overwritten
        new_state = copy.deepcopy(state)
        
        for i in range(self.n):
            # I compute the dot product between the weight matrix
            # and the state
            a = np.dot(weight,new_state)
            
            # I update the step
            new_state[i] = self.activate(a[i], i)
        
        # print 'New state with scan', new_state
        
        # I return the updated state
        return new_state
    
    
    
    # It updates the network according to the synchronous rule
    # state = n element input state vector
    # weight = n*n connection matrix
    def sync(self, weight, state):
        
        # I create an array equal to the state variable
        # In doing so, the state variable is not overwritten
        new_state = []
        
        # I compute the product between the weight matrix
        # and the old state
        a = np.dot(weight, state)
        
        for i in range(self.n):         
            # I update the corresponding state
            new_state.append(self.activate(a[i], i))
            
        
        # print 'New state using sync', new_state
        
        # I return the new state
        return new_state
    
    
    
    # Function which allows to evolve the system
    # update = the updating rule chosen to update the network
    # state = n element input state vector
    # weight = n*n connection matrix
    # n_iterations = number of iterations required
    def hopfield(self, update_name, weight, state, n_iterations):
        
        # I choose which of the updating rules
        # has to be selected. If the update_name 
        # is not within the possible choices, I return -1 after having printed message
        if update_name == 'scan': 
            update = self.scan
        elif update_name == 'async':
            update = self.async
        elif update_name == 'sync':
            update = self.sync
        else:
            print 'No update rules having that name. Try [scan, async, sync]'
            return -1
        
        # I create an array equal to the state variable.
        # In doing so, the state variable is not overwritten
        new_state = copy.deepcopy(state)
        
        # I reset the evolution list
        self.evolution = []
        
        # I append the initial state to the list
        self.evolution.append(state)
        
        # I apply the selected rule in order to update the state.
        # Also I store the states which are produced in a list (obtaining therefore a list of arrays)
        # in order to track the evolution and subsequently store it
        i = 0
        while i < n_iterations:
            
            # I retrieve the next state of the network
            new_state = update(weight, new_state)
            # print 'New state obtained', new_state, 'energy', self.energy(weight, new_state)
            
            # Store the new state in the corresponding tracking list
            self.evolution.append(new_state)
            
            # Increment i
            i += 1
            
        # Return the last state
        return self.evolution
    
    
    
    # Function which computes the energy of the system
    # The energy of one state can be defined in a vector 
    # notation as: H(u) = -1/2u^{T}*W*u - u^{T}b
    def energy(self, weight, state):
        
        # I convert the state, which could be a list, into an array.
        # So I can perform all the numerical operations I want
        state = np.array(state)

        # Definition of the energy of the system. The 
        # vector b is stored at the beginning, during the definition of the Hopfield network
        # the state and the weight function are given by the user.
        # Because of the behaviour of the dot product, I can ignore the transposition
        # operation (it is automatic)
        return -0.5 * np.dot(state, np.dot(weight, state)) - np.dot(state, self.b)
    
    
    
    # I test each of the 2^n possible states of the network
    # in order to detect the steady states. A state is steady if 
    # it remains constant over the time. The random asynchronous rule 
    # is not accepted (see report)
    def retrieve_steady_points(self, update_name, weight):

        # I choose which of the updating rules
        # has to be selected. If the update_name 
        # is not within the possible choices, I return -1 after having printed message        
        if update_name == 'scan': 
            update = self.scan
        elif update_name == 'sync':
            update = self.sync
        else:
            print 'It is not possible retrieve the steady state with that rule. Try [scan, sync]'
            return -1
        
        # I create all the 2^n possible binary states by utilizing
        # the product module of the itertool Python library. It generates
        # the cartesian product of the input (range(2))
        # repeated self.n times. I used the list 
        # comprehension just to convert the elements from tuples to lists
        for state in [list(x) for x in itertools.product(range(2), repeat=self.n)]:
        
            # If I apply the update function to the state and I obtain
            # exactly the same state, it is a steady point
            if update(weight, state) == state:
                self.steady_points.append(state)
            
            # Otherwise I do nothing
            
        # I return the steady points
        return self.steady_points
    
    
    
    # It retrieves the positions for which the energy function is
    # lowest
    # steady points = list of the steady points of the Hopfield network
    # weight = n*n connection matrix
    # strictly = specifies if strictly minimum points are requested (1) or not (0)
    def retrieve_minimum_points(self, steady_points, weight, strictly):
        
        # Quick check on the strictly values
        if not (strictly == 1 or strictly == 0):
            print 'The value of the strictly parameter is not allowed. Try [0,1]'
        
        # I cycle over each steady state
        for state in steady_points:
            # If the comparison returns 1, then the state is stable
            # Otherwise it is not stable
            if self.comparing_neighbour_energies(state, weight, strictly):
                self.minimum_points.append(state)
                
        # I return the stable points
        return self.minimum_points
            
    
    
    
    # It compares the energy of the provided state
    # with those of its neighbours
    def comparing_neighbour_energies(self, state, weight, strictly):
            
        # I compute the energy associated to the state
        energy_state = self.energy(weight, state)
        
        # I compute the energies associated to its neighbours.
        # For each unit I verify if the corresponding neighbour has 
        # a lower energy. If this is the case then the state is not stable.
        for i in range(len(state)):
            
            # I construct the next neighbour state
            if state[i] == 1: 
                neighbour = state[:i] + [0] + state[i+1:]
            else: 
                neighbour = state[:i] + [1] + state[i+1:]
            # print 'neighbour', neighbour, 'original', state
            
            # I compute the energy of the neighbour state. If this
            # energy is greater than the energy of the original state
            # then the point is not a stable state
            # print 'neighbour energy', self.energy(weight, neighbour), 'Original state energy', energy_state
            if strictly == 1:
                if self.energy(weight, neighbour) <= energy_state:
                    return 0
            else:
                if self.energy(weight, neighbour) < energy_state:
                    return 0                
            
        # If the test passes then the point is a stable point
        return 1