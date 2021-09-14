################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
env.play()

#Other liberaties
import time
import datetime
import numpy as np
    
#class EA (object)

#def parent_selection(self, x_old, f_old):
    '''This function selects the parents from a population 
    Select the two parents to crossover. The parents are selected in a way similar to a ring. The first with indices 0 
    and 1 are selected at first to produce two offspring. If there still remaining offspring to produce, then we select 
    parent 1 with parent 2 to produce another two offspring 
    
    Input: 
    x_old  -  array of population genes 
    y_old  -  array of population fitness 
    
    Output: 
    x_parents - array of parents genes 
    f_parents - array of parents fitness '''


#    return x_parents, f_parents 

#def recombination(self, x, f):
    '''This function recombine the parents to new children by an uniform crossover operator
    
    Input: 
    x - array of parent genes 
    y - array of parent fitness
    
    Output: 
    x_children - array of children with parent genes '''

#   return x_children

#def mutation (self, x): 
    '''This function mutates the new children with Gaussian noise
    
    Input: 
    x - array of children genes 
    
    Output: 
    x - array of mutated children genes '''

#def selection (self, x_old, x_children, f_old, f_children): 
    '''This function selects a number of genes in the total population to go to the next 
    
    Input: 
    x_old - array of population genes
    x_children - array of children genes  
    f_old - array of population fitness 
    f_children - array of children fitness 
    
    Output: 
    x - array of genes which survived 
    f - array of fitness which survived '''

#    return x, f


#functions for fitness
    #def find_seconds(date_start, date_end):
        '''seconds = (date_start - date_end)
        return seconds'''

#def evaluate(self, x):
    # into the fitness function 
    '''seconds = find_seconds()
        power_points_lost = x
        energie_enemy = x
        fitness = alpha * seconds + beta * power_points_lost + gamma * energie_enemy + penalty'''

#def step (self, x_old, f_old): 
    '''This function generates a new population of individuals 
    
    Input: 
    x_old - array of population genes 
    f_old - array of population fitness 
    
    Output: 
    x - array of new population genes 
    f - array of new population fitness'''

    #selection of parents from the old population:
#    x_parents, f_parents = self.parent_selection(x_old, f_old)
    #the parents recombine into children:
#    x_children  = self.recombination (x_parents, f_parents)
    #the children obtain some random mutation (gaussian noise):
#    x_children = self.mutation(x_children)
    #the children evaluate:
#    f_children = self.evaluate(x_children)
    # the population i: 
#    x, f = self.selection(x_old, x_children, f_old, f_children)
#    return x, f 
