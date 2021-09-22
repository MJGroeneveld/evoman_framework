
################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################p

import sys, os

from numpy.lib.function_base import select 
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

import time
import numpy as np
from math import fabs, sqrt
import glob, os
import random as rd

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def genetic_algorithm(survival_method):
    """
        Genetic algorithm solver
    """
    
    experiment_name = 'dummy_demo'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10
    
    ini = time.time() # Set the time marker
    
    # genetic algorithm parameters
    dom_l = 1
    dom_u = -1
    pop_size = 10
    no_of_generations = 20
    each_generation = 0
    

    # Initialize Environment
    env = Environment(
        experiment_name = experiment_name,
        enemies = [3],
        playermode = "ai",
        player_controller = player_controller(n_hidden_neurons),
        enemymode = "static",
        level = 2,
        speed = "fastest"
    )
    # check the environment state
    env.state_to_log()

    # No of weights for a multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
    
    population = generate_population(
        dom_l=dom_l,
        dom_u=dom_u,
        npop=pop_size,
        n_vars=n_vars
    )
    
    fitness_scores_matrix = np.zeros(shape=(no_of_generations, pop_size))
    population_fitness = evaluate(env, population)
    
    # min max scaling on the population fitness to prevent negative fitness scores
    population_fitness_positive = population_fitness + np.abs(np.min(population_fitness)) + 0.00001
    population_fitness_norm = population_fitness_positive / np.max(population_fitness_positive)
    
    fitness_scores_matrix[each_generation, :] = population_fitness
    
    best = np.argmax(population_fitness)
    std  =  np.std(population_fitness)
    mean = np.mean(population_fitness)

    # saves results for first pop
    file_aux  = open(experiment_name+'/results.txt','a')
    file_aux.write('\n\ngen best mean std')
    print( '\n GENERATION '+str(each_generation)+' '+str(round(population_fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(each_generation)+' '+str(round(population_fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()
    
    for each_generation in range(1, no_of_generations):

        print("Generation started " + str(each_generation))

        env.update_solutions(
            [
                population,
                population_fitness
            ]
        )
        
        selected_population = roulette_wheel_selection(
            population,
            population_fitness_norm
        )

        new_offspring = uniform_crossover(
                selected_population,
                dom_u,
                dom_l
            )
        
        new_offspring_fitness = evaluate(env, new_offspring)
 
        population = np.append(
            population,
            new_offspring,
            axis=0
        )

        population_fitness = np.append(
            population_fitness,
            new_offspring_fitness
        )

        if survival_method == "mu_lamda":
            population, population_fitness = survivors_selection_mu_comma_lambda(
                new_offspring,new_offspring_fitness
            )

        elif survival_method == "rr_tournament":
            population, population_fitness = survivors_selection_rr_tournament(
                population, population_fitness
            )
        
        elif survival_method == "rank":
            population, population_fitness = select_survivors(
                population, population_fitness
            )

        else:
            raise ValueError("Survival method not valid")


        fitness_scores_matrix[each_generation, :] = population_fitness


        print("Generation finished " + str(each_generation))

        best = np.argmax(population_fitness)
        std  =  np.std(population_fitness)
        mean = np.mean(population_fitness)


        # saves results
        file_aux  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(each_generation)+' '+str(round(population_fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        file_aux.write('\n'+str(each_generation)+' '+str(round(population_fitness[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        file_aux.close()

        # saves generation number
        file_aux  = open(experiment_name+'/gen.txt','w')
        file_aux.write(str(each_generation))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name+'/best.txt', np.array(population[best]))

        # saves simulation state
        solutions = [population, population_fitness]
        env.update_solutions(solutions)
        env.save_state()

    np.savetxt(
        "fitness_scores_matrix.txt",
        fitness_scores_matrix
    )

    return population, population_fitness


##############################################################################################
#### FUNCTIONS ####
##############################################################################################
    
def generate_population(dom_l, dom_u, npop, n_vars):
    """
        Generate population with the hidden weights of the no. of hidden neurons of 
        the evoman game.
            - output: array or arrays of random normal values for each weight
    """
    
    return np.random.uniform(dom_l, dom_u, (npop, n_vars))


def roulette_wheel_selection(population, fitness_array, number_parents=2):
    """
        Depending on the percentage contribution to the total population, 
        a fitness string is selected for mating to form the next generation.
          - Thus a fitness string with the highest fitness value.
    """
    total_fit = np.sum(fitness_array)
    relative_fitness = [f / total_fit for f in fitness_array]

    selected_indeces = np.random.choice(
        range(len(relative_fitness)), 
        p=relative_fitness, 
        size=number_parents, 
        replace=False
    )
    selected_parents = population[selected_indeces, :] 
    
    return selected_parents

def uniform_crossover(parents, dom_u, dom_l): 
    """
        Perform uniform crossover on parents.
    """

    new_offspring = []

    for p in range(0, parents.shape[0], 2):
        parent_1 = parents[p]
        parent_2 = parents[p+1]

        random_crossover_vector = np.random.randint(
            0, 2, parent_1.shape[0])
        
        child_1 = []
        child_2 = []
        
        for i in range(0, parent_1.shape[0]):
            
            vector_instance = random_crossover_vector[i]
            parent1_instance = parent_1[i]
            parent2_instance = parent_2[i]
            
            if vector_instance == 1:
                child_1.append(
                    parent2_instance
                )
                child_2.append(
                    parent1_instance
                )
            else:
                child_1.append(
                    parent1_instance
                )
                child_2.append(
                    parent2_instance
                )

        child_1 = mutate(child_1, dom_u, dom_l)
        child_2 = mutate(child_2, dom_u, dom_l)

        new_offspring.append(
            np.array(
                child_1
            )
        )
        new_offspring.append(
            np.array(
                child_2
            )
        )
        
    new_offspring_array = np.array(new_offspring)
            
    return new_offspring_array


def mutate(child, dom_u, dom_l, probability=0.2):
    """
        Mutate the offsprings and calculate fitness values for comparison
    """
    for gen in range(0, len(child)):
        if np.random.uniform(0, 1)<= probability:
            noise = np.random.normal(0, 1, 1)
        # add the noise to the child -> update the gene value:
            child += noise
        # make sure that the mutated children are within the range:
            child = np.clip(child, dom_u, dom_l)
    return child


#################################################################   
#### selection functies - Globaal, dus nog niet af!          ####
#################################################################
	
def survivors_selection_mu_comma_lambda (population, parent):
   #Children replace parents (mu, lambda)

   new_population =
   return new_population



def selection_mu_plus_lambda (child, parents, fit_child, fit_parents):
   x = np.concatenate([child, parents])
   f = np.concatenate([fit_child, fit_parents])
   #sort the total population based on their fitness:
   ranks = argsort(f)
   x = x[ranks]
   f = f[ranks]
   return x[:pop_size], f[:pop_size]



def survivors_selection_rr_tournament(population, fit_pop):
   offspring = []
   while len(offspring) != pop_size:
       participant1 = select_participant(population)
       participant2 = select_participant(population)

       fitness_participant1 = evaluate(participant1)
       fitness_participant2 = evaluate(participant2)

       if fitness_participant1 > fitness_participant2:
           offspring[participant1] = population[participant1]
       else:
           offspring[participant2] = population[participant2]
     
       #participant1 = select_participant(parents) 
       #participant2 = select_participant(offspring)

       #if parents[participant1] > offspring[participant2]:
       #    population[participant1] = parents[participant1]
       #else:
       #    population[participant2] = offspring[participant2]

   return offspring


def select_survivors(population, fitness):
    """
        Select survivors from the population
    
        Rank on fitness and choose the best 100
    """
    
    return population[0:10], fitness[0:10]

def select_participant (pop):
   #select a participant
   return np.random.choise(list(pop)) #Choose a random key from population; https://pynative.com/python-random-choice/


def calculate_fitness(env, x):
    """
        Run a simulation on an environment
        and return the player fitness value
            - Returns a value for the fitness
    """

    f, p, e, t = env.play(pcont=x)
    
    return f


def evaluate(env, population):
    """
        Run the simulation algorithm on the population
          - population: an array of arrays
        - return fitness values for every chromosome in the population
    """
    
    return np.array(
        list(
            map(
                lambda y: calculate_fitness(env, y),
                population
            )
        )
    )

#################################################################   
#### END FUNCTIONS ####
#################################################################


genetic_algorithm("rank")
