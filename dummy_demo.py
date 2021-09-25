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
    pop_size = 40
    n_children = 2
    n_parents = 10
    no_of_generations = 20
    each_generation = 0
    sigma = 0.2
    not_improved = 0
        

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

        new_offspring = uniform_crossover(
                population, population_fitness_norm, n_parents, n_children,
                dom_u,
                dom_l, sigma
            )  

        new_offspring_fitness = evaluate(env, new_offspring)

        if survival_method == "mu_comma_lambda":
            population, population_fitness = survivors_selection_mu_comma_lambda(
                new_offspring, new_offspring_fitness, pop_size
            )

        else:
            population = np.append(
                population,
                new_offspring,
                axis=0
            )

            population_fitness = np.append(
            population_fitness,
            new_offspring_fitness
            )
            if survival_method == "rr_tournament":
                population, population_fitness = survivors_selection_rr_tournament(
                    population, population_fitness, pop_size
                )
            
            elif survival_method == "rank":

                population, population_fitness = select_survivors(
                    population, population_fitness
                )

            else:
                raise ValueError("Survival method not valid")
        
        print("old sigma", sigma)
        sigma = adeptive_mutate(sigma, no_of_generations)
        print("new sigma", sigma)

        fitness_scores_matrix[each_generation, :] = population_fitness
        population_diversity = calculate_population_diversity(population)
        print("Population diversity", population_diversity)

        print("Generation finished " + str(each_generation))

        last_best = best
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

        if last_best >= best:
            not_improved += 1
        if not_improved > 10:
            population = selection_pressure(population, population_fitness, pop_size, n_vars, dom_l, dom_u)

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

def roulette_wheel_selection2(population, fitness_array_norm):
    """
        Depending on the percentage contribution to the total population, 
        a fitness string is selected for mating to form the next generation.
          - Thus a fitness string with the highest fitness value.
    """
    total_fit = np.sum(fitness_array_norm)
    relative_fitness = [f / total_fit for f in fitness_array_norm]

    selected_indeces = np.random.choice(
        range(len(relative_fitness)), 
        p=relative_fitness,
        replace=False
    )
    selected_parent = population[selected_indeces, :] 
    # population = np.delete(population, selected_indeces, axis=0)
    # population_fitness = np.delete(population_fitness, selected_indeces, axis=0)
    
    return selected_parent#, population, population_fitness


def roulette_wheel_selection(population, fitness_array_norm, number_parents):
    """
        Depending on the percentage contribution to the total population, 
        a fitness string is selected for mating to form the next generation.
          - Thus a fitness string with the highest fitness value.
    """
    total_fit = np.sum(fitness_array_norm)
    relative_fitness = [f / total_fit for f in fitness_array_norm]

    selected_indeces = np.random.choice(
        range(len(relative_fitness)), 
        p=relative_fitness,
        size = number_parents, 
        replace=False
    )
    selected_parent = population[selected_indeces, :] 
    # population = np.delete(population, selected_indeces, axis=0)
    # population_fitness = np.delete(population_fitness, selected_indeces, axis=0)
    
    return selected_parent#, population, population_fitness

def uniform_crossover(population, fitness_array_norm, number_parents, n_children, dom_u, dom_l, sigma): 
    """
        Perform uniform crossover on parents.
    """

    new_offspring = []
    parents = roulette_wheel_selection(population, fitness_array_norm, number_parents)
    
    for i in range(1, n_children+1):
        for p in range(0, parents.shape[0]):
            parent_1 = parents[p]
            parent_2 = parents[(p+i)%number_parents]

            random_crossover_vector = np.random.randint(
                0, 2, parent_1.shape[0])

            inverse_crossover_vector = (random_crossover_vector - 1) * -1


            child_1 = (parent_1 * random_crossover_vector) + (parent_2 * inverse_crossover_vector)
            child_2 = (parent_1 * inverse_crossover_vector) + (parent_2 * random_crossover_vector)

            child_1 = mutate(child_1, dom_u, dom_l, sigma)
            child_2 = mutate(child_2, dom_u, dom_l, sigma)
        

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

def uniform_crossover2(population, fitness_array_norm, n_children, dom_u, dom_l, sigma): 
    """
        Perform uniform crossover on parents.
    """

    new_offspring = []

    for p in range(0, population.shape[0]):
        parent_1 = roulette_wheel_selection2(population, fitness_array_norm)

        for i in range(0, int(n_children/2)):
            parent_2 = roulette_wheel_selection2(population, fitness_array_norm)
            random_crossover_vector = np.random.randint(
                0, 2, parent_1.shape[0])

            inverse_crossover_vector = (random_crossover_vector - 1) * -1


            child_1 = (parent_1 * random_crossover_vector) + (parent_2 * inverse_crossover_vector)
            child_2 = (parent_1 * inverse_crossover_vector) + (parent_2 * random_crossover_vector)

            child_1 = mutate(child_1, dom_u, dom_l, sigma)
            child_2 = mutate(child_2, dom_u, dom_l, sigma)
        

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


def mutate(child, dom_u, dom_l, mutation_step_size, probability=0.2):
    """
        Mutate the offsprings and calculate fitness values for comparison
    """

    probability_mask = np.random.uniform(0, 1, len(child)) < probability
    noise = np.random.normal(0, mutation_step_size, size=len(child)) * probability_mask
    child += noise
    
    child = np.clip(child, dom_u, dom_l)
    
    return child

def adeptive_mutate(sigma, gens):

    tau1 = 1/(2 * gens**(1/2))
    tau2 = 1/((2 * gens)**(1/2)**(1/2))
    new_sigma = sigma * np.exp((tau1 * np.random.normal(0,1))+ (tau2 * np.random.normal(0,1)))
    
    return new_sigma

def survivors_selection_mu_comma_lambda (offspring, fitness, pop_size):
   #sort the children offspring based on their fitness:
   ranks = fitness.argsort()[::-1]
   new_population = offspring[ranks]
   new_population_fitness = fitness[ranks]

   return new_population[0:pop_size], new_population_fitness[0:pop_size]


def tournament(agent, pop, fit_pop, number_of_games):
    win = 0
    r_chosen = np.random.choice(
            range(len(pop)), 
            size=number_of_games, 
            replace=False
        )
    for game in range(number_of_games):
        if fit_pop[agent] > fit_pop[r_chosen[game]]:
            win = win + 1
    return win

def survivors_selection_rr_tournament(pop, fit_pop, pop_size):
    number_of_wins_array = np.zeros(shape=(len(pop), 2))
    number_of_games = 10

    for agent in range(len(pop)):
        number_of_wins = tournament(agent, pop, fit_pop, number_of_games)
        number_of_wins_array[agent, :] = agent, number_of_wins
    
    number_of_wins_array = number_of_wins_array[number_of_wins_array[:, 1].argsort()][::-1]
    selected = number_of_wins_array[0:pop_size, 0].astype("int")

    return pop[selected, :], fit_pop[selected]


def selection_pressure(population, fitness, pop_size, n_vars, dom_l, dom_u):
    
    n_away = pop_size/3
    order = np.argsort(fitness)
    orderasc = order[0, n_away]

    for i in orderasc:
        population[i, :] = np.random.uniform(dom_l, dom_u, (pop_size, n_vars))


    return population

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

def calculate_population_diversity(population):
    pop_size = len(population)
    diversity_matrix = np.zeros((pop_size, pop_size))

    for i in range(0, pop_size):
        for j in range(0, pop_size):
            diversity_matrix[i, j] = np.sum(np.abs(population[i] - population[j]))

    return np.sum(diversity_matrix)


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


genetic_algorithm("rr_tournament")
