################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

import sys, os 
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

import time
import numpy as np
from math import fabs, sqrt
import glob, os

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


# Roulette wheel selection

def genetic_algorithm():
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
    npop = 100
    gens = 30
    # mutation = 0.2
    # last_best = 0
    ini_g = 0
    
    # fitness value at each generation change
    expected_value = 0

    no_of_generations = 30

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
        npop=npop,
        n_vars=n_vars
    )
    
    population_fitness = None
    
    for each_generation in range(0, no_of_generations):

        print("Generation started " + str(each_generation))

        population_fitness = evaluate(env, population)

        env.update_solutions(
            [
                population,
                population_fitness
            ]
        )
        
        selected_population = roulette_wheel_selection(
            population,
            population_fitness
        )

	new_offspring = uniform_crossover(
            selected_population,
            dom_u,
            dom_l
        )
 
	new_offspring_fitness = evaluate(env, new_offspring)
 
        population = np.vstack(
            population,
            new_offspring
        )
 
 
        population_fitness = np.append(
            population_fitness,
            new_offspring_fitness
        )


        survival_selection1 = survivors_selection_mu_comma_lambda(
            new_offspring,
            new_offspring_fitness
        )
	
	survival_selection2 = select_survivors(
            population,
            population_fitness
        )

        evaluate_fitness = evaluate(
            env,
            mutate_population
        )

        #if np.sum(evaluate_fitness) <= expected_value:
        #    break
        #else:
        #    expected_value = np.sum(evaluate_fitness)

        population = mutate_population

        print("Generation finished " + str(each_generation))

    np.savetxt(
        "test_population.txt",
        population
    )

    np.savetxt(
        "test_fitness_scores.txt",
        fitness
    )

    return population, fitness

    
def generate_population(dom_l, dom_u, npop, n_vars):
    """
        Generate population with the hidden weights of the no. of hidden neurons of 
        the evoman game.
            - output: array or arrays of random normal values for each weight
    """
    
    return np.random.uniform(dom_l, dom_u, (npop, n_vars))


def roulette_wheel_selection(population_set, fitness_list):
    """
        Depending on the percentage contribution to the total population, 
        a fitness string is selected for mating to form the next generation.
          - Thus a fitness string with the highest fitness value.
    """
   total_fit = fitness_list.sum()

   prob_list = fitness_list/total_fit

   #Notice there is the chance that a parent. mates with oneself
   parent_list_a = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
   parent_list_b = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)


   parent_list_a = population_set[parent_list_a]
   parent_list_b = population_set[parent_list_b]

   return np.array([parent_list_a,parent_list_b])


def uniform_crossover(parents, dom_u, dom_l):
    """
    """

    new_offspring = []

    for p in range(0, parents.shape[0], 2):
        
        parent_1 = parents[p]
        parent_2 = parents[p+1]

        random_crossover_vector = np.random.randint(
            0, 2, parent_1.shape[0]
        )

        child_1 = np.array([])
        child_2 = np.array([])
        for threshold in random_crossover_vector:
            if threshold == 1:
                np.append(
                    child_1,
                    parent_2[
                        random_crossover_vector.index(
                            threshold
                        )
                    ]
                )

                np.append(
                    child_2,
                    parent_1[
                        random_crossover_vector.index(
                            threshold
                        )
                    ]
                )
            else:
                np.append(
                    child_1,
                    parent_1[
                        random_crossover_vector.index(
                            threshold
                        )
                    ]
                )

                np.append(
                    child_2,
                    parent_2[
                        random_crossover_vector.index(
                            threshold
                        )
                    ]
                )

        child_1 = mutate(child_1, dom_u, dom_l)
        child_2 = mutate(child_2, dom_u, dom_l)

        new_offspring.append(
            child_1
        )

        new_offspring.append(
            child_2
        )

    return new_offspring, evaluate(new_offspring)


#################################################################   
#### selection functies - Globaal, dus nog niet af!          ####
#################################################################
#def selection(x_old, x_children, f_old, f_children):
   '''This function selects a number of genes in the total population to go to the next
  
   Input:
   x_old - array of population genes
   x_children - array of children genes 
   f_old - array of population fitness
   f_children - array of children fitness
  
   Output:
   x - array of genes which survived
   f - array of fitness which survived '''
	
def survivors_selection_mu_comma_lambda (child, fit_child):
   #Children replace parents (mu, lambda):
   x = child
   f = fit_child #you get fit_child from the evaluate function (fitness function)
   #sort the children based on their fitness:
   ranks = argsort(f)
   x = x[ranks]
   f = f[ranks]
   return x[:pop_size], f[:pop_size]



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
   while len(offspring) != population_size:
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


#################################################################   
#### I created something else. Need to check: 

def schedule(pop, fit_pop):
    #first create a schedule 
    schedule = []
    n = len(pop)
    if n % 2 == 1: #wanneer de lengte van de population even is  (players = players + [None] --> kan er nog bij als het dus niet even is)
       map = list(range(n)) #maakt een lijst aan -> wanneer n = 4: [0, 1, 2, 3]
       mid = n // 2         #het midden van de lijst 
    for i in range(n-1): 
        l1 = map[:mid]      #[0, 1]
        l2 = map[mid:]      #[2, 3]
        l2.reverse()        #[3, 2]
        round = []
            for j in range(mid): 
                t1 = pop[l1[j]] #hier voer je dan in welke speler tegen welke speler gaat 
                t2 = pop[l2[j]]
                if j == 0 and i % 2 == 1: 
                    round.append((t2, t1)) #hier voeg je dus wie tegen wie speelt in iedere ronde. Dit stopt zodra iedereen tegen iedereen heeft gespeelt. 
                else: 
                    round.append((t1, t2)) 
            schedule.append(round)
            map = map[mid:-1] + map[:mid] + map[-1:]
	    # [2, 0, 1, 3]
            # [1, 2, 0, 3]
            # [0, 1, 2, 3]
    return schedule 

def tournament(schedule, fit_pop):
    for round in schedule:
    	for m in round:
        	print(m[0], m[1])
    #1 4
    #2 3
    #4 3
    #1 2
    #2 4
    #3 1
    offspring = []
    #hier moet dus nog een for loopje? 
    if fit_pop[schedule([0][0][0]] > fit_pop[schedule([0][0][1])]: #dan krijg je de eerste van schedule en daarvan de fitness
        offspring.append(schedule[0][0][0])
    else: 
        offspring.append(schedule[0][0][1])
    return offspring


#bij elke individu in pop hoort een fit_pop. Wanneer die fit_pop > is dan die van de tegenstander dan stop je hem in een lijst.  ####
#################################################################


def select_participant (pop):
   #select a participant
   return np.random.choise(list(pop)) #Choose a random key from population; https://pynative.com/python-random-choice/


#################################################################   
#### END FUNCTIONS ####
#################################################################



def select_survivors(population, fitness):
    """
        Select survivors from the population
    
        Rank on fitness and choose the best 100
    """
    
    return population[0:99]


def mutate(child, dom_u, dom_l, probability=0.2):
    """
        Mutate the offsprings and calculate fitness values for comparison
    """
    
    for j in range(0, len(child)):
        if np.random.uniform(0, 1) <= probability:
            child[j] = child[j]+np.random.normal(0, 1)

    child = np.array(list(map(lambda y: limits(dom_u, dom_l, y), child)))
    
    return child


# limits
def limits(dom_l, dom_u, x):
    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x


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
