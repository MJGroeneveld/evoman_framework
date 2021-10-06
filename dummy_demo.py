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


def genetic_algorithm(survival_method, migration_method, experiment_name):
    """
        Genetic algorithm solver
    """

    n_hidden_neurons = 10

    ini = time.time()  # Set the time marker

    # genetic algorithm parameters
    dom_l = 1
    dom_u = -1
    pop_size = 100
    n_children = 2
    n_parents = 50
    no_of_generations = 30
    each_generation = 0
    sigma = 0.2
    not_improved = 0
    n_islands = 5
    n_mig_indv = 4
    migration_interval = 3  # magration happens every x generations

    enemies = [1,2,3]
    multiplemode = "yes" if len(enemies) > 1 else "no"

    experiment_name = f"output/{experiment_name}"
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Initialize Environment
    env = Environment(
        experiment_name=experiment_name,
        multiplemode=multiplemode,
        enemies=enemies,
        playermode="ai",
        player_controller=player_controller(n_hidden_neurons),
        enemymode="static",
        level=2,
        speed="fastest"
    )
    # check the environment state
    env.state_to_log()

    # No of weights for a multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    population = generate_population(
        dom_l=dom_l,
        dom_u=dom_u,
        n_islands=n_islands,
        npop=pop_size,
        n_vars=n_vars,
    )

    fitness_scores_matrix = np.zeros(shape=(n_islands, no_of_generations, pop_size))

    population_fitness = np.zeros(shape=(n_islands, pop_size))
    population_fitness_positive = np.zeros(shape=(n_islands, pop_size))
    population_fitness_norm = np.zeros(shape=(n_islands, pop_size))
    population_diversity = np.zeros(shape=(n_islands, pop_size))
    # solutions = np.zeros(shape=(n_islands, pop_size, n_vars))
    best_idx = np.zeros(shape=n_islands)
    std = np.zeros(shape=n_islands)
    mean = np.zeros(shape=n_islands)
    overall_best = np.zeros(shape=(n_islands, n_vars))
    overall_best_fitness = np.zeros(shape=n_islands)

    for island in range(0, n_islands):
        print("Start initializing population for island", island)

        population_fitness[island, :] = evaluate(env, population[island, :, :])

        # min max scaling on the population fitness to prevent negative fitness scores
        population_fitness_positive[island, :] = population_fitness[island, :] + np.abs(
            np.min(population_fitness[island, :])) + 0.00001
        population_fitness_norm[island, :] = population_fitness_positive[island, :] / np.max(
            population_fitness_positive[island, :])

        # fitness_scores_matrix[each_generation, :] = population_fitness
        best_idx[island] = np.argmax(population_fitness[island, :])
        std[island] = np.std(population_fitness[island, :])
        mean[island] = np.mean(population_fitness[island, :])

        overall_best[island, :] = population[island, int(best_idx[island]), :]
        overall_best_fitness[island] = population_fitness[island, int(best_idx[island])]
        print("overall best before", overall_best_fitness[island])

        # saves results for first pop
        filepath = f"output/{experiment_name}_island_{island}"
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        file_aux = open(f"{filepath}/results.txt", "a")
        file_aux.write('\n\ngen,best,mean,std')
        print('\n GENERATION ' + str(each_generation) + ' ' + str(
            round(population_fitness[island, int(best_idx[island])], 6)) + ' ' + str(
            round(mean[island], 6)) + ' ' + str(round(std[island], 6)))
        file_aux.write('\n' + str(each_generation) + ',' + str(
            round(population_fitness[island, int(best_idx[island])], 6)) + ',' + str(
            round(mean[island], 6)) + ',' + str(round(std[island], 6)))
        file_aux.close()

    for each_generation in range(1, no_of_generations):

        print("Generation started " + str(each_generation))

        for island in range(0, n_islands):

            print("Island number", island)

            env.update_solutions(
                [
                    population,
                    population_fitness
                ]
            )

            island_new_offspring = uniform_crossover(
                population[island, :, :], population_fitness_norm[island, :], n_parents, n_children,
                dom_u,
                dom_l, sigma
            )

            island_new_offspring_fitness = evaluate(env, island_new_offspring)

            if survival_method == "mu_comma_lambda":
                population[island, :, :], population_fitness[island, :] = survivors_selection_mu_comma_lambda(
                    island_new_offspring, island_new_offspring_fitness, pop_size
                )

            else:
                island_population = np.append(
                    population[island, :, :],
                    island_new_offspring,
                    axis=0
                )

                island_population_fitness = np.append(
                    population_fitness[island, :],
                    island_new_offspring_fitness
                )
                if survival_method == "rr_tournament":
                    population[island, :, :], population_fitness[island, :] = survivors_selection_rr_tournament(
                        island_population, island_population_fitness, pop_size
                    )
                else:
                    raise ValueError("Survival method not valid")

            sigma = adeptive_mutate(sigma, no_of_generations)

            # fitness_scores_matrix[each_generation, :] = population_fitness
            population_diversity[island, :] = calculate_population_diversity(population[island, :, :])
            print("Population diversity", population_diversity[island, :])

            print("Generation finished " + str(each_generation))

            best_idx[island] = np.argmax(population_fitness[island, :])
            std[island] = np.std(population_fitness[island, :])
            mean[island] = np.mean(population_fitness[island, :])

            if population_fitness[island, int(best_idx[island])] >= overall_best_fitness[island]:
                overall_best[island] = population[island, int(best_idx[island]), :]
                overall_best_fitness[island] = population_fitness[island, int(best_idx[island])]
            else:
                not_improved += 1

            # saves results
            file_aux = open(f"{filepath}/results.txt", "a")
            file_aux.write('\n\ngen,best,mean,std')
            print('\n GENERATION ' + str(each_generation) + ' ' + str(
                round(population_fitness[island, int(best_idx[island])], 6)) + ' ' + str(
                round(mean[island], 6)) + ' ' + str(round(std[island], 6)))
            file_aux.write('\n' + str(each_generation) + ',' + str(
                round(population_fitness[island, int(best_idx[island])], 6)) + ',' + str(
                round(mean[island], 6)) + ',' + str(round(std[island], 6)))
            file_aux.close()

            # saves generation number
            file_aux = open(filepath + '/gen.txt', 'w')
            file_aux.write(str(each_generation))
            file_aux.close()

            # saves file with the best solution
            np.savetxt(filepath + '/best.txt', population[island, int(best_idx[island]), :])
            np.savetxt(filepath + '/overall_best.txt', overall_best[island, :])

            # saves simulation state
            solutions = [population[island, :, :], population_fitness[island, :]]
            env.update_solutions(solutions)
            env.save_state()

            if not_improved > 10:
                population[island, :, :] = not_optimizing(population[island, :, :], population_fitness[island, :],
                                                          pop_size, n_vars, dom_l, dom_u)
                not_improved = 0

        if each_generation % migration_interval == 0:
            if migration_method == "random":
                migration_indices = select_random_migration(population, n_mig_indv)

            elif migration_method == "best":
                migration_indices = select_best_migration(population_fitness, n_mig_indv)

            else:
                raise ValueError("No valid migration_method selected")

            population = do_migration(population, migration_indices)


    # np.savetxt(
    #     "fitness_scores_matrix.txt",
    #     fitness_scores_matrix
    # )

    # # Evaluate overall best 5 times
    # file_aux = open(experiment_name + '/champion_boxplot.txt', 'w')
    # file_aux.write('fitness, individual_gain')
    # for i in range(0, 5):
    #     best_fitness = calculate_fitness(env, overall_best)
    #     best_gain = calculate_individual_gain(env, overall_best)
    #     file_aux.write('\n' + str(best_fitness) + ',' + str(best_gain))
    #
    #     print(f"Best individual gain {i}", best_gain)
    #
    # file_aux.close()

    return population, population_fitness


##############################################################################################
#### FUNCTIONS ####
##############################################################################################

def generate_population(dom_l, dom_u, n_islands, npop, n_vars):
    """
        Generate population with the hidden weights of the no. of hidden neurons of
        the evoman game.
            - output: array or arrays of random normal values for each weight
    """

    return np.random.uniform(dom_l, dom_u, (n_islands, npop, n_vars))


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
        size=number_parents,
        replace=False
    )
    selected_parent = population[selected_indeces, :]

    return selected_parent


def uniform_crossover(population, fitness_array_norm, number_parents, n_children, dom_u, dom_l, sigma):
    """
        Perform uniform crossover on parents.
    """

    new_offspring = []
    parents = roulette_wheel_selection(population, fitness_array_norm, number_parents)

    for i in range(1, n_children + 1):
        for p in range(0, parents.shape[0]):
            parent_1 = parents[p]
            parent_2 = parents[(p + i) % number_parents]

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
    tau1 = 1 / (2 * gens ** (1 / 2))
    tau2 = 1 / ((2 * gens) ** (1 / 2) ** (1 / 2))
    new_sigma = sigma * np.exp((tau1 * np.random.normal(0, 1)) + (tau2 * np.random.normal(0, 1)))

    return new_sigma


def survivors_selection_mu_comma_lambda(offspring, fitness, pop_size):
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


def not_optimizing(population, fitness, pop_size, n_vars, dom_l, dom_u):
    n_away = int(pop_size / 3)
    order = np.argsort(fitness)
    orderasc = order[0:n_away]

    for i in orderasc:
        population[i, :] = np.random.uniform(dom_l, dom_u, (1, n_vars))

    return population


def calculate_fitness(env, x):
    """
        Run a simulation on an environment
        and return the player fitness value
            - Returns a value for the fitness
    """

    f, p, e, t = env.play(pcont=x)

    return f


def calculate_individual_gain(env, x):
    """
        Run a simulation on an environment
        and return the player fitness value
            - Returns a value for the fitness
    """

    f, p, e, t = env.play(pcont=x)

    ind_gain = p - e

    return ind_gain


def calculate_population_diversity(population):
    pop_size = len(population)
    diversity_matrix = np.zeros((pop_size, pop_size))

    for i in range(0, pop_size):
        for j in range(0, pop_size):
            diversity_matrix[i, j] = np.sum(np.abs(population[i] - population[j]))

    return np.sum(diversity_matrix)


def select_random_migration(pop, n_mig_indv):
    n_islands = pop.shape[0]
    pop_size = pop.shape[1]

    return {
        island: list(np.random.choice(range(len(pop[island, :, :])), size=n_mig_indv, replace=False))
        for island in range(n_islands)
    }


def select_best_migration(pop_fitness, n_mig_indv):
    n_islands = pop_fitness.shape[0]
    pop_size = pop_fitness.shape[1]

    return {
        island: list(pop_fitness[island, :].argsort())[::-1][0:n_mig_indv]
        for island in range(n_islands)
    }


def do_migration(pop, migration_indices):
    n_islands = pop.shape[0]

    # Making an new array (right size) with all the individuals shifted for migration
    migrants = np.array(
        [pop[island, migration_indices[island], :]
         for island in range(n_islands)]
    )

    # Append migrants to the original population
    shifted_migrants = np.array(
        [migrants[(island - 1) % n_islands, :]
         for island in range(n_islands)]
    )
    new_pop = np.append(pop, shifted_migrants, axis=1)

    # Removing the selected migrants from their original island
    return np.array(
        [np.delete(new_pop[island], migration_indices[island], axis=0)
         for island in range(n_islands)]
    )


def evaluate(env, population):
    """
        Run the simulation algorithm on the population
          - population: an array of arrays
        - return fitness values for every chromosome in the population
    """

    x = np.array(
        list(
            map(
                lambda y: calculate_fitness(env, y),
                population
            )
        )
    )
    print(x)
    return x


#################################################################
#### END FUNCTIONS ####
#################################################################

for run in range(0, 3):
    print(f"############### START RUN {run} ###############")

    # np.random.seed(1)

    survival_method = "rr_tournament"
    migration_method = "best"
    experiment_name = f"{survival_method}_{migration_method}_{run}"
    genetic_algorithm(survival_method, migration_method, experiment_name)
