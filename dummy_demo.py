################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

#imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

#imports other liberies
import time

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'individual_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

experiment_name = 'dummy_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[7],
                  multiplemode="yes",
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")
env.play()


#################################################################
#### FUNCTIES ####
#################################################################

def evaluate_candidate(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

def select_parents_Roulette_Wheel(population_set, fitness_list):  #Roulette Wheel Selection
    total_fit = fitness_list.sum()
    prob_list = fitness_list/total_fit
    
    #Notice there is the chance that a parent. mates with oneself
    parent_list_a = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
    parent_list_b = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
    
    parent_list_a = population_set[parent_list_a]
    parent_list_b = population_set[parent_list_b]

    return np.array([parent_list_a,parent_list_b])


def select_parents_by_rank(population_set, fitness_list, n_pairs): #Rank Selection
    n = len(population_set)
    rank_sum = n * (n + 1) / 2
    prob_list = [i/rank_sum for i in range(1,101)][::-1]

    f = fitness_list.argsort()
    
    #Notice there is the chance that a progenitor. mates with oneself
    parent_list_a = np.random.choice(f, n_pairs, p=prob_list, replace=True)
    parent_list_b = np.random.choice(f, n_pairs, p=prob_list, replace=True)
    
    parent_list_a = population_set[parent_list_a]
    parent_list_b = population_set[parent_list_b]
    
    
    return np.array([parent_list_a,parent_list_b])


def tournament(fitness_list):
		fit1, ch1 = fitness_list[random.randint(0, len(fitness_list) - 1)]
		fit2, ch2 = fitness_list[random.randint(0, len(fitness_list) - 1)]

		return ch1 if fit1 > fit2 else ch2


def select_parents_by_tournament(fitness_list): #I think we have to give the parent_list as input here as well
    while 1:
        parent1 = tournament(fitness_list)
        parent2 = tournament(fitness_list)
        yield (parent1, parent2)


def recombination(array_parents):
    offspring = array_parents
    children = []
    for p in range(0,array_parents.shape[0], 2):
        parent1_idx = np.random.randint(array_parents.shape[0], size=1)
        parent1 = array_parents[parent1_idx, :]
        array_parents = np.delete(array_parents, parent1_idx, axis=0)
        
        parent2_idx = np.random.randint(array_parents.shape[0], size=1)
        parent2 = array_parents[parent2_idx, :]
        array_parents = np.delete(array_parents, parent2_idx, axis=0)
        for i in range(2):
                randomcrossover_vector = np.random.randint(0, 2, size = np.array(parent1).shape)
                for j in randomcrossover_vector:
                    for x in j:
                        idx = 0
                        if x == 1:
                            temporaty_store = parent1[idx] 
                            parent1[idx] = parent2[idx]
                            parent2[idx] = temporaty_store
                        idx = idx + 1

                    child1 = mutation(parent1)
                    child2 = mutation(parent2)       
                
                #children must be appended to the list
                    children.append(child1)
                    children.append(child2)

        offspring.append(children)    
    return offspring

def mutation (child): 
    '''This function mutates the new children with Gaussian noise
    
    Input: 
    x - array of children genes 
    
    Output: 
    x - array of mutated children genes '''

    if np.random.uniform(0, 1)<= mutation_rate:
            noise = np.random.normal(0, 1)
            #noise = np.random.normal(0, std)*bound
        # add the noise to the child -> update the gene value:
            child += noise
        # make sure that the mutated children are within the range:
            child = np.clip(child, bounds_min, bounds_max)
    return child

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
	
def selection_mu_comma_lambda (child, parents, fit_child):
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

def tournament(parents, offspring, fit_pop): 
    population = []
    while len(population) != population_size:
        participant1 = select_participant(parents) 
        participant2 = select_participant(offspring)

        fitness_participant1 = evaluate(participant1)
        fitness_participant2 = evaluate(participant2)

        if fitness_participant1 > fitness_participant2: 
            population[participant1] = parents[participant1]
        else: 
            population[participant2] = offspring[participant2]
       
        #participant1 = select_participant(parents)  
        #participant2 = select_participant(offspring)

        #if parents[participant1] > offspring[participant2]: 
        #    population[participant1] = parents[participant1]
        #else: 
        #    population[participant2] = offspring[participant2]

    return population 

def select_participant (pop): 
    #select a participant
    return np.random.choise(list(pop)) #Choose a random key from population; https://pynative.com/python-random-choice/

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


#################################################################    
#### END FUNCTIONS ####
#################################################################

#Variables
num_generations = 25 
pop_size = 25
bounds_min = [-2., 0., -5., 0.]
bounds_max = [10., 10., 20., 2500.]
dom_u = 1
dom_l = -1
gens = 100
mutation_rate = 0.2
last_best = 0
bound = np.array(bounds_max) - np.array(bounds_min)

# initializes population generating new ones
if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate_candidate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

#Use the values of the hyperparameters for which you obtained the best results 
# do not iterate over them 

# saves results for first pop
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):

    offspring = crossover(pop)  # crossover
    fit_offspring = evaluate_candidate(offspring)   # evaluation
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate_candidate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    fit_pop_cp = fit_pop
    fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
    probs = (fit_pop_norm)/(fit_pop_norm).sum()
    chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    chosen = np.append(chosen[1:],best)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]


    # searching new areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:

        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop,fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()



fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state
