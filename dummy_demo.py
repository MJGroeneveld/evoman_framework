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

#def evaluate_candidate_population
#Fitness function


#def select_parents(x_old, f_old):
    '''This function selects the parents from a population  Select the two parents to crossover. The parents are selected in a way similar to a ring. The first with indices 0 
    and 1 are selected at first to produce two offspring. If there still remaining offspring to produce, then we select parent 1 with parent 2 to produce another two offspring 
    
    Input: 
    x_old  -  array of population genes 
    y_old  -  array of population fitness 
    
    Output: 
    x_parents - array of parents genes 
    f_parents - array of parents fitness '''

#    return x_parents, f_parents 

def create_children(parent1, parent2):
    children = []
    for i in range(n_children):
                randomcrossover_vector = np.random.randint(0, 2, size = np.array(..).shape)
                for j in randomcrossover_vector:
                    if j == 1:
                        temporaty_store = parent1[j] 
                        parent1[j] = parent2[j]
                        parent2[j] = temporaty_store

                child1 = parent1
                child2 = parent2       
            #children must be appended to the list
            children.append(child1, child2)
    return np.array(children)

def recombination(array_parents):
    offspring = array_parents
    while len(array_parents) > 0:
        parent1 = array_parents.pop()
        parent2 = array_parents.pop()
        children = create_children(parent1, parent2)
        offspring.append(children)    
    return offspring

def mutation (x): 
    '''This function mutates the new children with Gaussian noise
    
    Input: 
    x - array of children genes 
    
    Output: 
    x - array of mutated children genes '''

    #mutate each child:
    for i in x:
    # get a random noise scaled to the gene range
        if np.rndom.uniform(0, 1)<=mutation:
            noise = np.random.normal(0, std)*bound
        # add the noise to the child -> update the gene value:
            i += noise
        # make sure that the mutated children are within the range:
            i = np.clip(i, bounds_min, bounds_max)
    return x


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

#    return x, f

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
mutation = 0.2
last_best = 0

# initializes population generating new ones
if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
    fit_pop = evaluate(pop)
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
    fit_offspring = evaluate(offspring)   # evaluation
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
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
