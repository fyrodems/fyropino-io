import pygad
import numpy
import math


def realValuedFunction(a, b, c):
    result = math.sin(a) + math.sqrt(b) * (c ** 2)
    return result

S = [1.00, 2.00, 3.00]

gene_space = numpy.arange(0.00, 10.00, 1.00)


def fitness(ga_instance, solution, ids):
    print(solution[0])
    result = realValuedFunction(solution[0], solution[1], solution[2])
    return result


fitness_function = fitness

sol_per_pop = 10
num_genes = len(S)
num_parents_mating = 5
num_generations = 20
keep_parents = 3
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 33.33

ga_instance = pygad.GA(gene_space=gene_space,
                       num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))

prediction = numpy.sum(S*solution)
print("Predicted output based on the best solution : {prediction}".format(
    prediction=prediction))

ga_instance.plot_fitness()
