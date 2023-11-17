import pygad
import numpy
import math

# Definicja funkcji rzeczywistej, której parametry będą optymalizowane
def realValuedFunction(a, b, c):
    result = math.sin(a) + math.sqrt(b) * (c ** 2)
    return result

# Zadane wartości parametrów, dla których optymalizujemy funkcję
S = [10.00, 10.00, 10.00]

# Przestrzeń genów - zakres, z którego są generowane wartości parametrów
gene_space = numpy.arange(0.00, 10.01, 1.00)

# Funkcja fitness, oceniająca jakość chromosomu
def fitness(ga_instance, solution, ids):
    # Wypisanie wartości pierwszego genu (parametru a)
    print(solution[0])
    # Obliczenie wartości funkcji rzeczywistej dla danego zestawu parametrów
    result = realValuedFunction(solution[0], solution[1], solution[2])
    return result

# Przypisanie funkcji fitness do zmiennej fitness_function
fitness_function = fitness

# Parametry algorytmu genetycznego
sol_per_pop = 10
num_genes = len(S)
num_parents_mating = 5
num_generations = 20
keep_parents = 3
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 33.33

# Inicjalizacja instancji algorytmu genetycznego
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

# Uruchomienie algorytmu genetycznego
ga_instance.run()

# Pobranie najlepszego rozwiązania
solution, solution_fitness, solution_idx = ga_instance.best_solution()

# Wypisanie parametrów najlepszego rozwiązania i ich wartości funkcji fitness
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(
    solution_fitness=solution_fitness))

# Przewidywanie na podstawie najlepszego rozwiązania
prediction = numpy.sum(S * solution)
print("Predicted output based on the best solution : {prediction}".format(
    prediction=prediction))

# Wygenerowanie wykresu zmiany wartości funkcji fitness w kolejnych generacjach
ga_instance.plot_fitness()
