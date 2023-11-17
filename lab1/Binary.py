import pygad
import numpy

# Przestrzeń genów określająca możliwe wartości dla każdego genu
gene_space = numpy.arange(0.00, 1.01, 1.00)

# Wektor reprezentowany jako odwrócona liczba binarna, S to wynik idealny
S = [1, 1, 1, 1, 1, 1, 1, 0]

# Funkcja zamieniająca odwrócone liczby binarne na wartość dziesiętną
def to_decimal(number, power):
    if power == 7 and number == 1:
        return -128
    else:
        if number == 0:
            return 0
        else:
            return 2**power

# Funkcja fitness oceniająca jakość rozwiązania
def fitness(ga_instance, solution, ids):
    i = 0
    result = 0

    while i < len(solution):
        result = result + to_decimal(solution[i], i)
        i = i + 1

    return result

# Ustawienia algorytmu genetycznego
fitness_function = fitness
sol_per_pop = 5
num_genes = 8
num_parents_mating = 5
num_generations = 10
keep_parents = 3
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 12.5

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

# Wyświetlenie parametrów najlepszego rozwiązania
print("Parametry najlepszego rozwiązania: {solution}".format(solution=solution))
print("Wartość funkcji fitness najlepszego rozwiązania = {solution_fitness}".format(
    solution_fitness=solution_fitness))

# Przykładowe przewidywane wyjście na podstawie najlepszego rozwiązania
prediction = 127
print("Przewidywane wyjście na podstawie najlepszego rozwiązania: {prediction}".format(
    prediction=prediction))

# Wyświetlenie wykresu ewolucji funkcji fitness w czasie
ga_instance.plot_fitness()
