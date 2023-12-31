import pygad
import numpy

# Definicja zestawu danych - przedmiotów (nazwa, waga, wartość)
S = [("Chemia Organiczna", 7, 55),
    ("Statystyka", 3, 85),
    ("Logika", 6, 95),
    ("Ekologia", 5, 30),
    ("Medycyna", 8, 65),
    ("Psychiatria", 1, 40),
    ("Inżynieria Materiałowa", 9, 90),
    ("Malarstwo", 3, 15),
    ("Historia Sztuki", 6, 80),
    ("Literatura Porównawcza", 7, 50),
    ("Teoria Muzyki", 10, 10),
    ("Rachunek Różniczkowy", 4, 75),
    ("Teoria Gier", 2, 55),
    ("Historia Naturalna", 5, 60),
    ("Inżynieria Elektryczna", 8, 25),
    ("Psychometria", 9, 40),
    ("Teoria Chaosu", 7, 85),
    ("Genetyka", 6, 50),
    ("Bioinformatyka", 3, 75),
    ("Mikroekonomia", 1, 30),
    ("Logika Matematyczna", 10, 95),
    ("Psychologia Społeczna", 2, 45),
    ("Teoria Grafov", 4, 20),
    ("Antropologia Kulturowa", 8, 70),
    ("Astrofizyka", 6, 60),
    ("Zarządzanie Projektami", 9, 45),
    ("Teoria Liczb", 5, 80),
    ("Teoria Koloru", 3, 35)]

maxWeight = 500

# Funkcja fitness - ocenia jakość rozwiązania na podstawie wartości przedmiotów
def fitness_func(ga_instance, solution, solution_idx):
    takes = []
    for (i, v) in enumerate(solution):
        if v == 1:
            takes.append(S[i])
    sumWeight = numpy.sum(item[2] for item in takes)
    sumValue = numpy.sum(item[1] for item in takes)
    if (sumWeight > maxWeight):
        fitness = 0
        return fitness
    fitness = sumValue
    return fitness

# Przypisanie funkcji fitness do zmiennej fitness_function
fitness_function = fitness_func

# Przestrzeń genów - chromosomy to listy binarne
gene_space = [0, 1]

# Parametry algorytmu genetycznego
sol_per_pop = 20
num_genes = len(S)
num_parents_mating = 5
num_generations = 30
keep_parents = 1
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 10

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

# Wypisanie nazw wybranych przedmiotów w optymalnym rozwiązaniu
value = 0
for (i, v) in enumerate(solution):
    if v == 1:
        value = value + S[i][1]
        print(S[i][0])
print(value)

# Wygenerowanie wykresu przedstawiającego ewolucję wartości fitness w kolejnych generacjach
ga_instance.plot_fitness()
