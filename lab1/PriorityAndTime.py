import pygad
import numpy

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

def fitness_func(ga_instance, solution, solution_idx):
    takes = []
    for (i, v) in enumerate(solution):
        if v == 1:
            takes.append(S[i])
    sumWeight = numpy.sum(v[2] for v in takes)
    sumValue1 = numpy.sum(v[1] for v in takes)
    if (sumWeight > maxWeight):
        fitness = 0
        return fitness
    fitness = sumValue1
    return fitness
    # zgodnosc = numpy.sum(preferencje_nauczycieli[:, solution])
    # return zgodnosc


fitness_function = fitness_func

gene_space = [0, 1]

sol_per_pop = 20
num_genes = len(S)

num_parents_mating = 5
num_generations = 30
keep_parents = 1

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

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

value = 0
for (i, v) in enumerate(solution):
    if v == 1:
        value = value + S[i][1]
        print(S[i][0])
print(value)

ga_instance.plot_fitness()
