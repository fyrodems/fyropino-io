import random
import math
import matplotlib.pyplot as plt
import numpy as np
import pyswarms as ps
from IPython.display import Image
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (
    plot_cost_history, plot_contour, plot_surface)
from pyswarms.single.global_best import GlobalBestPSO


def objective_function(X):
    # Funkcja Rastrigina
    A = 10
    y = A * 2 + sum([(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X])
    return y


def f(x):
    n_particles = x.shape[0]
    j = [objective_function(x[i]) for i in range(n_particles)]
    return np.array(j)


def run_experiment(num_particles, dimensions, options, iterations, experiment_name):
    # Inicjalizacja optymalizatora PSO
    optimizer = GlobalBestPSO(n_particles=num_particles, dimensions=dimensions, options=options)

    # Optymalizacja funkcji celu
    cost, pos = optimizer.optimize(f, iters=iterations)
    cost_history = optimizer.cost_history

    # Wygenerowanie wykresu historii kosztu
    plot_cost_history(cost_history)
    plt.title(f'Cost History - {experiment_name}')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig(f'cost_history_{experiment_name}.png')
    plt.show()

    # Stworzenie meshera
    m = Mesher(func=fx.sphere)

    # Wygenerowanie animacji trajektorii ruchu cząstek
    animation = plot_contour(pos_history=optimizer.pos_history,
                             mesher=m,
                             mark=(0, 0))
    animation.save(f'plot_{experiment_name}.gif', writer='imagemagick', fps=4)
    Image(url=f'plot_{experiment_name}.gif')


# Eksperyment 1: Wpływ Liczby Cząstek
options1 = {'c1': 1, 'c2': 2, 'w': 0.75}
run_experiment(num_particles=500, dimensions=2, options=options1, iterations=100, experiment_name='experiment1')

# Eksperyment 2: Wpływ Parametrów Algorytmu
options2 = {'c1': 1.5, 'c2': 1.5, 'w': 0.9}
run_experiment(num_particles=800, dimensions=2, options=options2, iterations=100, experiment_name='experiment2')

# Eksperyment 3: Wpływ Inicjalizacji Cząstek
options3 = {'c1': 0.5, 'c2': 1.8, 'w': 0.5}
run_experiment(num_particles=1000, dimensions=2, options=options3, iterations=100, experiment_name='experiment3')
