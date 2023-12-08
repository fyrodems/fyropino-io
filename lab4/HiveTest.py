from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import plot_contour
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

# Ustawienia algorytmu PSO
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = GlobalBestPSO(n_particles=1000, dimensions=2, options=options)

# Funkcja celu do optymalizacji
def fitness_func(solution):
    fitness = (solution + solution % 5) % 3
    return -fitness

# Funkcja wrapper dla funkcji celu
def f(x):
    n_particles = x.shape[0]
    j = [fitness_func(x[i]) for i in range(n_particles)]
    return np.array(j)

# Optymalizacja funkcji celu za pomocą algorytmu PSO
cost, pos = optimizer.optimize(f, iters=25)

# Tworzenie animacji konturu zmian pozycji cząsteczek
animation = plot_contour(pos_history=optimizer.pos_history, mark=(0, 0))

# Zapisanie animacji do pliku GIF
animation.save('plot0.gif', writer='imagemagick', fps=4)

# Wyświetlenie animacji w notatniku Jupyter
Image(url='plot0.gif')
