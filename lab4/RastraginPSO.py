import math
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.functions import single_obj as fx
import matplotlib.pyplot as plt


# Funkcja celu z dostosowywalnymi parametrami
def calculate_objective(X, A=10):
    return A * 2 + np.sum([(x ** 2 - A * np.cos(2 * math.pi * x)) for x in X])


# Zmodyfikowana funkcja celu, aby akceptować dostosowywalne parametry
def calculate_modified_objective(X, A=10):
    return np.array([calculate_objective(x, A) for x in X])


# Zaawansowany optymalizator z możliwością dostosowania parametrów
def initialize_optimizer(n_particles=1000, dimensions=2, options=None):
    return GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options)


# Niestandardowy wykres historii kosztów z dodatkowym stylem
def plot_cost_history(cost_history):
    plt.plot(cost_history, label='Historia Kosztów', color='b')
    plt.title('Proces Optymalizacji')
    plt.xlabel('Iteracje')
    plt.ylabel('Koszt')
    plt.legend()
    plt.grid(True)
    plt.show()


# Przeszukiwanie siatkowe różnych kombinacji parametrów
def grid_search_best_params(param_grid, objective_function, dimensions=2, iterations=100):
    best_cost = float('inf')
    best_params = None

    for params in param_grid:
        options = {'c1': params[0], 'c2': params[1], 'w': params[2]}
        optimizer = initialize_optimizer(dimensions=dimensions, options=options)
        cost, _ = optimizer.optimize(objective_function, iters=iterations)

        if cost < best_cost:
            best_cost = cost
            best_params = params

    return best_params


# Przetestuj różne kombinacje parametrów
param_grid = [(1, 2, 0.75), (1.5, 2.5, 0.9), (1.2, 2.0, 0.8)]  # Dodaj więcej kombinacji według potrzeb
best_parameters = grid_search_best_params(param_grid, calculate_modified_objective, dimensions=2, iterations=100)

# Użyj najlepszych parametrów do wykonania optymalizacji
best_optimizer = initialize_optimizer(dimensions=2, options={'c1': best_parameters[0], 'c2': best_parameters[1],
                                                             'w': best_parameters[2]})
best_cost, best_positions = best_optimizer.optimize(calculate_modified_objective, iters=150)
best_cost_history = best_optimizer.cost_history

# Narysuj historię kosztów z najlepszymi parametrami
plot_cost_history(best_cost_history)

# Stwórz Meshera
# m = Mesher(func=fx.rastrigin)
m = Mesher(func=fx.sphere)

# Narysuj kontur z Mesherem i zoptymalizowanymi pozycjami przy użyciu najlepszych parametrów
custom_animation = plot_contour(pos_history=best_optimizer.pos_history,
                                mesher=m,
                                mark=(0, 0))

# Zapisz animację
custom_animation.save('best_plot.gif', fps=4)

import matplotlib.animation as animation

# Funkcja do aktualizacji ramki animacji
def update(frame, line, iterations_text):
    line.set_data(*best_optimizer.pos_history[:, :, frame])
    iterations_text.set_text(f'Iteracje: {frame + 1}')
    return line, iterations_text

# Inicjalizacja wykresu
fig, ax = plt.subplots()
ax.set_title('Animacja Optymalizacji')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Inicjalizacja linii trajektorii i tekstu z liczbą iteracji
line, = ax.plot([], [], lw=2)
iterations_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Ustawienia osi
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Funkcja inicjalizująca animację
def init():
    line.set_data([], [])
    iterations_text.set_text('')
    return line, iterations_text

# Utworzenie animacji
ani = animation.FuncAnimation(fig, update, frames=len(best_optimizer.pos_history[0, 0, :]),
                              fargs=(line, iterations_text), init_func=init, blit=True)

# Zapisz animację
ani.save('trajectory_animation.gif', writer='imagemagick', fps=4)

