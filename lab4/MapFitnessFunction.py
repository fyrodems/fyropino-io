from pyswarms.utils.plotters.formatters import Mesher
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (
    plot_cost_history, plot_contour, plot_surface)

from lab4.GenerateMap import genSearchField

size = 100

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 2, 'p': 1}
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=2, options=options)
searchMap = genSearchField(size, 5)


def objective_func(solution, searchField):

    position = solution*size
    pos1 = int(position[0])
    pos2 = int(position[1])
    if pos1 >= size:
        pos1 = size - 1
    if pos2 >= size:
        pos2 = size - 1

    targetHit = searchField[pos1][pos2]
    return targetHit


def f(x):
    n_particles = x.shape[0]
    j = [objective_func(x[i], searchMap) for i in range(n_particles)]
    return np.array(j)


cost, pos = optimizer.optimize(f, iters=100, verbose=True)
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()

m = Mesher(func=fx.sphere)
animation = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0, 0))

animation.save('plot0.gif', writer='imagemagick', fps=4)
Image(url='plot0.gif')
