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
    A = 10
    y = A*2 + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in X])
    return y


def f(x):
    n_particles = x.shape[0]
    j = [objective_function(x[i]) for i in range(n_particles)]
    return np.array(j)


options = {'c1': 1, 'c2': 2, 'w': 0.75}
optimizer = GlobalBestPSO(n_particles=1000, dimensions=2, options=options)


cost, pos = optimizer.optimize(f, iters=100)
cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()

m = Mesher(func=fx.sphere)
animation = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0, 0))

animation.save('plot0.gif', writer='imagemagick', fps=4)
Image(url='plot0.gif')
