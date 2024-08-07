import numpy as np

def muller_brown_potential(x, y):
    energy = 0.0
    A = [-200.0, -100.0, -170.0, 15.0]
    a = [-1.0, -1.0, -6.5, 0.7]
    b = [0.0, 0.0, 11.0, 0.6]
    c = [-10.0, -10.0, -6.5, 0.7]
    x0 = [1.0, 0.0, -0.5, -1.0]
    y0 = [0.0, 0.5, 1.5, 1.0]
    for i in range(4):
        energy += A[i] * np.exp(a[i] * (x - x0[i])**2 + b[i] * (x - x0[i]) * (y - y0[i]) + c[i] * (y - y0[i])**2)
    return energy

