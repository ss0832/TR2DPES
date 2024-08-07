import numpy as np

def df_dx(f, x, y, h=1e-7):
    return (f(x + h, y) - f(x - h, y)) / (2 * h)

def df_dy(f, x, y, h=1e-7):
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

def df2_dx2(f, x, y, h=1e-7):
    return (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2

def df2_dy2(f, x, y, h=1e-7):
    return (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2

def df2_dxdy(f, x, y, h=1e-7):
    return (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)

def grad(f, x, y, h=1e-7):
    return np.array([df_dx(f, x, y, h), df_dy(f, x, y, h)]).reshape(2, 1)

def hess(f, x, y, h=1e-7):
    return np.array([df2_dx2(f, x, y, h), df2_dxdy(f, x, y, h), df2_dxdy(f, x, y, h), df2_dy2(f, x, y, h)]).reshape(2, 2)   

