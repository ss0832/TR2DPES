import numpy

def BFGS_hessian_update(old_point, point, old_grad, grad, old_hess):
    s = point - old_point
    y = grad - old_grad
    rho = 1.0 / (numpy.dot(y.T, s) + 1e-10)
    I = numpy.eye(len(point))
    A1 = I - rho * numpy.dot(s, y.T)
    A2 = I - rho * numpy.dot(y, s.T)
    B = rho * numpy.dot(s, s.T)
    d_hess = numpy.dot(numpy.dot(A1, old_hess), A2) + B

    return d_hess