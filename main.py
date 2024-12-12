import mb_pot
import derivatives
import bfgs

import matplotlib.pyplot as plt
import numpy as np

iteration = 500
threshold = 1e-6
calcFC = 5

trust_radius = 0.01
min_trust_radius = 1e-10
max_trust_radius = 0.3

print("Trust Radius method (Dogleg method) for Muller-Brown potential")
init_point = np.array([[-1.423499], [-0.928038]]) #initial point
init_energy = mb_pot.muller_brown_potential(init_point[0], init_point[1])
init_grad = derivatives.grad(mb_pot.muller_brown_potential, init_point[0], init_point[1])
init_hess = derivatives.hess(mb_pot.muller_brown_potential, init_point[0], init_point[1])
point_list = [init_point]
energy_list = [init_energy]

energy = init_energy
grad = init_grad
hessian = init_hess
point = init_point

for j in range(iteration):
    
    print("Iteration: ", j)
    print("trust_radius: ", trust_radius)
    p_U = -1 * np.dot(grad.T, grad) / np.dot(grad.T, np.dot(hessian, grad)) * grad

    if np.linalg.norm(p_U) >= trust_radius:
        print("p_U >= trust_radius")
        step = -1 * trust_radius * grad / np.linalg.norm(grad)
    else:
        p_B = -1 * np.dot(np.linalg.pinv(hessian), grad)
        if np.linalg.norm(p_B) <= trust_radius:
            print("p_B <= trust_radius")
            step = p_B
        else:
            tau = (trust_radius - np.linalg.norm(p_U)) / (np.linalg.norm(p_B - p_U))
            if tau <= 1.0:
                print("0 < tau <= 1")
                step = tau * p_U
            else:
                print("tau > 1")
                step = p_U + (tau - 1) * (p_B - p_U)
    old_point = point
    point = point + step
    old_energy = energy
    
    old_grad = grad
    old_hess = hessian
    
    energy = mb_pot.muller_brown_potential(point[0], point[1])
    grad = derivatives.grad(mb_pot.muller_brown_potential, point[0], point[1])
    if j % calcFC == 0 or j == 0: 
        hessian = derivatives.hess(mb_pot.muller_brown_potential, point[0], point[1])
    else:
        d_hess = bfgs.BFGS_hessian_update(old_point, point, old_grad, grad, old_hess)
        hessian = hessian + d_hess

    point_list.append(point)
    energy_list.append(energy)
    
    rho = (old_energy - energy) / (energy - energy - np.dot(grad.T, step) - 0.5 * np.dot(step.T, np.dot(hessian, step)))    
    
    if rho < 0.25:
        trust_radius = 0.25 * trust_radius
    elif rho > 0.75:
        trust_radius = min(2.0 * trust_radius, max_trust_radius)
    else:
        trust_radius = trust_radius
    
    if np.linalg.norm(grad) < threshold:
        break
    
    
point_list = np.array(point_list)
energy_list = np.array(energy_list)

level = []
for i in range(-10, 40):
    level.append(15*i)


plt.plot(point_list.T[0][0], point_list.T[0][1], 'w.--')
x_list = np.linspace(-3.5, 1.5, 400)
y_list = np.linspace(-1.0, 2.5, 400)
plt.title('Iteration: {}'.format(j))
plt.xlabel('x')
plt.ylabel('y')
x_mesh, y_mesh = np.meshgrid(x_list, y_list)
f_mesh = mb_pot.muller_brown_potential(x_mesh, y_mesh)
cont = plt.contourf(x_mesh, y_mesh, f_mesh, levels=level, cmap='jet')
plt.colorbar()
plt.savefig('result.png'.format(j))
plt.close()

print("Optimization is finished.")
