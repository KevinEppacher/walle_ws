#!/usr/bin/env python3

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define optimization variables
    x = ca.MX.sym('x', 2)

    # Define the objective function
    f = (x[0] - 1)**2 + (x[1] - 2)**2

    # Define constraints
    g = ca.vertcat(x[0] + x[1] - 3, x[0] - x[1] + 1)

    # Create an NLP solver
    nlp = {'x': x, 'f': f, 'g': g}
    opts = {'ipopt.print_level': 0, 'print_time': 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve the problem
    sol = solver(x0=[0, 0], lbg=[0, 0], ubg=[0, 0])

    # Extract and print the solution
    x_opt = sol['x']
    x_opt_numpy = x_opt.full().flatten()  # Convert to a NumPy array and flatten it
    print(f"Optimal solution: x1 = {x_opt_numpy[0]:.4f}, x2 = {x_opt_numpy[1]:.4f}")

    # Visualize the objective function
    x1 = np.linspace(-1, 3, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = (X1 - 1)**2 + (X2 - 2)**2

    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z, levels=50)
    plt.plot(x_opt_numpy[0], x_opt_numpy[1], 'ro', label='Optimal Solution')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Objective Function Contour Plot')
    plt.legend()
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()
