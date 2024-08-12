import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 0.5  # [s] sampling time
N = 8  # prediction horizon
rob_diam = 0.3

v_max = 0.6
v_min = -v_max
omega_max = np.pi / 4
omega_min = -omega_max

# Define the state variables and control inputs
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.size1()

v = ca.SX.sym('v')
omega = ca.SX.sym('omega')
controls = ca.vertcat(v, omega)
n_controls = controls.size1()

# Define the system dynamics
rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
f = ca.Function('f', [states, controls], [rhs])

# Define decision variables
U = ca.SX.sym('U', n_controls, N)
P = ca.SX.sym('P', n_states + N * (n_states + n_controls))

X = ca.SX.sym('X', n_states, N + 1)

# Define the objective function and constraints
obj = 0
g = []

Q = np.diag([1, 1, 0.5])
R = np.diag([0.5, 0.05])

# Initial state constraint
st = X[:, 0]
g = ca.vertcat(g, st - P[0:n_states])

# Cost function and dynamic constraints
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj += ca.mtimes([(st - P[n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2]).T, Q,
                      (st - P[n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2])]) \
        + ca.mtimes([(con - P[n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)]).T, R,
                     (con - P[n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)])])
    
    st_next = X[:, k + 1]
    f_value = f(st, con)
    st_next_euler = st + T * f_value
    g = ca.vertcat(g, st_next - st_next_euler)

# Flatten decision variables
OPT_variables = ca.vertcat(ca.reshape(X, n_states * (N + 1), 1), ca.reshape(U, n_controls * N, 1))

# Define NLP problem
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

# Solver options
opts = {
    'ipopt.max_iter': 1500,
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.acceptable_tol': 1e-8,
    'ipopt.acceptable_obj_change_tol': 1e-6
}

# Create solver
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Constraints bounds
args = {
    'lbg': np.zeros((3 * (N + 1), 1)),  # Equality constraints
    'ubg': np.zeros((3 * (N + 1), 1)),  # Equality constraints
    'lbx': np.full((3 * (N + 1) + 2 * N, 1), -ca.inf),
    'ubx': np.full((3 * (N + 1) + 2 * N, 1), ca.inf),
    'p': np.zeros((n_states + N * (n_states + n_controls),))  # Initialize 'p'
}

# State bounds
args['lbx'][0:3 * (N + 1):3] = -15  # x lower bound
args['ubx'][0:3 * (N + 1):3] = 15  # x upper bound
args['lbx'][1:3 * (N + 1):3] = -2  # y lower bound
args['ubx'][1:3 * (N + 1):3] = 2  # y upper bound
args['lbx'][2:3 * (N + 1):3] = -ca.inf  # theta lower bound
args['ubx'][2:3 * (N + 1):3] = ca.inf  # theta upper bound

# Control bounds
args['lbx'][3 * (N + 1)::2] = v_min  # v lower bound
args['ubx'][3 * (N + 1)::2] = v_max  # v upper bound
args['lbx'][3 * (N + 1) + 1::2] = omega_min  # omega lower bound
args['ubx'][3 * (N + 1) + 1::2] = omega_max  # omega upper bound

# Simulation loop
t0 = 0
x0 = np.array([0, 0, 0.0])  # Initial condition
xs = np.array([1.5, 1.5, 0.0])  # Reference posture

xx = np.zeros((3, 1))
xx[:, 0] = x0
t = [t0]

u0 = np.zeros((N, 2))
X0 = np.tile(x0, (N + 1, 1))

sim_tim = 30  # Maximum simulation time

mpciter = 0
xx1 = []
u_cl = []

def shift(T, t0, x0, u, f):
    st = x0
    con = u[0, :]
    f_value = f(st, con)
    st_next = st + T * f_value.full().flatten()
    t0 = t0 + T
    u0 = np.vstack((u[1:, :], u[-1, :]))
    return t0, st_next, u0

# Initialize interactive plot
plt.ion()
fig, ax = plt.subplots()
robot_line, = ax.plot(xx[0, :], xx[1, :], 'b-', label='Trajectory')
goal_point, = ax.plot(xs[0], xs[1], 'ro', label='Goal')
ax.set_xlim([-2, 15])
ax.set_ylim([-2, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Robot Trajectory')
ax.legend()
plt.grid(True)

# Main simulation loop
while mpciter < sim_tim / T:
    current_time = mpciter * T
    args['p'][0:3] = x0  # Initial condition

    # Set the reference trajectory
    for k in range(N):
        t_predict = current_time + k * T
        x_ref = 0.5 * t_predict
        y_ref = 1
        theta_ref = 0
        u_ref = 0.5
        omega_ref = 0
        if x_ref >= 12:
            x_ref = 12
            y_ref = 1
            theta_ref = 0
            u_ref = 0
            omega_ref = 0
        args['p'][n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2] = [x_ref, y_ref, theta_ref]
        args['p'][n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)] = [u_ref, omega_ref]

    # Solve the NLP
    args['x0'] = np.concatenate((X0.T.flatten(), u0.T.flatten()))
    sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])
    u = np.reshape(sol['x'][3 * (N + 1):].full(), (N, 2))
    xx1.append(np.reshape(sol['x'][:3 * (N + 1)].full(), (N + 1, 3)))

    u_cl.append(u[0, :])
    t.append(t0)

    # Apply the control and shift the solution
    t0, x0, u0 = shift(T, t0, x0, u, f)
    xx = np.hstack((xx, x0[:, None]))
    X0 = np.vstack((xx1[-1][1:], xx1[-1][-1, :]))
    mpciter += 1

    # Update plot
    robot_line.set_xdata(xx[0, :])
    robot_line.set_ydata(xx[1, :])
    plt.draw()
    plt.pause(0.01)

plt.ioff()  # Disable interactive mode
plt.show()

# Plot control inputs
plt.figure()
plt.plot(t, [uc[0] for uc in u_cl], label='v')
plt.plot(t, [uc[1] for uc in u_cl], label='omega')
plt.xlabel('time')
plt.ylabel('control inputs')
plt.title('Control inputs')
plt.legend()
plt.grid()
plt.show()
