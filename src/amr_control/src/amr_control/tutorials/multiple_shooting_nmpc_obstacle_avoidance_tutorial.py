import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 0.2  # [s]
N = 10  # prediction horizon
rob_diam = 0.3

v_max = 0.6
v_min = -v_max
omega_max = np.pi / 4
omega_min = -omega_max

# Obstacle parameters
obs_x = 0.8  # meters
obs_y = 0.8  # meters
obs_diam = 0.6 # meters

# States and controls
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.size1()

v = ca.SX.sym('v')
omega = ca.SX.sym('omega')
controls = ca.vertcat(v, omega)
n_controls = controls.size1()
rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)

f = ca.Function('f', [states, controls], [rhs])

# Decision variables
U = ca.SX.sym('U', n_controls, N)
P = ca.SX.sym('P', n_states + n_states)

X = ca.SX.sym('X', n_states, (N + 1))

# Objective function and constraints
obj = 0
g = []

Q = np.diag([1, 5, 0.1])
R = np.diag([0.5, 0.05])

# Initial condition constraint
st = X[:, 0]
g = ca.vertcat(g, st - P[:3])  

# Dynamic and obstacle constraints
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj = obj + (st - P[3:]).T @ Q @ (st - P[3:]) + con.T @ R @ con
    st_next = X[:, k + 1]
    f_value = f(st, con)
    st_next_euler = st + T * f_value
    g = ca.vertcat(g, st_next - st_next_euler)  # System dynamics constraint

# Obstacle avoidance constraints
for k in range(N + 1):
    obs_constraint = -ca.sqrt((X[0, k] - obs_x) ** 2 + (X[1, k] - obs_y) ** 2) + (rob_diam / 2 + obs_diam / 2)
    g = ca.vertcat(g, obs_constraint)

# Make the decision variable one column vector
OPT_variables = ca.vertcat(ca.reshape(X, n_states * (N + 1), 1), ca.reshape(U, n_controls * N, 1))

nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

opts = {'ipopt.max_iter': 2000,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Calculate total number of constraints
num_constraints = 3 * (N + 1) + (N + 1)  # Equality + Obstacle constraints

args = {
    'lbg': np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((N + 1, 1), -np.inf))),  # Include inequality constraints
    'ubg': np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((N + 1, 1), 0))),  # Upper bound for inequality
    'lbx': np.full((3 * (N + 1) + 2 * N, 1), -ca.inf),
    'ubx': np.full((3 * (N + 1) + 2 * N, 1), ca.inf)
}

# State bounds
args['lbx'][0:3 * (N + 1):3] = -2
args['ubx'][0:3 * (N + 1):3] = 2
args['lbx'][1:3 * (N + 1):3] = -2
args['ubx'][1:3 * (N + 1):3] = 2
args['lbx'][2:3 * (N + 1):3] = -ca.inf
args['ubx'][2:3 * (N + 1):3] = ca.inf

# Control bounds
args['lbx'][3 * (N + 1)::2] = v_min
args['ubx'][3 * (N + 1)::2] = v_max
args['lbx'][3 * (N + 1) + 1::2] = omega_min
args['ubx'][3 * (N + 1) + 1::2] = omega_max

# Simulation loop
t0 = 0
x0 = np.array([0, 0, 0])
xs = np.array([1.5, 1.5, np.pi])

xx = np.zeros((3, 1))
xx[:, 0] = x0
t = [t0]

u0 = np.zeros((N, 2))
X0 = np.tile(x0, (N + 1, 1))

sim_tim = 20  # Maximum simulation time

mpciter = 0
xx1 = []
u_cl = []

# Initialize plot
plt.ion()
fig, ax = plt.subplots()
robot_line, = ax.plot(xx[0, :], xx[1, :], 'b-', label='Trajectory')
goal_point, = ax.plot(xs[0], xs[1], 'ro', label='Goal')
obstacle_circle = plt.Circle((obs_x, obs_y), obs_diam / 2, color='r', fill=False, linestyle='--', label='Obstacle')
ax.add_patch(obstacle_circle)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Robot Trajectory with Obstacle Avoidance')
ax.legend()
plt.grid(True)

# Predicted trajectory plot
predicted_line, = ax.plot([], [], 'g--', label='Predicted Trajectory')

# Main simulation loop
while np.linalg.norm(x0 - xs, 2) > 1e-2 and mpciter < sim_tim / T:
    args['p'] = np.concatenate((x0, xs))
    args['x0'] = np.concatenate((X0.T.flatten(), u0.T.flatten()))

    sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

    u = np.reshape(sol['x'][3 * (N + 1):].full(), (N, 2))
    xx1.append(np.reshape(sol['x'][:3 * (N + 1)].full(), (N + 1, 3)))

    u_cl.append(u[0, :])
    t.append(t0)

    # Apply the control and shift the solution
    f_value = f(x0, u[0, :]).full().flatten()
    x0 = x0 + T * f_value
    xx = np.hstack((xx, x0[:, None]))
    X0 = np.vstack((xx1[-1][1:], xx1[-1][-1, :]))
    mpciter += 1

    # Update plot
    robot_line.set_xdata(xx[0, :])
    robot_line.set_ydata(xx[1, :])

    # Update predicted trajectory plot
    predicted_line.set_xdata(xx1[-1][:, 0])
    predicted_line.set_ydata(xx1[-1][:, 1])

    plt.draw()
    plt.pause(0.01)

plt.ioff()  # Disable interactive mode
ss_error = np.linalg.norm(x0 - xs, 2)
print(f'Steady-state error: {ss_error}')

# Final plot adjustments
plt.figure()
plt.plot(xx[0, :], xx[1, :], 'b', label='Trajectory')
plt.plot(xs[0], xs[1], 'ro', label='Goal')
plt.gca().add_patch(obstacle_circle)  # Only add obstacle once
plt.xlabel('x')
plt.ylabel('y')
plt.title('Final Trajectory')
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(t, [u[0] for u in u_cl], label='v')
plt.plot(t, [u[1] for u in u_cl], label='omega')
plt.xlabel('time')
plt.ylabel('control inputs')
plt.title('Control Inputs Over Time')
plt.legend()
plt.grid()
plt.show()
