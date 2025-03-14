import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 0.5  # [s] sampling time
N = 20  # prediction horizon
rob_diam = 0.3

v_max = 0.6
v_min = -v_max
omega_max = np.pi / 4
omega_min = -omega_max

# Obstacle parameters
obs_x = 6  # meters
obs_y = 0.8  # meters
obs_diam = 1.5  # meters

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

# Loop over the prediction horizon
for k in range(N):
    # Extract current state and control
    st = X[:, k]
    con = U[:, k]
    
    # Calculate the objective function
    state_ref = P[n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2]
    control_ref = P[n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)]
    
    state_error = st - state_ref
    control_error = con - control_ref
    
    obj += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([control_error.T, R, control_error])
    
    # System dynamics constraints
    st_next = X[:, k + 1]
    f_value = f(st, con)
    st_next_euler = st + T * f_value
    g = ca.vertcat(g, st_next - st_next_euler)

# Obstacle avoidance constraints
for k in range(N + 1):
    obs_constraint = -ca.sqrt((X[0, k] - obs_x) ** 2 + (X[1, k] - obs_y) ** 2) + (rob_diam / 2 + obs_diam / 2)
    g = ca.vertcat(g, obs_constraint)

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
num_constraints = 3 * (N + 1) + (N + 1)  # Equality + Obstacle constraints

args = {
    'lbg': np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((N + 1, 1), -np.inf))),  # Include inequality constraints
    'ubg': np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((N + 1, 1), 0))),  # Upper bound for inequality
    'lbx': np.full((3 * (N + 1) + 2 * N, 1), -ca.inf),
    'ubx': np.full((3 * (N + 1) + 2 * N, 1), ca.inf)
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
obstacle_circle = plt.Circle((obs_x, obs_y), obs_diam / 2, color='r', fill=False, linestyle='--', label='Obstacle')
ax.add_patch(obstacle_circle)
ax.set_xlim([-2, 15])
ax.set_ylim([-2, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Robot Trajectory with Obstacle Avoidance and Reference Tracking')
ax.legend()
plt.grid(True)

# Main simulation loop
while mpciter < sim_tim / T:
    current_time = mpciter * T
    args['p'] = np.zeros((n_states + N * (n_states + n_controls),))  # Initialize 'p' array
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
    
    predicted_states = []
    predicted_states.append(np.reshape(sol['x'][:3 * (N + 1)].full(), (N + 1, 3)))

    predicted_states = np.array(predicted_states).squeeze()
        
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

# Open a new window and plot control inputs
plt.figure()
plt.plot(t[:-1], [uc[0] for uc in u_cl], label='v')
plt.plot(t[:-1], [uc[1] for uc in u_cl], label='omega')
plt.xlabel('Time [s]')
plt.ylabel('Control Inputs')
plt.title('Control Inputs over Time')
plt.legend()
plt.grid()
plt.show()
