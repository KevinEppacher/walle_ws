import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Setup
T = 0.2  # sampling time [s]
N = 25  # prediction horizon
rob_diam = 0.3

v_max = 0.6
v_min = -v_max
omega_max = np.pi / 4
omega_min = -omega_max

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

U = ca.SX.sym('U', n_controls, N)
P = ca.SX.sym('P', n_states + n_states)

X = ca.SX.sym('X', n_states, N + 1)
X[:, 0] = P[0:3]

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    f_value = f(st, con)
    st_next = st + (T * f_value)
    X[:, k + 1] = st_next

ff = ca.Function('ff', [U, P], [X])

# Objective function
obj = 0
g = []

Q = np.diag([1, 5, 0.1])
R = np.diag([0.5, 0.05])

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    obj += ca.mtimes([(st - P[3:6]).T, Q, (st - P[3:6])]) + ca.mtimes([con.T, R, con])

for k in range(N + 1):
    g = ca.vertcat(g, X[0, k])
    g = ca.vertcat(g, X[1, k])

OPT_variables = ca.reshape(U, 2 * N, 1)
nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

opts = {'ipopt.max_iter': 2000,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

args = {'lbg': -2, 'ubg': 2}
args['lbx'] = ca.DM.zeros((2 * N, 1))
args['ubx'] = ca.DM.zeros((2 * N, 1))
args['lbx'][0:2*N-1:2] = v_min
args['ubx'][0:2*N-1:2] = v_max
args['lbx'][1:2*N:2] = omega_min
args['ubx'][1:2*N:2] = omega_max

# Simulation loop
t0 = 0
x0 = np.array([0, 0, 0])
xs = np.array([1.5, 1.5, np.pi])
xx = np.zeros((3, 1))
xx[:, 0] = x0
global t
t = np.array([t0])
u0 = np.zeros((N, 2))
sim_tim = 20
mpciter = 0
xx1 = []
u_cl = []

def shift(T, t0, x0, u, f):
    st = x0
    con = u[0, :]
    f_value = f(st, con)
    st = st + (T * f_value.full().flatten())
    t0 = t0 + T
    u0 = np.vstack((u[1:, :], u[-1, :]))
    return t0, st, u0

# Plot setup
fig, ax = plt.subplots()
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)
line, = ax.plot([], [], 'b-', lw=2)
point, = ax.plot([], [], 'ro')
trajectory, = ax.plot([], [], 'g--', lw=1)

def init():
    line.set_data([], [])
    point.set_data([], [])
    trajectory.set_data([], [])
    return line, point, trajectory

def animate(i):
    global t0, x0, u0, mpciter, xx

    if np.linalg.norm(x0 - xs, 2) > 1e-2 and mpciter < sim_tim / T:
        args['p'] = np.concatenate((x0, xs))
        args['x0'] = u0.reshape(2 * N, 1)

        sol = solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                     lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = sol['x'].full().reshape(N, 2)
        ff_value = ff(u.T, args['p'])
        xx1.append(ff_value.full())
        u_cl.append(u[0, :])

        global t
        t = np.vstack((t, t0))
        [t0, x0, u0] = shift(T, t0, x0, u, f)
        xx = np.hstack((xx, x0.reshape(3, 1)))
        mpciter += 1

    line.set_data(xx[0, :], xx[1, :])
    point.set_data(xs[0], xs[1])
    if xx1:
        trajectory.set_data(xx1[-1][0, :], xx1[-1][1, :])
    return line, point, trajectory

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=300, interval=100, blit=True)

plt.show()
