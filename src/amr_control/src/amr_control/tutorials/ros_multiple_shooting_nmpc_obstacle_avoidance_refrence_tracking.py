#!/usr/bin/env python3

# ROS1 imports
import rospy
from geometry_msgs.msg import Twist, Pose, PoseArray
from nav_msgs.msg import Odometry
import tf.transformations

# Casadi imports
import casadi as ca
import numpy as np

# Callback function to update the robot's current state from Odometry
def odom_callback(msg):
    global x0
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation

    # Convert quaternion to yaw angle
    siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
    cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    x0 = np.array([position.x, position.y, yaw])

# Shift function to update the state and control
def shift(T, t0, x0, u, f):
    st = x0
    con = u[0, :]
    f_value = f(st, con)
    st_next = st + T * f_value.full().flatten()
    t0 = t0 + T
    u0 = np.vstack((u[1:, :], u[-1, :]))
    return t0, st_next, u0

# Initialize ROS node
rospy.init_node('mpc_controller')

# Subscribe to the /odom topic
rospy.Subscriber('/odom', Odometry, odom_callback)

# Publisher for the /cmd_vel topic
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

# Publisher for predicted trajectory
predicted_pose_array_pub = rospy.Publisher('/predicted_trajectory', PoseArray, queue_size=10)

# Parameters
T = 0.5  # [s] sampling time
N = 8  # prediction horizon
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

# Initialize state and control
t0 = 0
x0 = np.array([0, 0, 0.0])  # Initial condition
u0 = np.zeros((N, 2))
X0 = np.tile(x0, (N + 1, 1))

sim_tim = 30  # Maximum simulation time
mpciter = 0

# Main simulation loop
rate = rospy.Rate(1 / T)  # Control loop rate
while not rospy.is_shutdown() and mpciter < sim_tim / T:
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
    
    # Extract predicted states for visualization
    predicted_states = np.reshape(sol['x'][:3 * (N + 1)].full(), (N + 1, 3))

    # Publish predicted trajectory as PoseArray
    pose_array_msg = PoseArray()
    pose_array_msg.header.stamp = rospy.Time.now()
    pose_array_msg.header.frame_id = 'odom'

    for i in range(N + 1):
        pose = Pose()
        pose.position.x = predicted_states[i, 0]
        pose.position.y = predicted_states[i, 1]
        quaternion = tf.transformations.quaternion_from_euler(0, 0, predicted_states[i, 2])
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        pose_array_msg.poses.append(pose)

    predicted_pose_array_pub.publish(pose_array_msg)

    # Apply the control and shift the solution
    t0, x0, u0 = shift(T, t0, x0, u, f)
    mpciter += 1

    # Publish control commands
    cmd_msg = Twist()
    cmd_msg.linear.x = u[0, 0]
    cmd_msg.angular.z = u[0, 1]
    cmd_vel_pub.publish(cmd_msg)

    rate.sleep()
