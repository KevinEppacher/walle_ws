#!/usr/bin/env python3

# ROS1 imports
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray

# Casadi imports
import casadi as ca
from casadi.tools import *

# Other imports
import numpy as np
import math
from amr_control.visualizer import Visualizer

class nMPC:
    def __init__(self, model, max_obstacles, N=100, Q=np.diag([0.1, 0.1, 0.001]), R=np.diag([0.05, 0.05]), T=0.1):
        self.model = model
        self.n_obstacles = max_obstacles
        self.N = N
        self.Q = Q
        self.R = R
        self.T = T
        self.v_min, self.v_max = -0.2, 0.2
        self.omega_min, self.omega_max = -0.2, 0.2
        self.viz = Visualizer()

        self.u0 = np.zeros((N, 2))
        self.xx1 = []
        self.X0 = self.initialize_state()

        self.solver = self.define_optimization_problem()

    def initialize_state(self):
        x0 = np.array([0, 0, 0])
        return np.tile(x0, (self.N + 1, 1))

    def define_optimization_problem(self):
        n_states = self.model.n_states
        n_controls = self.model.n_controls
        n_obstacles = self.n_obstacles
        N = self.N
        
        U = ca.SX.sym('U', n_controls, N)
        
        P = ca.SX.sym('P', n_states + N * (n_states + n_controls) + n_obstacles * 3)
                
        X = ca.SX.sym('X', n_states, N + 1)

        obj, g = self.define_objective_and_constraints(X, U, P)

        OPT_variables = ca.vertcat(ca.reshape(X, n_states * (N + 1), 1), ca.reshape(U, n_controls * N, 1))
        
        nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

        opts = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': False,
            # 'ipopt.sb': 'yes',
            'ipopt.acceptable_tol': 1e-10,
            'ipopt.acceptable_obj_change_tol': 1e-8
        }

        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def define_objective_and_constraints(self, X, U, P):
        n_states = self.model.n_states
        Q = self.Q
        R = self.R
        N = self.N
        n_obstacles = self.n_obstacles

        obj = 0
        g = []

        g.append(X[:, 0] - P[:3])

        for k in range(N):
            state_ref, control_ref = self.get_references(P, k, n_states)
            obj += self.calculate_objective(X[:, k], U[:, k], state_ref, control_ref, Q, R)
            g.append(self.apply_dynamics_constraint(X[:, k], X[:, k + 1], U[:, k]))

        for k in range(N + 1):
            for i in range(n_obstacles):
                g.append(self.obstacle_avoidance_constraint(X[:, k], P, i))

        g = ca.vertcat(*g)
        return obj, g

    def get_references(self, P, k, n_states):
        n_controls = self.model.n_controls
        state_ref = P[n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2]
        control_ref = P[n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)]
        return state_ref, control_ref

    def calculate_objective(self, state, control, state_ref, control_ref, Q, R):
        state_error = state - state_ref
        control_error = control - control_ref
        return ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([control_error.T, R, control_error])

    def apply_dynamics_constraint(self, state, next_state, control):
        T = self.T
        f_value = self.model.f(state, control)
        return next_state - (state + T * f_value)

    def obstacle_avoidance_constraint(self, state, P, obstacle_index):
        obs_x = P[-(3 * (obstacle_index + 1))]
        obs_y = P[-(3 * (obstacle_index + 1) - 1)]
        obs_diam = P[-(3 * (obstacle_index + 1) - 2)]
        rob_diam = self.model.diam
        return -ca.sqrt((state[0] - obs_x) ** 2 + (state[1] - obs_y) ** 2) + (rob_diam / 2 + obs_diam / 2)

    def solve_mpc(self, current_state, ref_traj, target_state, obstacles):
        self.obstacles = obstacles
        
        args = self.initialize_solver_arguments(current_state, ref_traj, target_state)

        # Solve the optimization problem
        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        # Extract and update the control inputs and states
        u = self.update_state_and_control(sol)

        # Publish the predicted trajectory
        self.publish_trajectory(sol)

        return u[0]

    def initialize_solver_arguments(self, current_state, ref_traj, target_state):
        N = self.N
        n_states = self.model.n_states
        n_controls = self.model.n_controls
        n_obstacles = self.n_obstacles

        total_constraints = 3 * (N + 1) + n_obstacles * (N + 1)

        args = {
            'lbg': np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((n_obstacles * (N + 1), 1), -np.inf))),
            'ubg': np.concatenate((np.zeros((3 * (N + 1), 1)), np.full((n_obstacles * (N + 1), 1), 0))),
            'lbx': np.full((3 * (N + 1) + 2 * N, 1), -ca.inf),
            'ubx': np.full((3 * (N + 1) + 2 * N, 1), ca.inf),
            'p': np.zeros((n_states + N * (n_states + n_controls) + 3 * n_obstacles,))
        }

        self.set_bounds(args)

        args['p'][0:3] = current_state
        self.set_reference_trajectory(args, ref_traj, target_state)

        args['x0'] = np.concatenate((self.X0.T.flatten(), self.u0.T.flatten()))

        return args

    def set_bounds(self, args):
        N = self.N
        args['lbx'][0:3 * (N + 1):3] = -ca.inf
        args['ubx'][0:3 * (N + 1):3] = ca.inf
        args['lbx'][1:3 * (N + 1):3] = -ca.inf
        args['ubx'][1:3 * (N + 1):3] = ca.inf
        args['lbx'][2:3 * (N + 1):3] = -ca.inf
        args['ubx'][2:3 * (N + 1):3] = ca.inf

        args['lbx'][3 * (N + 1)::2] = self.v_min
        args['ubx'][3 * (N + 1)::2] = self.v_max
        args['lbx'][3 * (N + 1) + 1::2] = self.omega_min
        args['ubx'][3 * (N + 1) + 1::2] = self.omega_max

    def set_reference_trajectory(self, args, ref_traj, target_state):
        N = self.N
        n_states = self.model.n_states
        n_controls = self.model.n_controls
        n_obstacles = self.n_obstacles

        size_ref_traj = len(ref_traj)
        ref_traj_array = []

        for k in range(N):
            if k < size_ref_traj:
                ref_traj_array.append(ref_traj[k].pose)
                x_ref = ref_traj[k].pose.position.x
                y_ref = ref_traj[k].pose.position.y
                theta_ref = self.get_yaw_from_quaternion([
                    ref_traj[k].pose.orientation.x,
                    ref_traj[k].pose.orientation.y,
                    ref_traj[k].pose.orientation.z,
                    ref_traj[k].pose.orientation.w
                ])
                theta_ref = self.normalize_angle(theta_ref)
                u_ref = 10
                omega_ref = 0
                
            if size_ref_traj < N:
                x_ref = target_state[0]
                y_ref = target_state[1]
                theta_ref = self.normalize_angle(target_state[2])
                u_ref = 0
                omega_ref = 0

            args['p'][n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2] = [x_ref, y_ref, theta_ref]
            args['p'][n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)] = [u_ref, omega_ref]

        reserved_obstacles = n_obstacles - len(self.obstacles)
        # Dynamically set the obstacles
        offset = n_states + N * (n_states + n_controls)
        for i, obs in enumerate(self.obstacles):
            args['p'][offset + 3 * i: offset + 3 * (i + 1)] = obs
            self.viz.publish_obstacle_marker(obs)
            
        offset_reserved_obstacles = offset + 3 * len(self.obstacles)

        for i in range(reserved_obstacles):
            args['p'][offset_reserved_obstacles + 3 * i: offset_reserved_obstacles + 3 * (i + 1)] = [10000, 10000, -1000]

        self.viz.publish_refrence_trajectory(ref_traj_array)

    def extract_control_input(self, sol):
        N = self.N
        return np.reshape(sol['x'][3 * (N + 1):].full(), (N, 2))

    def update_state_and_control(self, sol):
        N = self.N
        u = np.reshape(sol['x'][3 * (N + 1):].full(), (N, 2))
        
        # Update control inputs
        self.u0 = np.vstack((u[1:, :], u[-1, :]))

        # Update state trajectory
        self.xx1.append(np.reshape(sol['x'][:3 * (N + 1)].full(), (N + 1, 3)))
        
        # Set the initial state for the next iteration
        self.X0 = np.vstack((self.xx1[-1][1:], self.xx1[-1][-1, :]))

        return u

    def publish_trajectory(self, sol):
        N = self.N
        predicted_states = np.reshape(sol['x'][:3 * (N + 1)].full(), (N + 1, 3))
        self.viz.publish_predicted_trajectory(predicted_states)

    def get_yaw_from_quaternion(self, quaternion):
        norm = math.sqrt(sum([x * x for x in quaternion]))
        quaternion = [x / norm for x in quaternion]
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
