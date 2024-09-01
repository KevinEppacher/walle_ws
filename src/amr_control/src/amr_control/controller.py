#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray

import casadi as ca
import numpy as np
import math
from amr_control.visualizer import Visualizer

class nMPC:
    def __init__(self, model, max_obstacles, N=70, Q=np.diag([10, 10, 0.001]), R=np.diag([0.5, 0.05]), T=0.1):
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
        self.x0 = np.array([0, 0, 0])
        self.xx = np.zeros((3, 1))
        self.xx[:, 0] = self.x0
        self.xx1 = []
        self.X0 = np.tile(self.x0, (self.N + 1, 1))
        
        self.solver = self.define_optimization_problem()

    def define_optimization_problem(self):
        n_states = self.model.n_states
        n_controls = self.model.n_controls
        Q = self.Q
        R = self.R
        N = self.N
        T = self.T
        
        U = ca.SX.sym('U', n_controls, N)
        P = ca.SX.sym('P', n_states + N * (n_states + n_controls) + self.n_obstacles * 3)
        X = ca.SX.sym('X', n_states, N + 1)
        rob_diam = self.model.diam

        # Definiere die Kostenfunktion
        obj = 0
        g = []
        
        st = X[:, 0]
        g = ca.vertcat(g, st - P[:3])

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
            f_value = self.model.f(st, con)
            st_next_euler = st + T * f_value
            g = ca.vertcat(g, st_next - st_next_euler)
                            
        # Obstacle avoidance constraints
        for k in range(N + 1):
            for i in range(self.n_obstacles):
                obs_x = P[-(3 * (i + 1))]
                obs_y = P[-(3 * (i + 1) - 1)]
                obs_diam = P[-(3 * (i + 1) - 2)]
                obs_constraint = -ca.sqrt((X[0, k] - obs_x) ** 2 + (X[1, k] - obs_y) ** 2) + (rob_diam / 2 + obs_diam / 2)
                g = ca.vertcat(g, obs_constraint)

        # Make the decision variable one column vector
        self.OPT_variables = ca.vertcat(ca.reshape(X, n_states * ( N + 1), 1), ca.reshape(U, n_controls * N, 1))
        
        nlp_prob = {'f': obj, 'x': self.OPT_variables, 'g': g, 'p': P}
        
        opts = {'ipopt.max_iter': 2000,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-10,
                'ipopt.acceptable_obj_change_tol': 1e-8}

        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def solve_mpc(self, current_state, ref_traj, target_state, obstacles):
        self.obstacles = obstacles
        self.target_state = target_state
        n_obstacles = self.n_obstacles
        N = self.N

        args = {
            'lbg': np.concatenate((np.zeros((3 * (self.N + 1), 1)), np.full((self.n_obstacles * (self.N + 1), 1), -np.inf))),
            'ubg': np.concatenate((np.zeros((3 * (self.N + 1), 1)), np.full((self.n_obstacles * (self.N + 1), 1), 0))),
            'lbx': np.full((3 * (self.N + 1) + 2 * self.N, 1), -ca.inf),
            'ubx': np.full((3 * (self.N + 1) + 2 * self.N, 1), ca.inf),
            'p': np.zeros((self.model.n_states + self.N * (self.model.n_states + self.model.n_controls) + 3 * self.n_obstacles,))
        }
        
        # State bounds
        args['lbx'][0:3 * (N + 1):3] = -200
        args['ubx'][0:3 * (N + 1):3] = 200
        args['lbx'][1:3 * (N + 1):3] = -200
        args['ubx'][1:3 * (N + 1):3] = 200
        args['lbx'][2:3 * (N + 1):3] = -ca.inf
        args['ubx'][2:3 * (N + 1):3] = ca.inf

        args['lbx'][3 * (self.N + 1)::2] = self.v_min
        args['ubx'][3 * (self.N + 1)::2] = self.v_max
        args['lbx'][3 * (self.N + 1) + 1::2] = self.omega_min
        args['ubx'][3 * (self.N + 1) + 1::2] = self.omega_max
        args['p'][0:3] = current_state
        
        size_ref_traj = len(ref_traj)
        
        ref_traj_array = []

        for k in range(self.N):
            if k < size_ref_traj:
                ref_traj_array.append(ref_traj[k].pose)
                # Use reference trajectory from ref_traj if available
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
                # Use reference trajectory from ref_traj if available
                x_ref = self.target_state[0]
                y_ref = self.target_state[1]
                theta_ref = self.target_state[2]
                theta_ref = self.normalize_angle(theta_ref)
                u_ref = 0
                omega_ref = 0

            args['p'][self.model.n_states + k * (self.model.n_states + self.model.n_controls):self.model.n_states + (k + 1) * (self.model.n_states + self.model.n_controls) - 2] = [x_ref, y_ref, theta_ref]
            args['p'][self.model.n_states + (k + 1) * (self.model.n_states + self.model.n_controls) - 2:self.model.n_states + (k + 1) * (self.model.n_states + self.model.n_controls)] = [u_ref, omega_ref]

        reserved_obstacles = n_obstacles - len(self.obstacles)

        offset = self.model.n_states + self.N * (self.model.n_states + self.model.n_controls)
        
        for i, obs in enumerate(self.obstacles):
            args['p'][offset + 3 * i: offset + 3 * (i + 1)] = obs
            
        offset_reserved_obstacles = offset + 3 * len(self.obstacles)

        for i in range(reserved_obstacles):
            args['p'][offset_reserved_obstacles + 3 * i: offset_reserved_obstacles + 3 * (i + 1)] = [10000, 10000, -1000]
            
            self.viz.publish_obstacle_marker(obs)

        args['x0'] = np.concatenate((self.X0.T.flatten(), self.u0.T.flatten()))
        
        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = np.reshape(sol['x'][3 * (self.N + 1):].full(), (self.N, 2))
        
        self.u0 = np.vstack((u[1:, :], u[-1, :]))
        
        self.xx1.append(np.reshape(sol['x'][:3 * (self.N + 1)].full(), (self.N + 1, 3)))
        
        self.X0 = np.vstack((self.xx1[-1][1:], self.xx1[-1][-1, :]))
        
        f_value = self.model.f(self.x0, u[0, :]).full().flatten()
        
        self.x0 = self.x0 + self.T * f_value
        
        self.xx = np.hstack((self.xx, self.x0[:, None]))
        
        self.X0 = np.vstack((self.xx1[-1][1:], self.xx1[-1][-1, :]))

        self.viz.publish_predicted_trajectory(self.xx1[-1])
        
        return u[0]

    def get_yaw_from_quaternion(self, quaternion):
        quaternion = [x / math.sqrt(sum([x * x for x in quaternion])) for x in quaternion]
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
