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
    def __init__(self, model, max_obstacles, N=20, Q=np.diag([1, 1, 0.001]), R=np.diag([0.5, 0.05]), T=0.3, penalty_weight=1):
        self.model = model
        self.n_obstacles = max_obstacles
        self.N = N
        self.Q = Q
        self.R = R
        self.T = T
        self.v_min, self.v_max = -0.2, 0.2
        self.omega_min, self.omega_max = -0.2, 0.2
        self.viz = Visualizer()
        
        # Add penalty weight
        self.penalty_weight = penalty_weight

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
        
        U = ca.SX.sym('U', n_controls, N)
        # Erweitere den Parametervektor um 1 zusätzlichen Eintrag für T
        P = ca.SX.sym('P', n_states + N * (n_states + n_controls) + self.n_obstacles * 3 + 1)
        X = ca.SX.sym('X', n_states, N + 1)
        rob_diam = self.model.diam

        # Definiere die Kostenfunktion
        obj = 0
        g = []
        
        st = X[:, 0]
        g = ca.vertcat(g, st - P[:3])

        # Extrahiere T aus dem Parametervektor P (letzter Eintrag in P)
        T = P[-1]

        # Loop über den Prediction-Horizont
        for k in range(N):
            st = X[:, k]
            con = U[:, k]
            
            # Berechne die Kostenfunktion
            state_ref = P[n_states + k * (n_states + n_controls): n_states + (k + 1) * (n_states + n_controls) - 2]
            control_ref = P[n_states + (k + 1) * (n_states + n_controls) - 2: n_states + (k + 1) * (n_states + n_controls)]
            
            state_error = st - state_ref
            control_error = con - control_ref
            
            obj += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([control_error.T, R, control_error])
            
            # Systemdynamik Constraints
            st_next = X[:, k + 1]
            f_value = self.model.f(st, con)
            st_next_euler = st + T * f_value  # Verwende T aus dem Parametervektor
            g = ca.vertcat(g, st_next - st_next_euler)
                            
        # Obstacle avoidance constraints (hard and soft)
        for k in range(N + 1):
            for i in range(self.n_obstacles):
                obs_x = P[-(3 * (i + 1) + 1)]
                obs_y = P[-(3 * (i + 1))]
                obs_diam = P[-(3 * (i + 1) - 1)]
                
                # Hard constraint: obstacle avoidance
                hard_obs_constraint = -ca.sqrt((X[0, k] - obs_x) ** 2 + (X[1, k] - obs_y) ** 2) + (rob_diam / 2 + obs_diam / 2)
                g = ca.vertcat(g, hard_obs_constraint)  # Ensure the robot does not collide with the obstacle

                # Soft constraint: penalty for proximity to obstacles
                soft_obs_cost = ca.fmax(0, (rob_diam / 2 + obs_diam / 2) - ca.sqrt((X[0, k] - obs_x) ** 2 + (X[1, k] - obs_y) ** 2))
                obj += self.penalty_weight * soft_obs_cost  # Penalize proximity to obstacles

        # Entscheidungsvariable zu einem Spaltenvektor machen
        OPT_variables = ca.vertcat(ca.reshape(X, n_states * (N + 1), 1), ca.reshape(U, n_controls * N, 1))
        
        nlp_prob = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}
        
        opts = {'ipopt.max_iter': 2000,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol': 1e-8,
                'ipopt.acceptable_obj_change_tol': 1e-6}

        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts)


    def solve_mpc(self, current_state, ref_traj, target_state, T, obstacles):
        self.obstacles = obstacles
        n_obstacles = self.n_obstacles
        N = self.N

        # Arguments for solver
        args = {
            'lbg': np.concatenate((np.zeros((3 * (self.N + 1), 1)), np.full((self.n_obstacles * (self.N + 1), 1), -np.inf))),  # Include obstacle constraints
            'ubg': np.concatenate((np.zeros((3 * (self.N + 1), 1)), np.full((self.n_obstacles * (self.N + 1), 1), 0))),  # Upper bound for inequality (obstacle)
            'lbx': np.full((3 * (self.N + 1) + 2 * self.N, 1), -ca.inf),  # Lower bounds for state and control variables
            'ubx': np.full((3 * (self.N + 1) + 2 * self.N, 1), ca.inf),  # Upper bounds for state and control variables
            'p': np.zeros((self.model.n_states + self.N * (self.model.n_states + self.model.n_controls) + 3 * self.n_obstacles + 1,))
        }
        
        # State bounds
        args['lbx'][0:3 * (N + 1):3] = -ca.inf
        args['ubx'][0:3 * (N + 1):3] = ca.inf
        args['lbx'][1:3 * (N + 1):3] = -ca.inf
        args['ubx'][1:3 * (N + 1):3] = ca.inf
        args['lbx'][2:3 * (N + 1):3] = -ca.inf
        args['ubx'][2:3 * (N + 1):3] = ca.inf

        args['lbx'][3 * (self.N + 1)::2] = self.v_min
        args['ubx'][3 * (self.N + 1)::2] = self.v_max
        args['lbx'][3 * (self.N + 1) + 1::2] = self.omega_min
        args['ubx'][3 * (self.N + 1) + 1::2] = self.omega_max
        args['p'][0:3] = current_state
        
        size_ref_traj = len(ref_traj)
        ref_traj_array = []
        
        prediction_distance = 2
        euklidean_distance = np.linalg.norm(current_state[:2] - target_state[:2], 2)
        # print("Target State: ", target_state)
        # print("Current State: ", current_state)
        # print("Euklidean Distance: ", euklidean_distance)
        # print("")

        # Verwende jeden Punkt in der Referenztrajektorie
        for k in range(self.N):
            if k < size_ref_traj:
                # Verwende die Werte direkt aus dem NumPy-Array
                x_ref, y_ref, theta_ref = ref_traj[k]
                
                ref_traj_array.append([x_ref, y_ref, theta_ref])

                u_ref = 0.5  # Beispielwert für die lineare Geschwindigkeit
                omega_ref = 0  # Beispielwert für die Winkelgeschwindigkeit
                
            if euklidean_distance < prediction_distance:
                x_ref = ref_traj[-1][0]
                y_ref = ref_traj[-1][1]
                theta_ref = self.normalize_angle(ref_traj[-1][2])
                u_ref = 0
                omega_ref = 0

            # Fülle das args['p']-Array mit den Referenzwerten
            args['p'][self.model.n_states + k * (self.model.n_states + self.model.n_controls):self.model.n_states + (k + 1) * (self.model.n_states + self.model.n_controls) - 2] = [x_ref, y_ref, theta_ref]
            args['p'][self.model.n_states + (k + 1) * (self.model.n_states + self.model.n_controls) - 2:self.model.n_states + (k + 1) * (self.model.n_states + self.model.n_controls)] = [u_ref, omega_ref]

        reserved_obstacles = n_obstacles - len(self.obstacles)

        offset = self.model.n_states + self.N * (self.model.n_states + self.model.n_controls)
        offset_reserved_obstacles = offset + 3 * len(self.obstacles)
        
        for i, obs in enumerate(self.obstacles):
            args['p'][offset + 3 * i: offset + 3 * (i + 1)] = obs
            self.viz.publish_obstacle_marker(obs)

        for i in range(reserved_obstacles):
            args['p'][offset_reserved_obstacles + 3 * i: offset_reserved_obstacles + 3 * (i + 1)] = [10000, 10000, -1000]
            
        # Füge T zum Parametervektor hinzu (letzter Eintrag in args['p'])
        args['p'][-1] = T
              
        self.viz.publish_refrence_trajectory(ref_traj_array)

        args['x0'] = np.concatenate((self.X0.T.flatten(), self.u0.T.flatten()))

        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'], lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = np.reshape(sol['x'][3 * (self.N + 1):].full(), (self.N, 2))
                            
        # Throw away the first optimal control input and shift the rest (Dimension: N x 2)                        
        self.u0 = np.vstack((u[1:, :], u[-1, :]))
        
        # Compute the new predicted trajectory based on the current state and control inputs
        X_pred = np.zeros((N + 1, 3))
        X_pred[0, :] = current_state

        for k in range(N):
            f_value = self.model.f(X_pred[k, :], u[k, :]).full().flatten()
            X_pred[k + 1, :] = X_pred[k, :] + T * f_value

        # Update X0 with the new predicted trajectory
        self.X0 = X_pred

        self.xx1.append(X_pred)
        self.viz.publish_predicted_trajectory(X_pred)
                    
        return u[0]


    def get_yaw_from_quaternion(self, quaternion):
        quaternion = [x / math.sqrt(sum([x * x for x in quaternion])) for x in quaternion]
        _, _, yaw = euler_from_quaternion(quaternion)
        return yaw

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
