#!/usr/bin/env python3

# ROS1 imports
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray  # Assuming you need multiple floats for position
from nav_msgs.msg import Odometry  # Import Odometry message type
from tf.transformations import euler_from_quaternion
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray
# Casadi imports
import casadi as ca
from casadi.tools import *
# Other imports
import numpy as np
import math

class Model:
    @staticmethod
    def predict(xk, uk, T):
        x_next = uk[0] * ca.cos(xk[2]) * T + xk[0]
        y_next = uk[0] * ca.sin(xk[2]) * T + xk[1]
        theta_next = uk[1] * T + xk[2]
        return ca.vertcat(x_next , y_next, theta_next)

class nMPC:
    def __init__(self, N, xRef, uRef, x0, S, Q, R, uMax, uMin, T=0.01):
        rospy.init_node('nmpc_node', anonymous=True)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.pub_control_input = rospy.Publisher('control_input', Float32MultiArray, queue_size=10)
        self.subscription = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        # Marker Publisher
        self.publisher = rospy.Publisher('/pose_markers', MarkerArray, queue_size=10)        
        
        print("Controller initialized")

        # Time step
        self.T = T
        # Prediction horizon
        self.N = N
        # Weight matrices for states
        self.Q = Q
        # Weight matrices for control inputs
        self.R = R
        # Terminal set
        self.S = S
        # Constraints on control inputs
        self.uMax = uMax
        self.uMin = uMin
        # Model parameters
        self.model = Model()
        # Reference signals
        self.xRef = xRef
        self.uRef = uRef
        # Initial state
        self.x0 = x0

        # Timer setup after all variables are initialized
        self.timer = rospy.Timer(rospy.Duration(self.T), self.controller_loop)
        
        # # Der nächste Zustand und Steuerung
        self.next_states = np.ones((3, self.N+1))
        self.next_states = self.next_states * self.x0.reshape(3, 1)
        self.u0 = np.zeros((2, self.N))

        self.setup_controller()

    def setup_controller(self):
        # Define optimization variables
        self.opti = ca.Opti()
        
        # Define optimization states
        self.opt_states = self.opti.variable(3, self.N+1)
        x = self.opt_states[:,0]
        y = self.opt_states[:,1]
        theta = self.opt_states[:,2]

        # Define optimization control inputs
        self.opt_controls = self.opti.variable(2, self.N)
        self.v = self.opt_controls[0]
        self.w = self.opt_controls[1]
        
        # Model equations
        model = Model()
        
        # Parameters, these parameters are the reference trajectories of the pose and inputs
        self.opt_x_ref = self.opti.parameter(3, self.N+1)
        self.opt_u_ref = self.opti.parameter(2, self.N)

        
        # Anfangsbedingung
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_predicted = model.predict(self.opt_states[:, i], self.opt_controls[:, i], self.T)
            self.opti.subject_to(self.opt_states[:, i+1] == x_predicted)
            
        # Cost function J
        obj = 0
        for i in range(self.N):
            state_error = self.opt_states[:,i] - self.opt_x_ref[:, i+1]
            control_error = self.opt_controls[:,i] - self.opt_u_ref[:,i]           
            obj += ca.mtimes([state_error.T, self.Q, state_error]) + ca.mtimes([control_error.T, self.R, control_error])
        self.opti.minimize(obj)
        
        # Begrenzungen der Steuerungen
        self.opti.subject_to(self.opti.bounded(self.uMin[0], self.v, self.uMax[0]))
        self.opti.subject_to(self.opti.bounded(self.uMin[1], self.w, self.uMax[1]))
        
        # IPOPT-Optionen
        opts = {
            'ipopt.max_iter': 2000,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-8,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }
        
        self.opti.solver('ipopt', opts)
        
    def solve(self, next_trajectories, next_controls):
        # Setzen der Parameter
        self.opti.set_value(self.opt_x_ref, next_trajectories.T)  # Transponieren der Trajektorien
        self.opti.set_value(self.opt_u_ref, next_controls.T)      # Transponieren der Steuerungen
        
        print("Current pose:", self.current_pose)
        # Setzen der Anfangsbedingung für die Zustände
        self.opti.set_initial(self.opt_states[:, 0], self.current_pose)
            
        # Anfangsschätzung für die Optimierungsziele
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0)
        
        # Problem lösen
        sol = self.opti.solve()
        
        # Erhalten der Steuerungseingaben
        self.u0 = sol.value(self.opt_controls)
        
        # self.next_states = sol.value(self.opt_states)
        
        predicted_states = [Model.predict(self.next_states[:, i], self.u0[:, i], self.T) for i in range(self.N)]

        # self.debug_print(next_trajectories.T, next_controls.T, predicted_states)
        
        return self.u0[:,0]
        
    def debug_print(self, next_trajectories, next_controls, predicted_states):
        print("\n--- Debugging Information ---")
        print("Referenz-Pose des Roboters bis N:")
        print(next_trajectories)
        print("\nDerzeitige Pose des Roboters:")
        print(self.next_states)
        print("\nReferenz-Steuerung u des Roboters bis N:")
        print(next_controls)
        print("\nVorhergesagte Zustände vom Modell bis N:")
        for i, state in enumerate(predicted_states):
            print(f"Schritt {i + 1}: {state}")
        print("--- End of Debugging Information ---\n")

    def get_yaw_from_quaternion(self, quaternion):
        # Ensure the quaternion is normalized
        norm = math.sqrt(sum([x * x for x in quaternion]))
        quaternion = [x / norm for x in quaternion]

        # Convert quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        
        return yaw

    def pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = self.get_yaw_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_pose = np.array([position.x, position.y, yaw])
        
    def publish_marker_trajectories(self, positions):
        marker_array = MarkerArray()
        
        for i, pos in enumerate(positions):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "reference_trajectory"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = 0.0
            quaternion = tf.transformations.quaternion_from_euler(0, 0, pos[2])
            marker.pose.orientation.x = quaternion[0]
            marker.pose.orientation.y = quaternion[1]
            marker.pose.orientation.z = quaternion[2]
            marker.pose.orientation.w = quaternion[3]
            marker.scale.x = 0.25  # Länge des Pfeils
            marker.scale.y = 0.05  # Breite des Pfeils
            marker.scale.z = 0.05  # Höhe des Pfeils
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            marker_array.markers.append(marker)
            
            self.publisher.publish(marker_array)
            
    def compute_straight_trajectory(self, start_pos, end_pos, num_points):
        return np.linspace(start_pos, end_pos, num_points)

    def controller_loop(self, event):
        
        init_pose = self.current_pose
        end_pose = [2.0, 0.5, 0.0]

        # Beispielhafte Referenztrajektorien und Steuerungen
        next_trajectories = np.tile(init_pose, (self.N+1, 1))
        
        next_trajectories = self.compute_straight_trajectory(init_pose, end_pose, self.N+1)
                
        self.publish_marker_trajectories(next_trajectories)

        next_controls = np.zeros((self.N, 2))
               
        control_input = self.solve(next_trajectories, next_controls)
        
        self.pub_cmd_vel.publish(Twist(linear=Point(x=control_input[0], y=0.0, z=0.0), angular=Point(x=0.0, y=0.0, z=control_input[1])))
        
        self.pub_control_input.publish(Float32MultiArray(data=control_input))
        print("Optimale Steuerungseingabe:", control_input)


def main():
    try:
        N = 100  # Prediction horizon

        # Initial state
        x0 = np.array([0.5, 0.5, 0.0])  # Initial state (x, y, theta)

        # Reference trajectory
        xRef = np.zeros((N+1, 3))  # Reference states
        for i in range(N+1):
            xRef[i, 0] = i * 1  # x reference position
            xRef[i, 1] = i * 1 # y reference position
            xRef[i, 2] = 0.0  # theta reference angle

        # Reference control inputs
        uRef = np.zeros((N, 2))  # Reference controls
        for i in range(N):
            uRef[i, 0] = 1  # Linear velocity reference
            uRef[i, 1] = 0.0  # Angular velocity reference

        # Weight matrices for states and controls
        Q = np.diag([10.0, 10.0, 1.0])  # State error weights
        R = np.diag([1.0, 1.0])  # Control effort weights
        S = np.diag([10.0, 10.0, 1.0])  # Terminal state weights

        # Constraints on control inputs
        uMax = np.array([0.22, 0.22])  # Max linear and angular velocity
        uMin = np.array([-0.22, -0.22])  # Min linear and angular velocity

        # Time step
        T = 0.1  # Time step for the controller
        
        controller = nMPC(N, xRef, uRef, x0, S, Q, R, uMax, uMin, T)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
