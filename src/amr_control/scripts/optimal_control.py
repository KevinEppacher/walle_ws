#!/usr/bin/env python3

# ROS1 imports
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray  # Assuming you need multiple floats for position
from nav_msgs.msg import Odometry  # Import Odometry message type
from tf.transformations import euler_from_quaternion

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


class Optimizer:
    def __init__(self):
        print("Optimizer initialized")
        # Define optimization variables
        opti = ca.Opti()
        X = opti.variables(3, self.N+1)
        U = opti.variables(2, self.N)

        opti.subject_to(X[:,0] == self.x0)

        for k in range(self.N):
            xk = X[:,k]
            xk_next = X[:,k+1]
            uk = U[:,k]
            opti.subject_to(xk_next == self.model.predict(uk, self.dt))

class nMPC:
    def __init__(self, N, xRef, uRef, x0, S, Q, R, uMax, uMin, T=0.01):
        rospy.init_node('nmpc_node', anonymous=True)
        self.publisher_ = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.subscription = rospy.Subscriber('odom', Odometry, self.odom_callback)
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
        # self.next_states = np.ones((self.N+1, 3)) * init_pos
        # self.u0 = np.zeros((self.N, 2))

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

        # model.predict(self.opt_states, self.opt_controls, self.T)

        
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
        print(obj)



    def get_yaw_from_quaternion(self, quaternion):
        # Ensure the quaternion is normalized
        norm = math.sqrt(sum([x * x for x in quaternion]))
        quaternion = [x / norm for x in quaternion]

        # Convert quaternion to Euler angles
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        
        return yaw

    def odom_callback(self, msg):
        # Extract quaternion from the message
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        
        # Extract position and orientation from the odometry message
        self.model.x = msg.pose.pose.position.x
        self.model.y = msg.pose.pose.position.y
        self.model.theta = self.get_yaw_from_quaternion(quaternion)

    def controller_loop(self, event):
        self.model.x = 0.0
        self.model.y = 0.0
        # opt = Optimizer()
        # opt.solve()
        
    def cost_function(self):
        

        return 0



def main():
    try:
        N = 10  # Prediction horizon

        # Initial state
        x0 = np.array([0.0, 0.0, 0.0])  # Initial state (x, y, theta)

        # Reference trajectory
        xRef = np.zeros((N+1, 3))  # Reference states
        for i in range(N+1):
            xRef[i, 0] = i * 0.1  # x reference position
            xRef[i, 1] = 0.0  # y reference position
            xRef[i, 2] = 0.0  # theta reference angle

        # Reference control inputs
        uRef = np.zeros((N, 2))  # Reference controls
        for i in range(N):
            uRef[i, 0] = 0.1  # Linear velocity reference
            uRef[i, 1] = 0.0  # Angular velocity reference

        # Weight matrices for states and controls
        Q = np.diag([10.0, 10.0, 1.0])  # State error weights
        R = np.diag([1.0, 1.0])  # Control effort weights
        S = np.diag([10.0, 10.0, 1.0])  # Terminal state weights

        # Constraints on control inputs
        uMax = np.array([1.0, math.pi/4])  # Max linear and angular velocity
        uMin = np.array([-1.0, -math.pi/4])  # Min linear and angular velocity

        # Time step
        T = 0.1  # Time step for the controller
        
        controller = nMPC(N, xRef, uRef, x0, S, Q, R, uMax, uMin, T)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
