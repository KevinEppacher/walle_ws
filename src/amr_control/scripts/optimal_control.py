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
from casadi import *
from casadi.tools import *
# Other imports
import numpy as np
import math

class Model:
    def __init__(self, x=0.0, y=0.0, theta=0.0, wheel_radius=0.05, axle_length=0.15):
        self.x = x
        self.y = y
        self.theta = theta
        self.wheel_radius = wheel_radius
        self.axle_length = axle_length

    def predict(self, u, dt):
        v, w = u
        x = self.x
        y = self.y
        theta = self.theta

        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt

        return x, y, theta

class Optimizer:
    def __init__():
        pass

    def solve():
        pass


class nMPC:
    def __init__(self, N, xRef, uRef, x0, S, Q, R, uMax, uMin, dt=0.01):
        rospy.init_node('nmpc_node', anonymous=True)
        self.publisher_ = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.subscription = rospy.Subscriber('odom', Odometry, self.odom_callback)
        
        # Initialize parameters after subscriptions to avoid missing initial updates
        self.model = Model()
        self.N = N
        self.dt = dt
        self.xRef = xRef
        self.uRef = uRef
        self.x0 = x0
        self.Q = Q
        self.R = R
        self.S = S
        self.uMax = uMax
        self.uMin = uMin

        # Timer setup after all variables are initialized
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.timer_callback)

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

    def timer_callback(self, event):
        u = (0.1, 0.1)
        predicted_pose = self.model.predict(u, self.dt)
        
        # Publish control commands
        msg = Twist()
        msg.linear.x = 1.0
        msg.angular.z = 0.0
        self.publisher_.publish(msg)



def main():
    try:
        controller = nMPC(10, None, None, None, None, None, None, None, None)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
