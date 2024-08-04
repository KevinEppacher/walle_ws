#!/usr/bin/env python3

# ROS1 imports
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist, Point, PoseWithCovarianceStamped, PoseArray, Pose
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

# Custom imports
from amr_control.controller import nMPC
from amr_control.robot_model import RobotModel
from amr_control.trajectory_planner import TrajectoryPlanner

def main():
    try:
        model = RobotModel()
        controller = nMPC(model)
        TrajectoryPlanner(model, controller, initial_state=[0, 0, 0], target_state=[1.5, 1.5, np.pi])
        # TrajectoryPlanner(model, controller, initial_state=[0, 0, 0], target_state=[0, 0, np.pi])
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
