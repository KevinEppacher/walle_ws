#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

def send_goal():
    # Initialisiere den Knoten
    rospy.init_node('send_goal_node', anonymous=True)

    # Publisher für das Ziel auf /move_base_simple/goal
    goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

    # Warte darauf, dass der Publisher initialisiert wird
    rospy.sleep(1)

    # Erstelle eine PoseStamped Nachricht
    goal_msg = PoseStamped()

    # Setze das Ziel im "map" Frame (oder "odom", je nach Anwendung)
    goal_msg.header.frame_id = "map"
    goal_msg.header.stamp = rospy.Time.now()

    # Setze die Position (x, y, z) und Orientierung (Quaternion) des Ziels
    # goal_msg.pose.position.x = 1.5
    # goal_msg.pose.position.y = 2.0
    # goal_msg.pose.position.z = 0.0
    
    goal_msg.pose.position.x = 7
    goal_msg.pose.position.y = 0.0
    goal_msg.pose.position.z = 0.0

    yaw = 0
    q = quaternion_from_euler(0.0, 0.0, yaw)    
    goal_msg.pose.orientation.x = q[0]
    goal_msg.pose.orientation.y = q[1]
    goal_msg.pose.orientation.z = q[2]
    goal_msg.pose.orientation.w = q[3]

    # Veröffentliche das Ziel
    rospy.loginfo("Sending goal to /move_base_simple/goal")
    goal_pub.publish(goal_msg)

    rospy.sleep(1)


if __name__ == '__main__':
    try:
        send_goal()
    except rospy.ROSInterruptException:
        pass