#!/usr/bin/env python
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import math

def move_cube():
    rospy.init_node('move_cube_up_down')

    # Warte, bis der Service verfügbar ist
    rospy.wait_for_service('/gazebo/set_model_state')

    # Erzeuge einen Proxy für den Service
    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    state_msg = ModelState()
    state_msg.model_name = 'cube'  # Name des Modells in Gazebo

    rate = rospy.Rate(10)  # Frequenz: 10 Hz
    t = 0

    while not rospy.is_shutdown():
        # Bewege den Würfel nach oben und unten mit einer Sinus-Funktion
        x_position = 3.0 + math.cos(t) / 2
        y_position = math.sin(t) / 2
        t += 0.04

        # Setze die neue Position
        state_msg.pose.position.z = 0.1
        state_msg.pose.position.x = x_position
        state_msg.pose.position.y = y_position

        # Rufe den Service auf, um den Zustand zu setzen
        set_model_state(state_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        move_cube()
    except rospy.ROSInterruptException:
        pass
