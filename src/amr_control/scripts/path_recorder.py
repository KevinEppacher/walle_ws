#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseWithCovarianceStamped

class PathRecorder:
    def __init__(self):
        # Initialisiere den Node
        rospy.init_node('path_recorder', anonymous=True)

        # Abonniere das amcl_pose-Topic
        self.pose_sub = rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)

        # Publisher für den Path
        self.path_pub_nMPC = rospy.Publisher('/robot_path_nMPC', Path, queue_size=10)
        self.path_pub_DWA = rospy.Publisher('/robot_path_DWA', Path, queue_size=10)

        # Erstelle eine Path-Nachricht
        self.path = Path()
        self.path.header.frame_id = "map"  # Passe dies an dein Frame an (z.B. 'map')

    def pose_callback(self, msg):
        # Erstelle eine PoseStamped-Nachricht aus dem /amcl_pose-Daten
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        # Füge die Pose zur Path-Nachricht hinzu
        self.path.poses.append(pose)

        # Aktualisiere den Header des Pfads
        self.path.header.stamp = rospy.Time.now()

        # Veröffentliche den Pfad
        self.path_pub_nMPC.publish(self.path)
        self.path_pub_DWA.publish(self.path)
        
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        path_recorder = PathRecorder()
        path_recorder.run()
    except rospy.ROSInterruptException:
        pass
