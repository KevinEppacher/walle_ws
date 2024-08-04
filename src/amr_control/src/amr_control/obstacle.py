import casadi as ca
import numpy as np

class ObstacleAvoidance:
    def __init__(self):
        print("Obstacle initialized")
        self.define_obstacle()

    def define_obstacle(self):
        self.x = 0.8
        self.y = 0.8
        self.diam = 0.3