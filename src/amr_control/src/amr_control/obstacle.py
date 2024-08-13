import casadi as ca
import numpy as np

class Obstacle:
    def __init__(self):
        self.define_obstacle()

    def define_obstacle(self):
        self.x = 4
        self.y = 4
        self.diam = 0.3
        self.height = 1