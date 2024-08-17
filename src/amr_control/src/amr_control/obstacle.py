import casadi as ca
import numpy as np

class Obstacle:
    def __init__(self):
        self.define_obstacle()

    def define_obstacle(self):
        self.x = 4
        self.y = 1
        self.diam = 0.8
        self.height = 1