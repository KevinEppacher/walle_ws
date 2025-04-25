import casadi as ca
import numpy as np

class Obstacle:
    def __init__(self):
        self.define_obstacle()

    def define_obstacle(self):
        self.x = 0
        self.y = 0
        self.diam = 0
        self.obstacle_states = ca.vertcat(self.x, self.y, self.diam)
        self.n_obstacle_states = self.obstacle_states.size1()
        self.height = 1
        
    def set_obstacle(self, state):
        self.x = state[0]
        self.y = state[1]
        self.diam = state[2]