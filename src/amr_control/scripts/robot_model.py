import casadi as ca
import numpy as np

class RobotModel:
    def __init__(self):
        print("RobotModel initialized")
        self.define_model()

    def define_model(self):
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        self.states = ca.vertcat(x, y, theta)
        self.n_states = self.states.size1()

        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        self.controls = ca.vertcat(v, omega)
        self.n_controls = self.controls.size1()

        rhs = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
        self.f = ca.Function('f', [self.states, self.controls], [rhs])
