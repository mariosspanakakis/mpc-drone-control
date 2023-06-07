import numpy as np
from math import sin, cos


class Drone:

    def __init__(self, constants: dict, initial_state: np.ndarray, Ts: float):
        # physical constants
        self.m = constants['m']
        self.L = constants['L']
        self.J = constants['J']
        self.c = constants['c']
        self.g = constants['g']
        # sample time
        self.Ts = Ts
        # drone state
        self.z = initial_state
        # operating point for linearization
        self.z_OP = np.array([0, 0, 0, 0, 0, 0])
        self.u_OP = np.array([self.g/(2*self.c), self.g/(2*self.c)])

    # integrate the system dynamics to obtain the follow-up state
    def step(self, u: np.ndarray):
        # obtain resulting thrust values
        thrust_tot = (u[0] + u[1]) * self.c
        thrust_dif = (u[1] - u[0]) * self.c
        # integrate system dynamics
        self.z[0] += self.z[1] * self.Ts
        self.z[1] += 1./self.m * (-thrust_tot) * sin(self.z[4]) * self.Ts
        self.z[2] += self.z[3] * self.Ts
        self.z[3] += 1./self.m * (thrust_tot * cos(self.z[4]) - self.g) * self.Ts
        self.z[4] += self.z[5] * self.Ts
        self.z[5] += 1./self.J * thrust_dif * self.L * self.Ts

    # discretize the system and return the state space representation
    def linearize(self):
        # continuous state space representation
        self.A = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, -self.g/self.m, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ])
        self.B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [self.c/self.m, self.c/self.m],
            [0, 0],
            [-self.c*self.L/self.J, self.c*self.L/self.J]
        ])

        # discrete state space representation
        self.Ad = np.eye(6) + self.A * self.Ts
        self.Bd = self.B * self.Ts

        return self.Ad, self.Bd


class DroneLinear:

    def __init__(self, constants: dict, initial_state: np.ndarray, Ts: float):
        # physical constants
        self.m = constants['m']
        self.L = constants['L']
        self.J = constants['J']
        self.c = constants['c']
        self.g = constants['g']
        # sample time
        self.Ts = Ts
        # drone state
        self.z = initial_state
        # operating point
        self.z_OP = np.array([0, 0, 0, 0, 0, 0])
        self.u_OP = np.array([self.g/(2*self.c), self.g/(2*self.c)])

        # continuous state space representation
        self.A = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, -self.g/self.m, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ])
        self.B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [self.c/self.m, self.c/self.m],
            [0, 0],
            [-self.c*self.L/self.J, self.c*self.L/self.J]
        ])

        # discrete state space representation
        self.Ad = np.eye(6) + self.A * self.Ts
        self.Bd = self.B * self.Ts

    # simulate the system dynamics
    def step(self, u: np.ndarray):
        self.z = self.Ad @ (self.z - self.z_OP) + self.Bd @ u