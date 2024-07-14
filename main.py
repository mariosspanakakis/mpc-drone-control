import numpy as np
import math
from copy import copy
import matplotlib.pyplot as plt
import yaml

from drone import Drone
from planner import LinearTrajectoryPlanner, RRTPlanner
from controller import LinearQuadraticMPC
from visualizer import Visualizer

with open('parameters.yaml', 'r') as file:
    params = yaml.safe_load(file)

# prediction horizon and sample time
N = params['N']
Ts = params['Ts']

drone_constants = {
    'm': params['m'],           # mass
    'L': params['L'],           # rotor distance from center of mass
    'J': params['J'],           # moment of inertia
    'c': params['c'],           # thrust constant
    'g': params['g']            # gravitational constant
}

initial_state = np.array([
    0.,                 # x
    0.,                 # x_dot
    2.,                 # y
    0.,                 # y_dot
    0.,                 # theta
    0.                  # theta_dot
])

xy_des = np.array([initial_state[0], initial_state[2]])

# nonlinear drone model
drone = Drone(constants=drone_constants, initial_state=copy(initial_state), Ts=Ts)
# linearization and discretization of the model to obtain the matrices Ad and Bd
Ad, Bd = drone.linearize()

# state constraints
xmin = np.array([
    -np.inf,            # x
    -4.,                # xdot
    0.2,                # y
    -4.,                # ydot
    -math.pi/8,         # theta
    -np.inf             # thetadot
])
xmax = np.array([
    np.inf,             # x
    4.,                 # xdot
    np.inf,             # y
    4.,                 # ydot
    math.pi/8,          # theta
    np.inf              # thetadot
])

# input constraints, shifted by standard input at operating point
umin = np.array([0., 0.]) - drone.u_OP
umax = np.array([1., 1.]) - drone.u_OP

# define weight matrices
Qx = np.diag([params['qx'], params['qy'], params['qvx'], params['qvy'], params['qtheta'], params['qdtheta']])
Qu = np.diag([params['qu'], params['qu']])
Qn = Qx

# set up linear MPC controller
K = LinearQuadraticMPC(
    Ad=Ad, Bd=Bd, N=N,
    xmin=xmin, xmax=xmax, umin=umin, umax=umax,
    Qx=Qx, Qu=Qu, Qn=Qn, info=False
)

# set up trajectory planner
planner = LinearTrajectoryPlanner(N=N, Ts=Ts, initial_state=initial_state)

# set up visualization and plotting module
visualizer = Visualizer()

# assign user click coordinates as new goal for the drone
def update_setpoint(x, y):
    planner.calculate_trajectory(
        P=np.array([drone.z[0], drone.z[2]]),
        Q=np.array([x, y]),
        v_max=4.,
        a=1.
    )

visualizer.set_click_handler(update_setpoint)

while True:
    # get current drone position
    xy_act = (drone.z[0], drone.z[2])

    # extract the relevant part of the static trajectory
    planner.extract_partial_trajectory()
    partial_trajectory = planner.partial_trajectory

    # get the optimal control input
    u = K.solve(x0=drone.z, trajectory=partial_trajectory) + drone.u_OP

    # apply the control input
    drone.step(u=u)

    # update path data
    visualizer.update_local_trajectory(
        points=np.c_[
            planner.partial_trajectory[:,0],
            planner.partial_trajectory[:,2]
        ]
    )
    
    # update the drone visualization
    visualizer.update_drone(x=drone.z[0], y=drone.z[2], theta=drone.z[4])

    plt.pause(0.001)