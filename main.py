import numpy as np
import math
from copy import copy
import matplotlib.pyplot as plt
import yaml

from drone import Drone
from planner import LinearTrajectoryPlanner
from controller import LinearQuadraticMPC
from visualizer import visualize_drone, move_drone


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
    4.,                 # y
    0.,                 # y_dot
    0.,                 # theta
    0.                  # theta_dot
])

xy_des = np.array([initial_state[0], initial_state[2]])

# nonlinear drone model
drone = Drone(
    constants=drone_constants, initial_state=copy(initial_state), Ts=Ts)
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

# parameter set obtained via Bayesian Optimization; does not perform too well for general trajectories
"""opt_params = {'q_x': 65.6121719067622,
              'q_y': 86.39821246377763,
              'q_xy_dot': 2.18474740680037,
              'q_theta': 32.17697421563099,
              'q_theta_dot': 1713.808254107117,
              'q_u': 4.386436533929654}"""

# define weight matrices
Qx = np.diag([params['qx'], params['qy'], params['qvx'], params['qvy'], params['qtheta'], params['qdtheta']])
Qu = np.diag([params['qu'], params['qu']])
#Qx = np.diag([opt_params['q_x'], opt_params['q_xy_dot'], opt_params['q_y'], opt_params['q_xy_dot'], opt_params['q_theta'], opt_params['q_theta_dot']])
#Qu = np.diag([opt_params['q_u'], opt_params['q_u']])
Qn = Qx

# generate linear MPC controller
K = LinearQuadraticMPC(Ad=Ad, Bd=Bd, N=N,
                       xmin=xmin, xmax=xmax, umin=umin, umax=umax,
                       Qx=Qx, Qu=Qu, Qn=Qn, info=False)

# trajectory planner
planner = LinearTrajectoryPlanner(N=N, Ts=Ts, initial_state=initial_state)

# visualization
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim([-10, 10])
ax.set_ylim([-1, 20])
ax.plot([-50, 50], [0, 0], 'black')
# make plot interactive
plt.ion()
# generate a rectangle visualizing the drone
drone_rect = visualize_drone(x=drone.z[0], y=drone.z[2], theta=drone.z[4], ax=ax, color='red')
# prepare a scatter plot object for path visualization
path_scatter = ax.scatter(0, 0, marker='.', c='blue', s=2)

plt.show()

# assign user click coordinates as new goal for the drone
def on_click(event):
    if event.inaxes:
        planner.calculate_trajectory(P=np.array([drone.z[0], drone.z[2]]),
                                     Q=np.array([event.xdata, event.ydata]),
                                     v_max=4., a=1.)
    #K.update_weight_matrices(Qx=Qx, Qu=Qu, Qn=Qn)
plt.connect('button_press_event', on_click)

# stop program execution when window is closed
def on_close(event):
    quit()
fig.canvas.mpl_connect('close_event', on_close)

# animation cycle
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
    path_scatter.set_offsets(np.c_[planner.partial_trajectory[:,0], planner.partial_trajectory[:,2]])
    fig.canvas.draw_idle()
    
    # update the drone visualization
    move_drone(drone=drone_rect, x=drone.z[0], y=drone.z[2], theta=drone.z[4])

    plt.pause(0.001)