from matplotlib.patches import Rectangle
from math import degrees


def visualize_drone(x, y, theta, ax, color) -> Rectangle:
    w_drone = 0.8
    h_drone = 0.1
    theta = degrees(theta)
    drone = Rectangle((x-w_drone/2, y-h_drone/2), w_drone, h_drone,
                           angle=theta,
                           rotation_point='center',
                           facecolor=color)
    ax.add_patch(drone)

    return drone

def move_drone(drone: Rectangle, x, y, theta):
    w_drone = 0.8
    h_drone = 0.1
    drone.set_xy(xy=(x-w_drone/2, y-h_drone/2))
    drone.set_angle(angle=degrees(theta))