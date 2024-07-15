import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib.patches import Rectangle
from typing import Callable


class Visualizer:

    def __init__(self):

        self.fig, self.ax = plt.subplots()
        self.fig.set_size_inches(6, 4.5)

        self.ax.set_aspect('equal')
        self.ax.set_xlim([-10, 10])
        self.ax.set_ylim([-1, 10])
        self.ax.set_xlabel('x in m')
        self.ax.set_ylabel('y in m')
        self.ax.plot([-50, 50], [0, 0], 'black')

        # make plot interactive
        plt.ion()

        # define drone sprite dimensions
        self.w_drone = 0.8
        self.h_drone = 0.1

        # prepare the drone sprite visualization
        self.drone_sprite = Rectangle(
            xy=(self.w_drone/2, self.h_drone/2),
            width=self.w_drone,
            height=self.h_drone,
            angle=0,
            rotation_point='center',
            facecolor='red'
        )
        self.ax.add_patch(self.drone_sprite)

        # prepare scatter plots for trajectory visualization
        self.local_trajectory = self.ax.scatter(0, 0, marker='.', c='blue', s=2)
        self.global_trajectory = self.ax.scatter(0, 0, marker='.', c='black', s=1)
        
        # connect click event to click handler
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # stop program execution when window is closed
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        plt.show()

    def update_drone(self, x: int, y: int, theta: float):
        self.drone_sprite.set_xy(xy=(x-self.w_drone/2, y-self.h_drone/2))
        self.drone_sprite.set_angle(angle=math.degrees(theta))

    def update_local_trajectory(self, points: np.ndarray):
        self.local_trajectory.set_offsets(points)
        self.fig.canvas.draw_idle()

    def update_global_trajectory(self, points: np.ndarray):
        self.global_trajectory.set_offsets(points)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes:
            self.handle_click(event.xdata, event.ydata)

    def set_click_handler(self, handler: Callable):
        self.handle_click = handler

    def on_close(self, _):
        quit()