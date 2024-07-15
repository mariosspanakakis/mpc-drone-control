import numpy as np
from scipy.interpolate import CubicSpline

from rrt import RRT


class TrajectoryPlanner:

    def __init__(self):
        self.stepsize = 1
        self.update_global_trajectory(P=np.array([0, 2]), Q=np.array([0, 2]))

    def update_global_trajectory(self, P: np.ndarray, Q: np.ndarray):

        # RRT path planning
        self.rrt = RRT(
            P,
            Q,
            map_lims=np.array([[-10, 10], [0, 10]]),
            step_size=0.5,
            max_iter=1000
        )
        self.global_trajectory = self.rrt.plan()

        """dist = np.linalg.norm(P - Q)
        if (dist > 0):
            # linear path planning
            #num = int(dist / self.stepsize)
            #self.global_trajectory = np.linspace(start=P, stop=Q, num=num)

        else:
            self.global_trajectory = np.array([[0, 2]])"""

    # get an N-samples long local trajectory
    def update_local_trajectory(self, x: float, y: float, N: int, Ts: float):

        # find closest point of the global trajectory
        dists = np.linalg.norm(np.array([x, y]) - self.global_trajectory, axis=1)
        closest_idx = np.argmin(dists)

        # add current position to upcoming global trajectory
        points = np.vstack((np.array([x, y]), self.global_trajectory[closest_idx:]))

        # define coordinates along the splines
        k = np.linspace(0., 1., len(points))

        # interpolate the global trajectory using cubic splines
        xspline = CubicSpline(x=k, y=points[:,0])
        yspline = CubicSpline(x=k, y=points[:,1])

        # obtain the spline's total length
        spline_length = self.__get_spline_length(xspline, yspline)

        # obtain the curvature spline
        curvature_spline = self.__get_spline_curvature(xspline, yspline)

        # integrate the velocities along the spline to calculate distance per time
        locations = np.array([0.0])
        for _ in range(N - 1):
            v = self.__get_velocity_at_location(curvature_spline, x=locations[-1]/spline_length)
            next_location = min(1.0, (locations[-1] + v * Ts / spline_length))
            locations = np.append(locations, next_location)

        # calculate the resulting local trajectory in x and y
        self.local_xy_trajectory = np.column_stack((xspline(locations), yspline(locations)))

        # augment the (until now purely in xy coordinates) local trajectory
        self.local_trajectory = np.column_stack((
            self.local_xy_trajectory[:,0],
            np.diff(self.local_xy_trajectory[:,0], append=self.local_xy_trajectory[-1,0]) / Ts,
            self.local_xy_trajectory[:,1],
            np.diff(self.local_xy_trajectory[:,1], append=self.local_xy_trajectory[-1,1]) / Ts,
            np.zeros(N),
            np.zeros(N)
        ))

    def __get_velocity_at_location(self, curvature_spline: CubicSpline, x: float) -> float:
        vel = 2.0 * np.exp(- curvature_spline(x)**2 / (2 * 500**2))
        return vel

    def __get_spline_curvature(self, xspline: CubicSpline, yspline: CubicSpline) -> CubicSpline:
        
        t = np.linspace(0., 1., 1000)

        # first derivatives
        dx_dt = xspline(t, 1)
        dy_dt = yspline(t, 1)

        # second derivatives
        d2x_dt2 = xspline(t, 2)
        d2y_dt2 = yspline(t, 2)

        # curvature profile
        curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2)**(3/2)
        curvature_spline = CubicSpline(x=t, y=curvature)

        return curvature_spline

    def __get_spline_length(self, xspline: CubicSpline, yspline: CubicSpline) -> float:
        # evaluate the given spline at locations along its length
        t = np.linspace(0., 1., 100)
        points = np.column_stack((xspline(t), yspline(t)))

        # calculate the Euclidean distances between each pair of points
        diffs = np.diff(points, axis=0)
        dists = np.linalg.norm(diffs, axis=1)

        # calculate the total length
        length = np.sum(dists)

        return length