import numpy as np


class LinearTrajectoryPlanner:
    def __init__(self, N: int, Ts: float, initial_state: np.ndarray):
        """
        Trajectory planner to generate a simple trajectory connecting
        two points by a line.
        
        Arguments:
        -----
        N             : int
                        prediction horizon, equal to the MPCs horizon
        Ts            : float
                        sampling time, equal to the MPCs sampling time
        initial_state : 1D array-like (nx)
                        initial state of the controlled system
        """
        self.N = N
        self.Ts = Ts
        
        self.static_trajectory = np.ones((1, 6)) * initial_state
        self.len_static_trajectory = 1
        self.traj_idx = 0
    
    def calculate_trajectory(self, P, Q, v_max, a):
        """
        Calculate a linear trajectory connecting points P and Q. Accelerate and 
        decelerate the motion in accordance to the given parameters.
        
        Arguments:
        -----
        P             : 1D array-like (nx)
                        starting point
        Q             : 1D array-like (nx)
                        goal point
        v_max         : float
                        maximum allowed velocity on the trajectory
        a             : float
                        acceleration and deceleration
        """

        # total distance to travel
        d_tot = np.sqrt(sum((Q - P)**2))
        # travel direction
        dir = (Q - P) / d_tot

        # acceleration phase duration
        t_acc = v_max/a
        # distance travelled during acceleration/deceleration
        d_acc = 0.5 * a * t_acc**2

        # if the maximum velocity is reachable:
        if d_acc < d_tot / 2:
            # distance travelled at constant speed
            d_const = d_tot - 2 * d_acc
            # time at maximum speed
            t_const = d_const / v_max
            # total time travelled
            t_tot = t_const + 2 * t_acc

            time = np.linspace(0, t_tot, int(t_tot/self.Ts))
            xy_trajectory = np.zeros((int(t_tot/self.Ts), 2))

            for idx, t in enumerate(time):
                # acceleration phase
                if t < t_acc:
                    xy_trajectory[idx] = P + 0.5 * a * t**2 * dir
                # constant speed phase
                elif t > t_acc and t < t_const + t_acc:
                    xy_trajectory[idx] = xy_trajectory[idx-1] + v_max * self.Ts * dir
                # deceleration phase
                else:
                    xy_trajectory[idx] = Q - 0.5 * a * (t_tot - t)**2 * dir

        # if the maximum velocity is not reachable:
        else:
            # total travelling time
            t_tot = 2 * np.sqrt((2 * d_tot)/a)

            time = np.linspace(0, t_tot, int(t_tot/self.Ts))
            xy_trajectory = np.zeros((int(t_tot/self.Ts), 2))

            for idx, t in enumerate(time):
                # acceleration phase
                if t < t_tot/2:
                    xy_trajectory[idx] = P + 0.5 * a * t**2 * dir/2
                # deceleration phase
                else:
                    xy_trajectory[idx] = Q - 0.5 * a * (t_tot - t)**2 * dir/2

        trajectory = np.zeros((int(t_tot/self.Ts), 6))

        # augment the (until now purely in xy-coordinates) system state to be 6D and include velocity
        for idx, state in enumerate(trajectory):
            trajectory[idx] = np.array([xy_trajectory[idx, 0] * np.sin(np.pi*idx/trajectory.shape[0]),  # add sine wave to make the trajectory more interesting
                                        (xy_trajectory[idx, 0] - xy_trajectory[idx-1, 0])/self.Ts,
                                        xy_trajectory[idx, 1],
                                        (xy_trajectory[idx, 1] - xy_trajectory[idx-1, 1])/self.Ts,
                                        0.,
                                        0.])

        self.static_trajectory = trajectory
        self.len_static_trajectory = trajectory.shape[0]
        self.traj_idx = 0

        return trajectory

    def extract_partial_trajectory(self):
        """
        Extracts from the system trajectory the upcoming part, which is an N-samples
        long snippet of it. If the trajectory has come to its end, the end state is
        appended to stabilize the system in it.        
        """
        # passing over the given trajectory
        if self.traj_idx + self.N + 1 < self.len_static_trajectory:
            self.partial_trajectory = self.static_trajectory[self.traj_idx:self.traj_idx+self.N+1, :]
        else:
            # before having reached the end
            if self.traj_idx < self.len_static_trajectory:
                self.partial_trajectory = np.vstack((self.static_trajectory[self.traj_idx:, :],
                                                     np.tile(self.static_trajectory[-1, :],
                                                             (self.traj_idx + self.N + 1 - self.len_static_trajectory, 1))))
        
        self.traj_idx += 1
