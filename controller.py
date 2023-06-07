import numpy as np
import osqp
import scipy.sparse as sparse
import matplotlib.pyplot as plt


class LinearQuadraticMPC:
    def __init__(self, Ad, Bd, N, xmin, xmax, umin, umax, Qx, Qu, Qn, info):
        """
        Implements a linear quadratic MPC controller.
        
        Arguments:
        -----
        Ad    : 2D array-like (nx, nx)
                discretized system matrix  
        Bd    : 2D array-like (nx, nu)
                discretized input matrix
        N     : int
                prediction horizon
        xmin  : 1D array-like (nx)
                state constraint minimum vector
        xmax  : 1D array-like (nx)
                state constraint maximum vector
        umin  : 1D array-like (nu)
                input constraint minimum vector
        umax  : 1D array-like (nu)
                input constraint maximum vector
        Qx    : 2D array-like (nx, nx)
                state cost matrix, typically diagonal
        Qu    : 2D array-like (nu, nu)
                input cost matrix, typically diagonal
        Qn    : 2D array-like (nx, nx)
                terminal state cost matrix, typically diagonal
        info  : boolean
                set True to print the matrices of the optimal control problem
        """

        self.Ad = Ad
        self.Bd = Bd

        nx, nu = np.shape(Bd)
        self.nx = nx
        self.nu = nu
        
        self.N = N

        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax

        self.Qx = Qx
        self.Qu = Qu
        self.Qn = Qn

        self.info = info

        # lift all matrices to cover the full prediction horizon
        # lifted system matrix
        self.A = np.zeros((nx*(N+1), nx))
        for i in range(0, N+1):
            self.A[i*nx:(1+i)*nx, :] = np.linalg.matrix_power(Ad, i)        
        if self.info: print(f"A:\n{self.A}")

        # lifted input matrix
        self.B = np.zeros((nx*(N+1), nu*N))
        for i in range(0, N):
            self.B[(i+1)*nx:(i+2)*nx, i*nu:(i+1)*nu] = Bd
        col = self.A[0:-nx]
        self.B[1*nx:, 0*nu:1*nu] = col[0:] @ Bd
        for i in range(1, N):
            self.B[(i+1)*nx:, i*nu:(i+1)*nu] = col[0:-i*nx] @ Bd
        if self.info: print(f"B:\n{self.B}")

        # state cost matrix
        self.Q = np.block([
            [np.kron(np.eye(N), Qx), np.zeros((nx*N, nx))],
            [np.zeros((nx, nx*N)), Qn]
        ])
        if self.info: print(f"Q:\n{self.Q}")

        # input cost matrix
        R = np.kron(np.eye(N), Qu)
        if self.info: print(f"R:\n{R}")
        
        # input constraints
        self.Umin = np.kron(np.ones(N), umin)
        self.Umax = np.kron(np.ones(N), umax)

        # assemble QP matrices
        P = self.B.T @ self.Q @ self.B + R
        if self.info: print(f"P:\n{P}")
        K = np.vstack((self.B, np.eye(nu*N)))
        if self.info: print(f"K:\n{K}")

        # define the QP problem
        self.problem = osqp.OSQP()

        # setup the QP problem using dummy values
        q, lb, ub = self._assemble_qp_matrices(np.zeros(nx),
                                              np.zeros(((N+1), nx)))
        self.problem.setup(sparse.csc_matrix(P), q, sparse.csc_matrix(K),
                           lb, ub, warm_start=True, verbose=False)

    # compute the QP problem matrices for the next optimization
    def _assemble_qp_matrices(self, x0, trajectory):
        """
        Assembles the matrices of the QP problem that depend on the initial state.
        
        Arguments:
        -----
        x0:        : 1D array-like (nx)
                     initial system state
        trajectory : 2D array-like (N, nx)
                     reference trajectory over the prediction horizon
        """
        Xr = np.concatenate(trajectory)
        q = (x0.T @ self.A.T - Xr) @ self.Q @ self.B
        Xmin = np.kron(np.ones(self.N+1), self.xmin) - self.A @ x0
        Xmax = np.kron(np.ones(self.N+1), self.xmax) - self.A @ x0
        lb = np.concatenate((Xmin, self.Umin))
        ub = np.concatenate((Xmax, self.Umax))

        return q, lb, ub

    # solve the QP problem and return the optimal input u
    def solve(self, x0, trajectory):
        """
        Given an initial state, solve the QP problem and determine the
        optimal control input u.
        
        Arguments:
        -----
        x0:        : 1D array-like (nx)
                     initial system state
        trajectory : 2D array-like (N, nx)
                     reference trajectory over the prediction horizon
        """

        # update the QP problem
        q, lb, ub = self._assemble_qp_matrices(x0, trajectory)
        self.problem.update(q=q, l=lb, u=ub)

        # solve the QP problem and check solver status
        result = self.problem.solve()
        if result.info.status != 'solved':
            raise ValueError('No solution found!')

        # return the first element of the optimal input sequence u
        u_opt = np.array(result.x[0:self.nu])
        return u_opt


if __name__ == '__main__':
    """
    Applies the linear MPC controller class to a simple dynamic system
    and plots the system response.
    """

    # constants and initial state
    x0 = np.array([10., 0])
    Ts = 0.1
    Te = 30.
    m = 2.
    b = 0.3
    N = 20

    # reference trajectory, set to a constant value
    trajectory = np.ones((2, N+1)) * 2.

    # continuous system description
    Ac = np.array([
        [0., 1.],
        [0., -b/m]
    ])
    Bc = np.array([
        [0.],
        [1./m]
    ])

    # discrete system description
    [nx, nu] = Bc.shape
    Ad = np.eye(nx) + Ac * Ts
    Bd = Bc * Ts

    # state and input limits
    xmin = np.array([0., -np.inf])
    xmax = np.array([12., np.inf])
    umin = np.array([-1.])
    umax = np.array([1.])

    # weight matrices
    Qx = np.diag([0.5, 0.1])
    Qu = np.array([2.])
    Qn = 4 * Qx

    # controller
    K = LinearQuadraticMPC(Ad=Ad, Bd=Bd, N=N, xmin=xmin, xmax=xmax,
                           umin=umin, umax=umax, Qx=Qx, Qu=Qu, Qn=Qn, info=False)

    # simulation data
    tsim = np.linspace(0, Te, int(Te/Ts))
    usim = []
    xsim = []
    x = x0

    for i in tsim:
        u = K.solve(x0=x, trajectory=trajectory)
        x = Ad @ x + Bd @ u

        usim.append(u)
        xsim.append(x[0])

    fig, axs = plt.subplots(2)

    axs[0].plot(tsim, xsim)
    axs[0].set_ylabel('x')
    axs[1].plot(tsim, usim)
    axs[1].set_ylabel('u')
    axs[1].set_xlabel('t')

    plt.show()