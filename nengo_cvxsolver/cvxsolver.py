import cvxpy
import nengo
import numpy as np


class CVXBoundedSolver(nengo.solvers.Solver):
    def __init__(self, reg=0.01, upper=1, lower=-1):
        super(CVXBoundedSolver, self).__init__(weights=False)
        self.reg = reg
        self.upper = upper
        self.lower = lower

    def __call__(self, A, Y, rng=np.random, E=None):
        N = A.shape[1]   # number of neurons
        D = Y.shape[1]   # number of dimensions

        dec = []
        rmses = []
        for i in range(D):
            d = cvxpy.Variable(N)
            loss = cvxpy.sum_squares(A * d - Y[:, i])
            if self.reg > 0:
                loss = loss + cvxpy.sum_squares(d) * self.reg
            cvx_prob = cvxpy.Problem(cvxpy.Minimize(loss),
                                     [self.lower <= d, d <= self.upper]
                                     )
            cvx_prob.solve()
            decoder = d.value.flatten()
            rmse = np.sqrt(np.mean((Y[:, i] - np.dot(A, decoder))**2))
            dec.append(decoder)
            rmses.append(rmse)
        return np.array(dec).T, dict(rmses=np.array(rmses))
