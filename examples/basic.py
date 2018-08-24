import nengo
import nengo_cvxsolver

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=100, dimensions=1)
    b = nengo.Ensemble(n_neurons=100, dimensions=1)

    conn = nengo.Connection(a, b, solver=nengo_cvxsolver.CVXBoundedSolver())

sim = nengo.Simulator(model)
dec = nengo.data[conn].weights
print(dec)

