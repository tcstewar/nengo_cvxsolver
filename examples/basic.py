import nengo
import nengo_cvxsolver

model = nengo.Network()
with model:
    a = nengo.Ensemble(n_neurons=100, dimensions=1)
    b = nengo.Ensemble(n_neurons=100, dimensions=1)

    conn = nengo.Connection(a, b, 
        solver=nengo_cvxsolver.CVXBoundedSolver(upper=1, lower=-1))

sim = nengo.Simulator(model)
dec = sim.data[conn].weights
print(dec)

