import numpy as np

import nengo
import nengo_function_space as nfs

domain = np.linspace(-1, 1, 200)


# define kinds of functions you'd like to represent
def gaussian(mag, mean, sd):
    v = mag * np.exp(-(domain)**2/(2*sd**2))
    return np.roll(v, int(mean*100))
    
# build the function space, generate the basis functions
fs = nfs.FunctionSpace(
    nfs.Function(
        gaussian,
        mean=nengo.dists.Uniform(-1, 1),
        sd=0.2,
        mag=1),
    n_basis=10, n_samples=1000)

model = nengo.Network()
with model:

    # create an ensemble to represent the weights over the basis functions
    ens = nengo.Ensemble(n_neurons=2000, dimensions=fs.n_basis+1, radius=1.5,
            neuron_type=nengo.Direct())
    # create combined distribution, which uses both FunctionSpaceDistribution
    # and normal Nengo distributions across different dimensions
    ens.encoders = nfs.Combined([
        fs.project(nfs.Function(gaussian,
                                        mean=nengo.dists.Uniform(-1, 1),
                                        sd=nengo.dists.Uniform(0.2, 0.2),
                                        mag=1)),
        nengo.dists.UniformHypersphere(surface=True)],
        [fs.n_basis, 1],
        weights=[1, 1],
        normalize_weights=True)
    # train over the magnitude in the range 0-1 since no guarantee
    # input will be height 1 TODO: better / more accurate explanation
    ens.eval_points = nfs.Combined([
        fs.project(nfs.Function(gaussian,
                                        mean=nengo.dists.Uniform(-1, 1),
                                        sd=nengo.dists.Uniform(0.2, 0.2),
                                        mag=nengo.dists.Uniform(0, 1))),
        nengo.dists.UniformHypersphere(surface=False)],
        [fs.n_basis, 1],
        weights=[1, 1],
        normalize_weights=True)


    # the function for choosing the most salient stimulus and shifts it
    def collapse(x):
        speed = x[-1]
        x = x[:-1]
        # reconstruct the represented function from weights x
        pts = fs.reconstruct(x)
        # find the max value of the represented function
        peak = np.argmax(pts)
        # create a Gaussian centered at the peak
        data = gaussian(mag=1, sd=0.2, mean=domain[peak])
        # TODO: 50 should be replaced with a gain based on the number
        # of steps in the domain representation, right?
        shift = int(speed*50)
        # multiplied by 1.1 to prevent signal attenuation
        data = fs.project(np.roll(data, shift))*1.1
        return data

    nengo.Connection(ens, ens[:-1], synapse=0.1, function=collapse)

    # input the head movement speed
    speed = nengo.Node([0])
    nengo.Connection(speed, ens[-1])

    # create a node to give a plot of the represented function
    plot = fs.make_plot_node(domain=domain, lines=1, n_pts=50)
    nengo.Connection(ens[:-1], plot[:fs.n_basis], synapse=0.01)
