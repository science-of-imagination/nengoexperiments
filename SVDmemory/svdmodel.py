__author__ = 'sterling, king of modelling'


from nengoutils.data import load_mini_mnist
import nengoutils.visualize
import nengoutils.collect
import numpy as np
from nengoutils import SVDCompressor as SVD
import nengo

#get images to put in memory (one line below# )
#print load_mini_mnist()[0:2][0]

images = np.array([np.array(image).flatten() for image in load_mini_mnist('train')])

#compress the images. storing the basis
compressor = SVD(images, 100)

anImage = compressor.compress(images[0])
dims = anImage.shape[0]

def stim_func(t):
    if t<100:
        return anImage
    else:
        return np.zeros(dims)

#import matplotlib.pyplot as plt
#plt.imshow(np.reshape(compressor.decompress(anImage), (28,28)), cmap='gray')
#plt.show()

def model(n1, n2, d1, d2):

    with nengo.Network() as net:
        zero = nengo.Node(stim_func)
        one = nengo.Ensemble(n1, d1, neuron_type=nengo.Direct())
        two = nengo.Ensemble(n2, d2)#, neuron_type=nengo.Direct())
        probe = nengo.Probe(two)

        nengo.Connection(zero, one, synapse=None)
        nengo.Connection(one, two, function=compressor.decompress,
                         synapse=0.05)
        nengo.Connection(two, two, synapse=0.05)

    return net, probe

net, probe = model(2000, 2000, dims, 784)
sim = nengo.Simulator(net)
sim.run(0.5)

data = nengoutils.collect.Data(data=sim.data[probe], dims=(28, 28))
nengoutils.visualize.mk_imgs('./images/', data)