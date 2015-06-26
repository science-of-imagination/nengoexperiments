__author__ = 'sterling, king of modelling'


from nengoutils.data import load_mini_mnist
import numpy as np
from nengoutils import SVDCompressor as SVD
import nengo

#get images to put in memory (one line below# )
#print load_mini_mnist()[0:2][0]

images = np.array([np.array(image).flatten() for image in load_mini_mnist('train')])

#compress the images. storing the basis
compressor = SVD(images, 100)

anImage = compressor.compress(images[0])
print(anImage.shape)

import matplotlib.pyplot as plt
plt.imshow(np.reshape(compressor.decompress(anImage), (28,28)), cmap='gray')
plt.show()
