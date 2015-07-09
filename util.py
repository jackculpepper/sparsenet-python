

from pylab import sqrt
from pylab import floor
from pylab import shape
from pylab import reshape
from pylab import zeros
from pylab import rand

from numpy import matrix


def choose_patches(IMAGES, L, batch_size=1000):
    sz = int(sqrt(L))
    imsz = shape(IMAGES)[0]
    num_images = shape(IMAGES)[2]
    BUFF = 4

    X = matrix(zeros([L,batch_size],'d'))
    for i in range(batch_size):
        j = int(floor(num_images * rand()))
        r = sz/2+BUFF+int(floor((imsz-sz-2*BUFF)*rand()))
        c = sz/2+BUFF+int(floor((imsz-sz-2*BUFF)*rand()))
        X[:,i] = reshape(IMAGES[r-sz/2:r+sz/2,c-sz/2:c+sz/2,j],[L,1])
    return X
