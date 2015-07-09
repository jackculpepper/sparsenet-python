
from pylab import ones
from pylab import sqrt
from pylab import shape
from pylab import floor
from pylab import reshape

from numpy import asarray


def render_network(A):
    [L,M] = shape(A)
    sz = int(sqrt(L))
    buf = 1
    A = asarray(A)

    if ( floor(sqrt(M)) ** 2 != M ):
        m = int(sqrt(M/2))
        n = M/m
    else:
        m = int(sqrt(M))
        n = m

    array = - ones([buf+m*(sz+buf),buf+n*(sz+buf)],'d')

    k = 0
    for i in range(m):
        for j in range(n):
            clim = max(abs(A[:,k]))
            x_offset = buf+i*(sz+buf)
            y_offset = buf+j*(sz+buf)
            array[x_offset:x_offset+sz,y_offset:y_offset+sz] = reshape(
                A[:,k],[sz,sz])/clim

            k += 1
    return array
