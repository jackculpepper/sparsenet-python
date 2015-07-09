

import pickle

from pylab import shape
from pylab import reshape
from pylab import rand
from pylab import eye
from pylab import sqrt
from pylab import squeeze
from pylab import asarray
from pylab import figure
from pylab import imshow
from pylab import transpose
from pylab import cm
from pylab import clf
from pylab import bar
from pylab import arange
from pylab import show

from numpy import matrix

from time import time

#import minimize

from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_ncg
from scipy.optimize import fmin_cg

from vis import render_network
from thresholding_circuit import tc
from util import choose_patches

images_filename = "../data/IMAGES.p"

class SparseNet:

    """
    An implementation of sparsenet using the thresholding circuit and
    conjugate gradient learning with weight decay.
    """

    def __init__(self, A=None, IMAGES=None, gamma=0.01, lbda=0.3, eta=0.1,
                 adapt=0.96, iter=150, soft=1, display_every=1, run_gd=0,
                 run_cg=1, eta_gd=0.01, cg_maxiter=20, cg_epsilon=0.00000001,
                 cg_gtol=0.0001, L=64, M=64):
        if (A == None):
            # init basis functions, normalize
            self.A = matrix(rand(L,M)-0.5)
            self.A *= matrix(eye(M)*1/sqrt(sum(self.A.A**2)))
        else:
            self.A = A

        if (IMAGES == None):
            self.IMAGES = pickle.load(open(images_filename))
        else:
            self.IMAGES = IMAGES

        self.update = 0

        self.run_gd = run_gd
        self.run_cg = run_cg

        # gd params
        self.eta_gd = eta_gd

        # cg params
        self.cg_maxiter = cg_maxiter
        self.cg_epsilon = cg_epsilon
        self.cg_gtol = cg_gtol

        self.gamma = gamma

        self.display_every = display_every
        self.display_bf = 1
        self.display_coef = 0
        self.display_norm = 1
        self.display_recon = 0

        # threshold circuit params
        self.lbda = float(lbda)
        self.eta = eta
        self.adapt = adapt
        self.iter = iter
        self.soft = soft



    def run(self, batch_size=1000, num_trials=10):
        L, M = shape(self.A)
        display_every = self.display_every
        sz = int(sqrt(L))

        for t in range(num_trials):
            tictime = time()
            X = choose_patches(self.IMAGES, L, batch_size)
            time_ch = time() - tictime

            tictime = time()
            S,U = tc(X, self.A, self.lbda, self.adapt, self.eta,
                     self.iter, self.soft)
            S = matrix(S)
            time_tc = time() - tictime

            tictime = time()
            args = (S, X, self.gamma)

            if (self.run_gd):
                x0 = squeeze(asarray(reshape(self.A,[L*M,1])))
                obj0 = f_l2_wd(x0, S, X, self.gamma)
                E = X - self.A*S
                dA = E*S.T/batch_size
                A1 = self.A + self.eta_gd * dA
                x1 = squeeze(asarray(reshape(A1,[L*M,1])))
                obj1 = f_l2_wd(x1, S, X, self.gamma)

            if (self.run_cg):
                x0 = squeeze(asarray(reshape(self.A,[L*M,1])))
                obj0 = f_l2_wd(x0, S, X, self.gamma)
#                x1 = fmin_ncg(f_l2_wd, x0, g_l2_wd,
#                              None, None, args, maxiter=self.cg_maxiter,
#                              epsilon=self.cg_epsilon)
#                x1 = fmin_cg(f_l2_wd, x0, g_l2_wd,
#                             args, maxiter=self.cg_maxiter,
#                             epsilon=self.cg_epsilon, gtol=self.cg_gtol,
#                             norm=2)
                x1 = fmin_cg(f_l2_wd, x0, g_l2_wd,
                             args, maxiter=self.cg_maxiter)

#                x1,fX,i = minimize.f_min(x0, objfun_l2_wd, 5, args)

                obj1 = f_l2_wd(x1, S, X, self.gamma)
                A1 = matrix(reshape(x1,[L,M]))

            time_ud = time() - tictime

            tictime = time()
            if ( display_every == 1 or (self.update % display_every == 0)):
                if ( self.display_bf ):
                    array = render_network(self.A)
                    figure(1)
                    imshow(transpose(array),cm.gray,interpolation='nearest')

                if ( self.display_recon ):
                    Ihat = matrix(self.A)*S
                    figure(2)
                    subplot(1,2,1)
                    imshow(reshape(Ihat[:,0],[sz,sz]),cm.gray,
                           interpolation='nearest')
                    subplot(1,2,2)
                    imshow(reshape(X[:,0],[sz,sz]),cm.gray,
                           interpolation='nearest')

                if ( self.display_norm ):
                    normA = sum(A1.A**2)
                    figure(3)
                    bar(arange(M),normA,0.1,color='b',hold=False)

            time_dy = time() - tictime

            # normalize to unit length
            A1 = A1*matrix(eye(M)*1/sqrt(sum(A1.A**2)))
            self.A = asarray(A1)
	
            print "update", self.update, "ch", '%.4f'% time_ch,
            print "if", '%.4f'% time_tc, "ud", '%.4f'% time_ud,
            print "dy", '%.4f'% time_dy, "o0", '%.4f'% obj0,
            print "o1", '%.4f'% obj1

            self.update += 1

def objfun_l2_wd(x0, S, I, gamma):
    M = shape(S)[0]
    L, batch_size = shape(I)
    A = matrix(reshape(x0,[L,M]))
    E = I - A*S 
    f = 0.5*(E.A**2).sum()/batch_size + 0.5*gamma*(A.A**2).sum()
    g = -E*S.T/batch_size + gamma*A   
    return (f,g.A1)


def f_l2_wd(x0, S, I, gamma):
    M = shape(S)[0]
    L, batch_size = shape(I)
    A = matrix(reshape(x0,[L,M]))
    E = I - A*S
    f = 0.5*(E.A**2).sum()/batch_size + 0.5*gamma*(A.A**2).sum()
    return f

def g_l2_wd(x0, S, I, gamma):
    M = shape(S)[0]
    L, batch_size = shape(I)
    A = matrix(reshape(x0,[L,M]))
    E = I - A*S
    g = -E*S.T/batch_size + gamma*A
    return g.A1


def check():
    gamma = 0.01
    X = matrix(rand(64,10))
    S = matrix(rand(64,10))
    args = (S, X, gamma)
    x0 = rand(4096,)
    return check_grad(f_l2_wd, g_l2_wd, x0, *args)

def unittest_cg():
    sn = init()
    sn.run_cg = 1

    sn.run(1000,10)
    sn.run(2000,10)
    sn.run(5000,10)
    sn.run(10000,10)

def unittest_gd():
    sn = init()
    sn.run_gd = 1
    sn.display_every = 10000

    sn.run(1,1000000)

def unittest_gd_large():
    sn = init(L=64, M=1024)
    sn.run_gd = 1
    sn.display_every = 10000

    sn.run(1,10000000)

def unittest_cg_large():
    sn = init(L=64, M=1024)
    sn.run_cg = 1
    sn.run(1000,20)
    sn.run(2000,20)
    sn.run(5000,20)
    sn.run(10000,20)







