from pylab import shape

from pylab import shape
from pylab import zeros
from pylab import sys
from pylab import squeeze

from numpy import asarray
from numpy import matrix

from scipy import weave
from scipy.weave import converters 

def tc(X,A,lbda,adapt,eta,iter,soft):
    L,batch_size = shape(X)
    M = shape(A)[1]

    a = zeros([M,batch_size],'d')
    u = zeros([M,batch_size],'d')

    C = asarray(matrix(A).T * matrix(A))
    for i in range(M): C[i,i] = 0

    for n in range(batch_size):
        print '\r', n,
        sys.stdout.flush()

        b = squeeze(A.T * X[:,n])
        thresh = abs(b).max()

        a[:,n], u[:,n] = tc_fast(C,b,lbda,adapt,eta,iter,soft,thresh)
    print '\r',
    
    return (a,u)

def tc_fast(C,b,lbda,adapt,eta,iter,soft,thresh):
    M = shape(C)[0]
    C = asarray(C)
    b = asarray(b)
    Ca = zeros(M,'d')
    u = zeros(M,'d')
    a = zeros(M,'d')

    thresh = float(thresh)
    lbda = float(lbda)

    code = """
           int i,j,k;
           double *Ci;

           // Initialize u and a
           for (i = 0; i < M; i++) { u[i] = 0; a[i] = 0; }

           for (k = 0; k < iter; k++) {
               // compute C * a
               for (i = 0; i < M; i++) { Ca[i] = 0; }
               for (i = 0; i < M; i++) {
                   if (a[i] != 0) {
                       Ci = C + i*M;
                       for (j = 0; j < M; j++) { Ca[j] += a[i] * Ci[j]; }
                   }
               }

               // update u and a
               for (i = 0; i < M; i++) {
                   u[i] = eta * (b[i] - Ca[i]) + (1 - eta) * u[i]; 
                   if ((u[i] < thresh)&(u[i] > -thresh)) { a[i] = 0; }  
                   else if (soft == 1) {
                       if (u[i] > 0) { a[i] = u[i] - thresh; }
                       else  { a[i] = u[i] + thresh; }      
                   } else { a[i] = u[i]; }
               }

               // update the theshold
               if (thresh > lbda) { thresh = adapt * thresh; }
           }

           return_val = 0;

           """
    vars = ['M','C','b','Ca','u','a','lbda','adapt','eta',
            'iter','soft','thresh']
    val = weave.inline(code, vars)
    return (a,u)
