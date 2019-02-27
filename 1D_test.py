import matplotlib.pyplot as plt
import numpy as np
import Wavelet1d as wv
import pyfits as pf
import warnings
warnings.simplefilter("ignore")


def linormS(S, nit):
    """
      Estimates the maximal eigen value of a matrix A

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: maximal eigen value

       EXAMPLES

    """

    ns, nb = np.shape(S)
    x0 = np.random.rand(ns)
    x0 = x0 / np.sqrt(np.sum(x0 ** 2))

    for i in range(nit):
        x = np.dot(x0, S)
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = np.dot(xp, S.T)
        yn = np.sqrt(np.sum(y ** 2))
        if yn < np.dot(y.T, x0):
            break
        x0 = y / yn

    return 1./xn


def linormA(A, S, nit):
    """
      Estimates the maximal eigen value of a matrix A

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: maximal eigen value

       EXAMPLES

    """

    ns, nb = np.shape(A)
    n1, n2 = np.shape(S)
    x0 = np.random.rand(nb, n1)
    x0 = x0 / np.sqrt(np.sum(x0 ** 2))

    for i in range(nit):
        x = np.dot(np.dot(A, x0), S)
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = np.dot(np.dot(A.T, xp), S.T)
        yn = np.sqrt(np.sum(y ** 2))
        if yn < np.dot(y.T, x0):
            break
        x0 = y / yn

    return 1./xn


def G(a, sigma):
    G = 1./(sigma*np.sqrt(2*np.pi))*np.exp(-(a**2.)/(2*sigma**2))
    return G


def s1(a):
    # /G(0., 25))
    return G(a+40, 5)/G(0., 5)+ G(a, 2)/G(0, 2.)+ \
        G(a, 25)/G(0, 25.)+G(a-60, 5) / G(0., 5)+G(a+70, 5) / G(0., 5)


def s2(a):
    return 3*(G(a-10, 10)/G(0., 10)-G(a+20, 5) / G(0., 5)+G(a+20, 15) / G(0., 15))


x = np.linspace(-100, 100, 201)
X = np.linspace(-100, 100, 31)

plt.plot(s1(x), 'r')
plt.plot(s2(x), 'b')
plt.show()


def p1(a):#LR
    Res = G(a, 15)
    if np.size(a) > 1:
            Res[np.abs(a) > 100] = 0

    return Res/np.sum(G(x, 15))


def p2(a): #HR
    Res = G(a, 5)
    if np.size(a) > 1:
        Res[np.abs(a) > 100] = 0

    return Res/np.sum(G(x, 5))


plt.plot(p1(x), 'r')
plt.plot(p2(x), 'b')
plt.show()


def f1(a):
    return 0.1*s1(a)+0.55*s2(a)


def f2(a):
    return 0.3*s1(a)+0.4*s2(a)


def f3(a):
    return 0.6*s1(a)+0.05*s2(a)





plt.plot(x, f1(x))
plt.plot(x, f2(x))
plt.plot(x, f3(x))
plt.plot(X, f1(X))
plt.plot(X, f2(X))
plt.plot(X, f3(X))
plt.show()


def interp(a, A, Fm):
    # x: High resolution target grid
    # X: Input grid
    # Fm: Samples at positions X

    h=np.abs(A[1]-A[0])
    return np.array([Fm[k] * np.sinc((a-A[k])/h) for k in range(len(A))]).sum(axis=0)

    #X = np.sort(np.random.rand(X.size*2)*np.max(X)*2-np.max(X))


F_interp=interp(x, X, f1(X))
plt.title('Interpolation')
plt.plot(X, f1(X), 'r', drawstyle='steps-mid', label='LR')
plt.plot(x, f1(x), 'b', label='HR')
plt.plot(x, F_interp, 'g', label='Interp')
plt.plot(x, f1(x)-F_interp, 'g--', label='Interp')
plt.show()

    ############################################


def conv(a, xk, xm, p, xp, h):

    # x: numpy array, high resolution sampling
    # X: numpy array, low resolution sampling
    # p: numpy array, psf sampled at high resolution
    # xm: scalar, location of sampling in thelow resolution grid
    Fm=0

    for i, xi in enumerate(xp):
        Fm += p(xi)*np.sinc((xm-xk-xi)/h)

    return Fm   #* h/np.pi


def make_vec(a, xm, p, xp, h):

    vec=np.zeros(a.size)
    for k, xk in enumerate(a):
        vec[k]=conv(a, xk, xm, p, xp, h)

    return vec


def make_filter(a, A, p, xp):
    ## Low resolution sampling
    H=A[1]-A[0]
    h=a[1]-a[0]
    xm0=A[len(A)/2]
    m=np.min(a)
    M=np.max(a)
    n=len(a)
    #Extension of the sampling to account for sinc edges
    xx=np.linspace(m-n, M+n, 3*n+1)


    return make_vec(xx, xm0, p, xx, H)*(1-np.float(len(a))/len(xx))



def make_mat_alt(a,A,p,xp):
    vec = make_filter(a, A, p, xp)
    mat = np.zeros((a.size, A.size))
    n = len(a)
    h =A[1]-A[0]

    for k in range(A.size):

        mat[:,k] = vec[n+n/2-k*h : 2*n+n/2-k*h]/h
    return mat

def make_mat(a, A, p, xp):
    mat=np.zeros((a.size, A.size))
    H=np.abs(a[1]-a[0])
    for m, xm in enumerate(A):
        mat[:, m]=make_vec(a, xm, p, xp, H)
    return mat

if 1:
    mat_LR=make_mat_alt(x,X,p1,x)#make_mat(x, X, p1, x)
    mat_HR=make_mat_alt(x,x,p2,x)#make_mat(x, x, p2, x)
#

#
    hdus=pf.PrimaryHDU(mat_HR)
    lists=pf.HDUList([hdus])
    lists.writeto('mat_HR.fits', clobber=True)
    hdus=pf.PrimaryHDU(mat_LR)
    lists=pf.HDUList([hdus])
    lists.writeto('mat_LR.fits', clobber=True)
#


mat_LR0 = pf.open('mat_LR.fits')[0].data
mat_HR0 = pf.open('mat_HR.fits')[0].data
plt.subplot(131)
plt.imshow(mat_LR0); plt.colorbar()
plt.subplot(132)
plt.imshow(mat_LR); plt.colorbar()
plt.subplot(133)
plt.imshow(mat_LR0-mat_LR); plt.colorbar()
plt.show()
plt.subplot(131)
plt.imshow(mat_HR0); plt.colorbar()
plt.subplot(132)
plt.imshow(mat_HR); plt.colorbar()
plt.subplot(133)
plt.imshow(mat_HR0-mat_HR); plt.colorbar()
plt.show()

Y_LR=np.zeros((3, X.size))
Y_LR[0, :]=np.dot(f1(x), mat_LR)
Y_LR[1, :]=np.dot(f2(x), mat_LR)
Y_LR[2, :]=np.dot(f3(x), mat_LR)

Y_HR=np.zeros((3, x.size))
Y_HR[0, :]=np.dot(f1(x), mat_HR)
Y_HR[1, :]=np.dot(f2(x), mat_HR)
Y_HR[2, :]=np.dot(f3(x), mat_HR)

hdus=pf.PrimaryHDU(Y_HR)
lists=pf.HDUList([hdus])
lists.writeto('Y_HR.fits', clobber=True)
hdus=pf.PrimaryHDU(Y_LR)
lists=pf.HDUList([hdus])
lists.writeto('Y_LR.fits', clobber=True)

A=np.zeros((3, 2))
A[0, :]=0.1, 0.55
A[1, :]=0.3, 0.4
A[2, : ]=0.6, 0.05


Y_LR = pf.open('Y_LR.fits')[0].data
Y_HR = pf.open('Y_HR.fits')[0].data

sigma_HR = np.max(Y_HR)/20
sigma_LR = np.max(Y_LR)/200

var_norm = 1./sigma_HR**2 + 1./sigma_LR**2
wvar_HR = (1./sigma_HR**2)*(1./var_norm)
wvar_LR = (1./sigma_LR**2)*(1./var_norm)


#muLR = linormA(A, mat_LR, 10)
#muHR = linormA(A, mat_HR, 10)
muLR=linormS(mat_LR, 10)
muHR=linormS(mat_HR, 10)

plt.plot(x, Y_HR[2,:]+ np.random.randn(x.size)*sigma_HR, 'b')
plt.plot(X, Y_LR[2,:]+np.random.randn(X.size)*sigma_LR, 'r', drawstyle='steps-mid')
plt.plot(x, 0.6*s1(x)+0.05*s2(x), 'g--')
plt.show()

N = 100
SALLs = np.zeros((N,x.size))
SHRs = np.zeros((N,x.size))
SLRs = np.zeros((N,x.size))

for j in range(N):
    print(j)
    Y_LRs = Y_LR + np.random.randn(3,X.size)*sigma_LR
    Y_HRs = Y_HR + np.random.randn(3,x.size)*sigma_HR

    SALL=np.random.randn(1, x.size)*0
    SLR=np.random.randn(1, x.size)*0
    SHR=np.random.randn(1, x.size)*0

    for i in range(5000):

        SALL +=  muLR * np.dot(Y_LRs[2,:] - np.dot(SALL, mat_LR), mat_LR.T)*wvar_LR + \
            muHR * np.dot( Y_HRs[2,:]-np.dot(SALL, mat_HR), mat_HR.T)*wvar_HR
        SLR +=  muLR * np.dot(Y_LRs[2,:] - np.dot(SLR, mat_LR), mat_LR.T)#*wvar_LR
        if i<1000:
            SHR +=  muHR * np.dot(Y_HRs[2,:] - np.dot(SHR, mat_HR), mat_HR.T)#*wvar_HR

#    S = wv.mr_filter(S, 20, 5, wv.MAD(S),lvl = 6)

        SALL[SALL<0]=0
        SHR[SHR<0]=0
        SLR[SLR<0]=0


    SALLs[j,:] = SALL
    SHRs[j,:] = SHR
    SLRs[j,:] = SLR

Mall = np.mean(SALLs, axis = 0)
MHR = np.mean(SHRs, axis = 0)
MLR = np.mean(SLRs, axis = 0)
Sall = np.std(SALLs, axis = 0)
SHR = np.std(SHRs, axis = 0)
SLR = np.std(SLRs, axis = 0)
print(Mall.shape, Sall.shape)

plt.plot(x, Mall, 'r', label = '${S_{ALL}}$')
plt.plot(x, MHR, 'g', label = '${S_{HR}}$')
plt.plot(x, MLR, 'b', label = '${S_{LR}}$')
plt.plot(x, Mall+3*Sall, 'm', label = '${S_{ALL}}$')
plt.plot(x, MHR+3*SHR, 'y', label = '${S_{HR}}$')
plt.plot(x, MLR+3*SLR, 'c', label = '${S_{LR}}$')
plt.plot(x, Mall-3*Sall, 'm')
plt.plot(x, MHR-3*SHR, 'y')
plt.plot(x, MLR-3*SLR, 'c')
plt.plot(x, 0.6*s1(x)+0.05*s2(x), 'k--', label = '$S$ true')
#plt.plot(X, 0.6*s1(X)+0.05*s2(X), 'k--', label = '$S$ true', drawstyle='steps-mid')
#plt.plot(X, Y_LR[2,:], 'b--', label = '$S$ obs', drawstyle='steps-mid')
#plt.plot(x, Y_HR[2,:], 'c--', label = '$S$ obs')
plt.legend()
plt.title('Sources')
plt.show()

plt.plot(x, SALL[0,:]-0.6*s1(x)-0.05*s2(x), 'r')
plt.title('S_truth-tilde{S}')
plt.show()

plt.title('Residuals')
plt.plot(X, (Y_LR[2,:] -  np.dot(SALL, mat_LR))[0, :], drawstyle='steps-mid')
plt.plot(x, (Y_HR[2,:] -  np.dot(SALL, mat_HR))[0, :])
plt.show()

print('Now lets solve this!')
Y_LR = Y_LR + np.random.randn(3,X.size)*sigma_LR
Y_HR = Y_HR + np.random.randn(3,x.size)*sigma_HR

plt.plot(X, Y_LR.T, 'r', drawstyle='steps-mid')
plt.plot(x, Y_HR.T, 'b')
plt.plot(0,0,'r', drawstyle='steps-mid', label = 'Low Resolution')
plt.plot(0,0,'b', label = 'High Resolution')
plt.plot(x, f1(x), 'g--', label = 'HR no PSF')
plt.plot(x, f2(x), 'g--')
plt.plot(x, f3(x), 'g--')
plt.legend()
plt.title('mixtures')
plt.show()

niter=10
subiter = 500

As = np.random.randn(3,2)
As /= np.sum(As, axis=0)
muA1 = 0.0051
muA2 = 0.0051
S=np.random.randn(2, x.size)*0
for i in range(niter):
    print(i)
    for j in range(subiter):
        S +=  muLR * np.dot(np.dot(A.T, Y_LR - np.dot(A, np.dot(S, mat_LR))), mat_LR.T)*wvar_LR + \
            muHR * np.dot(np.dot(A.T, Y_HR-np.dot(A, np.dot(S, mat_HR))), mat_HR.T)*wvar_HR

        S[S<0] = 0
    As += muA1 * np.dot(np.dot(Y_LR - np.dot(As, np.dot(S, mat_LR)), mat_LR.T), S.T)*wvar_LR + \
        muA2 * np.dot(np.dot(Y_HR - np.dot(As, np.dot(S, mat_HR)), mat_HR.T), S.T)*wvar_HR
    As[As < 0] = 0
    As /= np.sum(As, axis=0)
    #    plt.plot(S.T)
    #    plt.show()

print('Result for A:', A, As)

plt.plot(x, S[0, :], 'r', label = 'tilde{S_1}')
plt.plot(x, s1(x), 'm', label = 'S_1')
plt.plot(x, S[1, :], 'b', label = 'tilde{S_2}')
plt.plot(x, s2(x), 'c', label = 'S_2')
plt.legend()
plt.title('Sources')
plt.show()

plt.plot(x, S[0, :]-s1(x), 'r')

plt.plot(x, S[1, :]-s2(x), 'b')
plt.title('S_truth-tilde{S}')
plt.show()

plt.title('Residuals')
plt.plot(X, (Y_LR - np.dot(A, np.dot(S, mat_LR)))[0, :], drawstyle='steps-mid')
plt.plot(x, (Y_HR - np.dot(A, np.dot(S, mat_HR)))[-1, :])
plt.show()
