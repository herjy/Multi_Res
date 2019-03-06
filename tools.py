import numpy as np
import matplotlib.pyplot as plt
import MuSCADeT as wine
import scipy.signal as scp
import SLIT
import time
import scipy.ndimage.filters as med
import warnings
warnings.simplefilter("ignore")


def interp1D(a, A, Fm):
    # x: High resolution target grid
    # X: Input grid
    # Fm: Samples at positions X

    h=np.abs(A[1]-A[0])
    return np.array([Fm[k] * np.sinc((a-A[k])/h) for k in range(len(A))]).sum(axis=0)


def conv1D(a, xk, xm, p, xp, h):

    # x: numpy array, high resolution sampling
    # X: numpy array, low resolution sampling
    # p: numpy array, psf sampled at high resolution
    # xm: scalar, location of sampling in thelow resolution grid
    Fm=0

    for i, xi in enumerate(xp):
        Fm += p(xi)*np.sinc((xm-xk-xi)/h)

    return Fm   #* h/np.pi


def make_vec1D(a, xm, p, xp, h):

    vec=np.zeros(a.size)
    for k, xk in enumerate(a):
        vec[k]=conv(a, xk, xm, p, xp, h)

    return vec


def make_filter1D(a, A, p, xp):
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



def make_mat_alt1D(a,A,p,xp):
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
########################2D##################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
def Image(f, a, b):
    n1 = np.int(np.sqrt(a.size))
    Img = np.zeros((n1, n1))
    xgrid, ygrid = np.where(np.zeros((n1, n1)) == 0)

    Img[xgrid, ygrid] = f(a, b)
    return Img

def interp2D(a, b, A, B, Fm):
    # x: High resolution target grid
    # X: Input grid
    # Fm: Samples at positions X
    hx = np.abs(A[1]-A[0])
    hy = np.abs(B[np.int(np.sqrt(B.size))+1] - B[0])

    return np.array([Fm[k] * np.sinc((a-A[k])/(hx)) * np.sinc((b-B[k])/(hy)) for k in range(len(A))]).sum(axis=0)


def conv2D(xp, yp, xk, yk, xm, ym, p, xpp, ypp, h):
    # x: numpy array, high resolution sampling
    # X: numpy array, low resolution sampling
    # p: numpy array, psf sampled at high resolution
    # xm: scalar, location of sampling in thelow resolution grid
    Fm = 0

    for i in range(np.size(xp)):
        Fm += np.sinc((xm-(xk+xp[i]))/h)*np.sinc((ym-(yk+yp[i]))/h)*p[xpp[i], ypp[i]]
    return Fm*h/np.pi

def make_vec2D(a, b, xm, ym, p, xp, yp, xpp, ypp, h):
    vec = np.zeros(a.size)

    for k in range(np.size(a)):

        vec[k] = conv2D(xp, yp, a[k], b[k], xm, ym, p, xpp, ypp, h)

    return vec.flatten()

def conv2D_fft(xk, yk, xm, ym, p, h):
    # x: numpy array, high resolution sampling
    # X: numpy array, low resolution sampling
    # p: numpy array, psf sampled at high resolution
    # xm: scalar, location of sampling in thelow resolution grid

    ker = np.zeros((np.int(xk.size**0.5), np.int(yk.size**0.5)))
    x,y = np.where(ker == 0)
    ker[x,y] = np.sinc((xm-(xk))/h)*np.sinc((ym-(yk))/h)

    return scp.fftconvolve(ker, p, mode = 'same')*h/np.pi

def make_vec2D_fft(a, b, xm, ym, p, h):

    vec = conv2D_fft(a, b, xm, ym, p, h)

    return vec.flatten()

def make_mat2D_fft(a, b, A, B, p):
    mat = np.zeros((a.size, B.size))
    h = a[1]-a[0]
    assert h!=0
    t0 = time.clock()
    for m in range(np.size(B)):
            #if (m % 1000+1 == True):
            #    print('Matrix line: ', m, ' out of ', np.size(B))
            #    print('time: ', time.clock()-t0)
            mat[:, m] = make_vec2D_fft(a, b, A[m], B[m], p, h)

    return mat

def make_mat2D(a, b, A, B, p, xp, yp, xpp,ypp):
    mat = np.zeros((a.size, B.size))
    h = a[1]-a[0]
    assert h!=0

    for m in range(np.size(B)):
            #if (m % 100+1 == True):
            #    print('Matrix line: ', m, ' out of ', np.size(B))
            mat[:, m] = make_vec2D(a, b, A[m], B[m], p, xp, yp, xpp, ypp, h)

    return mat

def make_filter2D(a, b, A, B, p, xp, yp, xpp, ypp):
    ## Low resolution sampling
    H=A[1]-A[0]
    h=a[1]-a[0]
    xm0=np.array(A[len(A)/2])
    ym0=np.array(B[len(B)/2])

    ma=np.min(a)
    Ma=np.max(a)
    mb=np.min(b)
    Mb=np.max(b)
    N=len(A)
    n=len(a)
    #Extension of the sampling to account for sinc edges
    xx=np.linspace(ma-n, Ma+n, 3*n+1)
    yy=np.linspace(mb-n, Mb+n, 3*n+1)

    # On essaie, mais je crois que c'est xpp ypp en vrai
    xxp,yyp = np.where(np.zeros(((3*n+1)**0.5,(3*n+1)**0.5))==0)

    return make_vec(xx, yy, xm0, ym0, p, xx, yy, xpp, ypp, h)*(1-np.float(len(a))/len(xx))


def linorm2D(S, nit):
    """
      Estimates the maximal eigen value of a matrix A

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: maximal eigen value

       EXAMPLES

    """

    n1, n2 = np.shape(S)
    x0 = np.random.rand(1, n1)
    x0 = x0 / np.sqrt(np.sum(x0 ** 2))

    for i in range(nit):
        x = np.dot(x0, S)
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = np.dot( xp, S.T)
        yn = np.sqrt(np.sum(y ** 2))

        if yn < np.dot(y, x0.T):
            break
        x0 = y / yn

    return 1./xn

def MAD(x,n=3):
    ##DESCRIPTION:
    ##  Estimates the noise standard deviation from Median Absolute Deviation
    ##
    ##INPUTS:
    ##  -x: a 2D image for which we look for the noise levels.
    ##
    ##OPTIONS:
    ##  -n: size of the median filter. Default is 3.
    ##
    ##OUTPUTS:
    ##  -S: the source light profile.
    ##  -FS: the lensed version of the estimated source light profile
    xw = wine.wave_transform.wave_transform(x, np.int(np.log2(x.shape[0])))[0,:,:]
    meda = med.median_filter(xw,size = (n,n))
    medfil = np.abs(xw-meda)#np.median(x))
    sh = np.shape(xw)
    sigma = 1.48*np.median((medfil))
    return sigma

def get_psf(Field,x,y,n, HST = False):
    assert len(x)==len(y)
    PSF = 0.
    PSF_n = 0
    xy0 = np.int(n/2)
    for i in range(len(x)):
        Star = Field[np.int(x[i]-xy0):np.int(x[i]+xy0+1),np.int(y[i]-xy0):np.int(y[i]+xy0+1)]

        xm,ym = np.where(Star == np.max(Star))
        xp = x[i]+(xm-xy0)
        yp = y[i]+(ym-xy0)
        Star = Field[np.int(xp-xy0):np.int(xp+xy0+1),np.int(yp-xy0):np.int(yp+xy0+1)]

        sigma = MAD(Star, n=3)
        PSFt = wine.MCA.mr_filter(Star, 20, 5, sigma, lvl = np.int(np.log2(n)))[0]
        PSF_n += Star/np.sum(Star)
        PSF+=PSFt/np.sum(PSFt)

    if HST == True:
        Res = PSF_n-PSF
        n1,n2 = Res.shape
        Sup = np.copy(Res)*0
        x0,y0 = n1/2, n2/2
        x,y = np.where(Sup==0)
        r = np.sqrt((x-x0)**2+(y-y0)**2).reshape(n1,n2)
        Sup[r<4] = Res[r<4]
        PSF+=Sup
    return PSF/np.sum(PSF)

def Combine2D(HR, LR, matHR, matLR, niter, verbosity = 0):

    n = HR.size
    N = LR.size
    sigma_HR = MAD(HR.reshape(np.int(n**0.5), np.int(n**0.5)))
    sigma_LR = MAD(LR.reshape(np.int(N**0.5), np.int(N**0.5)))

    var_norm = 1./sigma_HR**2 + 1./sigma_LR**2
    wvar_HR = (1./sigma_HR**2)*(1./var_norm)
    wvar_LR = (1./sigma_LR**2)*(1./var_norm)

    mu1 = linorm2D(matHR, 10)/10.
    mu2 = linorm2D(matLR, 10)/10.
    mu = (mu1+mu2)/2.

    Sa = np.zeros((HR.size))
    SH = np.zeros((HR.size))
    SL = np.zeros((HR.size))


    vec = np.zeros(niter)
    vec2 = np.zeros(niter)
    vec3 = np.zeros(niter)
    t0 = time.clock()
    for i in range(niter):
        if (i % 1000+1 == True) and (verbosity == 1):
            print('Current iteration: ', i, ', time: ', time.clock()-t0)
        Sa += mu * np.dot(LR - np.dot(Sa, matLR), matLR.T)*wvar_LR + mu * np.dot(HR-np.dot(Sa, matHR), matHR.T)*wvar_HR
    #plt.imshow(Sall.reshape(n1,n2)); plt.savefig('fig'+str(i))

        SL += mu2 * np.dot(LR - np.dot(SL, matLR), matLR.T)
        if i < niter:
            SH += mu1 * np.dot(HR - np.dot(SH, matHR), matHR.T)
            SH[SH < 0] = 0
        Sa[Sa < 0] = 0
        SL[SL < 0] = 0

        vec[i] = np.std((LR - np.dot(Sa, matLR))**2)/2./sigma_LR+ np.std((HR-np.dot(Sa, matHR))**2*wvar_LR)/2./sigma_HR
        vec2[i] = np.std((LR - np.dot(SL, matLR))**2)/sigma_LR
        vec3[i] = np.std((HR - np.dot(SH, matHR))**2)/sigma_HR
    #    plt.subplot(121)
    #    plt.imshow((LR - np.dot(Sall, matLR)).reshape(N1,N2))
    #    plt.subplot(122)
    #    plt.imshow((HR - np.dot(Sall, matHR)).reshape(n1,n2))
    #    plt.show()
    if verbosity == 1:
        plt.plot(vec, 'r', label = 'All', linewidth = 2)
        plt.plot(vec2, 'g', label = 'LR', linewidth = 3)
        plt.plot(vec3, 'b', label = 'HR', linewidth = 4)
        plt.show()

    return Sa, SH, SL


def linorm2D_filter(filter, filterT, shape, nit):
    """
      Estimates the maximal eigen value of a matrix A

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: maximal eigen value

       EXAMPLES

    """

    n1 = shape
    x0 = np.random.rand(1, n1)
    x0 = x0 / np.sqrt(np.sum(x0 ** 2))

    for i in range(nit):
        x = filter(x0)
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = filterT(xp)
        yn = np.sqrt(np.sum(y ** 2))

        if yn < np.dot(y, x0.T):
            break
        x0 = y / yn

    return 1./xn

def Combine2D_filter(HR, LR, filter_HR, filter_HRT, matLR, niter, verbosity = 0):

    n = HR.size
    n1 = np.int(HR.size**0.5)
    n2 = np.int(HR.size ** 0.5)
    N = LR.size
    sigma_HR = MAD(HR.reshape(np.int(n**0.5), np.int(n**0.5)))
    sigma_LR = MAD(LR.reshape(np.int(N**0.5), np.int(N**0.5)))

    var_norm = 1./sigma_HR**2 + 1./sigma_LR**2
    wvar_HR = (1./sigma_HR**2)*(1./var_norm)
    wvar_LR = (1./sigma_LR**2)*(1./var_norm)

    mu1 = linorm2D_filter(filter_HR, filter_HRT, HR.size, 10)/1.
    mu2 = linorm2D(matLR, 10)/1.
    mu = (mu1+mu2)



    Sa = np.zeros((HR.size))
    SH = np.zeros((HR.size))
    SL = np.zeros((HR.size))
    vec = np.zeros(niter)
    vec2 = np.zeros(niter)
    vec3 = np.zeros(niter)
    t0 =time.clock()
    for i in range(niter):
        if (i % 100+1 == True) and (verbosity == 1):
            print('Current iteration: ', i, ', time: ', time.clock()-t0)
        Sa += mu * np.dot(LR - np.dot(Sa, matLR), matLR.T)*wvar_LR + mu * filter_HRT(HR-filter_HR(Sa.reshape(n1,n2))).reshape(n)*wvar_HR
    #plt.imshow(Sall.reshape(n1,n2)); plt.savefig('fig'+str(i))

        SL += mu2 * np.dot(LR - np.dot(SL, matLR), matLR.T)
        if i < 10000:
            SH += mu1 * filter_HRT(HR-filter_HR(SH.reshape(n1,n2))).reshape(n)
            SH[SH < 0] = 0
        Sa[Sa < 0] = 0
        SL[SL < 0] = 0

        vec[i] = np.std(LR - np.dot(Sa, matLR))**2/2./sigma_LR**2 + np.std(HR-filter_HR(Sa.reshape(n1,n2)))**2*wvar_LR/2./sigma_HR**2
        vec2[i] = np.std(LR - np.dot(SL, matLR))**2/sigma_LR**2
        vec3[i] = np.std(HR - filter_HR(SH.reshape(n1,n2)))**2/sigma_HR**2
    #    plt.subplot(121)
    #    plt.imshow((LR - np.dot(Sall, matLR)).reshape(N1,N2))
    #    plt.subplot(122)
    #    plt.imshow((HR - np.dot(Sall, matHR)).reshape(n1,n2))
    #    plt.show()
    if verbosity == 1:
        plt.plot(vec, 'b', label = 'All', linewidth = 2)
        plt.plot(vec2, 'r', label = 'LR', linewidth = 3)
        plt.plot(vec3, 'g', label = 'HR', linewidth = 4)
        plt.show()

    return Sa, SH, SL

