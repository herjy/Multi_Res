import numpy as np
import scipy.signal as cp
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sc
import scipy.ndimage.filters as med


def wavelet(curve, lvl):
    """
    Performs starlet decomposition of an image
    INPUTS:
        img: image with size n1xn2 to be decomposed.
        lvl: number of wavelet levels used in the decomposition.
    OUTPUTS:
        wave: starlet decomposition returned as lvlxn1xn2 cube.
    OPTIONS:
        Filter: if set to 'Bspline', a bicubic spline filter is used (default is True).
        newave: if set to True, the new generation starlet decomposition is used (default is True).
        convol2d: if set, a 2D version of the filter is used (slower, default is 0).

    """
    mode = 'nearest'

    lvl = lvl-1
    sh = np.shape(curve)
    if np.size(sh) ==2:
        mn = np.min(sh)
        mx = np.max(sh)
        wave = np.zeros([lvl+1, mx,mn])
        for h in np.linspace(0,mn-1, mn):
            if mn == sh[0]:
                wave[:,:,h] = wavelet(curve[h,:],lvl+1)
            else:
                wave[:,:,h] = wavelet(curve[:,h],lvl+1)
        return wave
    n1 = curve.size

    h = [1./16, 1./4, 3./8, 1./4, 1./16]
    h = np.array(h)
    n = np.size(h)


    if n+2**(lvl-1)*(n-1) >= n1/2.:
        lvl = np.int_(np.log2((n1-1)/(n-1.))+1)

    c = curve
    ## wavelet set of coefficients.
    wave = np.zeros([lvl+1,n1])

    for i in np.linspace(0,lvl-1,lvl):
        newh = np.zeros((1,n+(n-1)*(2**i-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ######Calculates c(j+1)
        ###### Line convolution

        cnew = sc.convolve1d(c,newh[0,:],axis = 0, mode =mode)

#  hc = sc.convolve1d(cnew,newh[0,:],axis = 0, mode = mode)


            ###### wj+1 = cj-hcj+1
        wave[i,:] = c-cnew#hc


        c = cnew

    wave[i+1,:] = c

    return wave

def iuwt(wave):
    """
    Inverse Starlet transform.
    INPUTS:
        wave: wavelet decomposition of an image.
    OUTPUTS:
        out: image reconstructed from wavelet coefficients
    OPTIONS:
        convol2d:  if set, a 2D version of the filter is used (slower, default is 0)

    """
    mode = 'nearest'

    lvl,n1= np.shape(wave)
    h = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    n = np.size(h)

    cJ = np.copy(wave[lvl-1,:])


    for i in np.linspace(1,lvl-1,lvl-1):

        newh = np.zeros((1,n+(n-1)*(2**(lvl-1-i)-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ###### Line convolution

        cnew = sc.convolve1d(cJ,newh[0,:],axis = 0, mode = mode)
        cJ = cnew+wave[lvl-1-i,:]

    out = cJ
    return out


def MAD(x):
    """
      Estimates noise level in an image from Median Absolute Deviation

      INPUTS:
          x: image

      OUTPUTS:
          sigma: noise standard deviation

      EXAMPLES
    """
    meda = med.median_filter(x,size = (3))
    medfil = np.abs(x-meda)
    sh = np.shape(x)
    sigma = 1.48*np.median((medfil))
    return sigma

def mr_filter(curve, niter, k, sigma,lvl = 6, pos = False, harder = 0, posd = 0, sigi = [0,0]):
    """
      Computes wavelet iterative filtering on an image.

      INPUTS:
          img: image to be filtered
          niter: number of iterations (10 is usually recommended)
          k: threshold level in units of sigma
          sigma: noise standard deviation

      OUTPUTS:
          imnew: filtered image
          wmap: weight map

      OPTIONS:
          lvl: number of wavelet levels used in the decomposition, default is 6.
          pos: if set to True, positivity constrain is applied to the output image
          harder: if set to one, threshold levels are risen. This is used to compensate for correlated noise
          for instance.
          EXAMPLES
    """

    if np.sum(sigi)!=0:
        levels = sigi
        print(levels)
    else:
        levels = np.array([ 0.8907963 ,  0.20066385,  0.08550751,  0.04121745,  0.02042497,
            0.01018976,  0.00504662,  0.00368314])

    n1 = np.size(curve)

    M = np.zeros((lvl,n1))
    M[:,:] = 0
    M[-1,:] = 1

    sh = np.shape(M)
    th = np.ones((lvl,n1))*(k)

    th[0,:] = th[0,0]+1+5*harder


    th =np.multiply(th.T,levels[:sh[0]]).T*sigma
    th[np.where(th<0)] = 0
    th[-1,:] = 0
    imnew = 0
    i =0

    R= curve
    alpha = wavelet(R,lvl).reshape(lvl,n1)

    if pos == True :
         M[np.where(alpha-np.abs(th)> 0)] = 1
    else:

         M[np.where(np.abs(alpha)-np.abs(th)> 0)] = 1


    while i < niter:
        R = curve-imnew

        alpha = wavelet(R,lvl).reshape(lvl,n1)


        Rnew = iuwt(M*alpha)
        imnew = imnew+Rnew

        i = i+1
        if posd == True:
            imnew[np.where(imnew<0)]=0

    return imnew
