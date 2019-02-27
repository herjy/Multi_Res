import numpy as np

import matplotlib.pyplot as plt


def gauss(x, x0=0, s=1):
    return np.exp(-(x-x0)**2/(2*s**2)) / (np.sqrt(2*np.pi)) / s

def gauss_binned(x, x0=0, s=1, oversample=100):
    # Takes samples as averages over a range [x-h/1:x+h/2]
    # x = sample locations
    h = x[1]-x[0]
    x_, x__ = x - h/2, x + h/2
    return [gauss(np.linspace(x_[i], x__[i], oversample), x0=x0, s=s).mean() for i in range(len(x))]

def shannon_interp(X, xb, yb):
    #Interpolation of function with samples yb at locations xb into samples at positions X
    # yb: Samples
    # xb: sample location
    # X: interpolation location
    h = xb[1] - xb[0]
    return np.array([yb[k] * np.sinc((X-xb[k])/h) for k in range(len(xb))]).sum(axis=0)

def convolve(X, f, xf, p, xp):
    #Convolution operator
    #xf and xp have the same sampling rate
    # f, the function to sample
    # xf, its sampling points
    # p, the function for the psf
    # xp, its sampling
    # X the low resolution sampling of the convolved profile
    assert xf[1] - xf[0] == xp[1] - xp[0]
    h = xf[1] - xf[0]
    return h * np.array([[f[k] * p[l] * np.sinc((X-xf[k]-xp[l])/h) for l in range(len(xp))] for k in range(len(xf))]).sum(axis=(0,1)) # sum over k and l, not X

def diff_kernel(X, f, xf, f_, xf_, alpha=1e-8):
    # construct diff kernel and inverts resolution problem:
    # what are the optimal kernel values that (when using the convolution formula)
    # yield convolved f_ at locations xf_
    # X: Low resolution samples
    # f: intrinsic function
    # xf: high resolution sampling

    assert xf[1] - xf[0] == X[1] - X[0]
    h = xf[1] - xf[0]

    # sampling in k and l indentical, but grid of diff kernel is set by X
    B = h * np.array([[f[k] * np.sinc((X-xf[k]-X[l])/h) for l in range(len(X))] for k in range(len(xf))]).sum(axis=0).T # sum over k

    c = shannon_interp(X, xf_, f_)
    # direct inverse is not numerically stable...
    # direct = np.dot(np.linalg.inv(np.dot(B.T, B)), np.dot(B.T, c))

    # stabilized inverse
    ridge = np.dot(np.linalg.inv(np.dot(B.T, B) + alpha*np.eye(len(x))), np.dot(B.T, c))
    return ridge / ridge.sum()

    """
    # SVD based inverse
    # from http://www.lx.it.pt/~bioucas/IP/files/introduction.pdf, slides 17ff
    u,s,v = np.linalg.svd(B)
    direct_svd = np.dot(np.dot(u,1./s[:,None]*v), c)
    tfd_k = np.argmin(s > alpha)
    tfd = np.dot(np.dot(u[:,:tfd_k],1./s[:tfd_k,None]*v[:tfd_k,:]), c)
    wiener = np.dot(np.dot(u,s/(s**2 + alpha)[:,None] * v), c)
    """

x_,x__ = -100,100
samples = 201
oversample = 100
#Sampling
x = np.linspace(x_,x__,samples)
#Oversampling
X = np.linspace(x_,x__,samples*oversample)

# small Gaussian1
s1 = 2.3
#Oversampled gaussian by a factor 100
Y1 = gauss(X, s=s1)
x1 = np.linspace(-10,10,21) # narrower grid for speed, but same resolution as x
#Low resolution gaussian
yb1 = gauss_binned(x1, s=s1, oversample=oversample)

#Interpolation on grid X from low sampling x1
Ys1 = shannon_interp(X,x1,yb1)
plt.figure()
plt.plot(x1, yb1, 'b-', drawstyle='steps-mid', label='Low resolution')
plt.plot(X, Ys1, 'r-', label='Interpolated')
plt.plot(X, Y1, 'g-', label='Interpolated')
plt.plot(X, Y1-Ys1, 'g-', label='Interpolated')
plt.show()

"""
# convolve with another Gauss, only use the pixel convolution once
s2 = 4
yb11 = gauss_binned(x, s=np.sqrt(s1**2 + s2**2), oversample=oversample)
#plt.plot(x, yb11, 'bo')
plt.plot(x, yb11, 'b-', drawstyle='steps-mid')
yb11_ = convolve(x, gauss(x, s=s2), x, yb1, x)
plt.plot(x, yb11_, 'c-', lw=0.5, drawstyle='steps-mid')
plt.show()
"""

# larger Gaussian2 on larger and shifted grid
s2 = 8
h = 4.89
dx2 = 0.71
#samples at gap h
x2 = x_ + h*np.arange(int(np.ceil((x__-x_)/h)))

#Gaussian oversampled, on a shifted grid
Y2 = gauss(X + dx2, s=s2)

#Subsample by a factor 4.89 at shifted positions
yb2 = gauss_binned(x2 + dx2, s=s2, oversample=oversample)

#Interpolation at high resolution on the non-shifted grid
Ys2 = shannon_interp(X,x2,yb2)
plt.plot(x2, yb2, 'b-', drawstyle='steps-mid', lw=3, label='LR')
plt.plot(X, Ys2, 'r-', label='HR resampling')
plt.show()
# get diff_kernel: on resolution of Gaussian1
dp = diff_kernel(x, yb1, x1, yb2, x2)
plt.plot(x,dp, 'k--', label='Pd = G2 \ G1')

# convolve Gaussian1 with dp (at resolution 1), evaluate at sample position of 2
# should give binned Gaussian2
yb3 = convolve(x2, yb1, x1, dp, x)
plt.plot(x2,yb3, 'g-', drawstyle='steps-mid', label='G1 * Pd')
plt.legend(loc='upper right', frameon=False)
plt.show()
