import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp

def func(a, sigma, L):
    F = [a]
    F.append(F[-1]+np.random.randn(1)*sigma)
    for i in range(L):
        F.append(F[-1]+np.random.randn(1)*1.1)

    return np.array(F)


def gaussian(x, sigma, x0):
    return 1./np.sqrt(2*sigma)*np.exp(-(x-x0)**2/sigma**2)


l = 1000
F = func(1,0.1,1000)
x = np.linspace(0,10,1000)
G = gaussian(x,0.2,5)
G/=np.sum(G)

f = scp.convolve(F,G,mode = 'same')

t = np.linspace(0,10,f.size)
N1 = 35
N2 = 20
ts1 = np.linspace(0,10,N1)
ts2 = np.linspace(0,10,N2)

sub1 = np.zeros(f.size)
sub2 = np.zeros(f.size)
sub1[np.linspace(0,1000,N1).astype(int)] = f[np.linspace(0,1000,N1).astype(int)]
sub2[np.linspace(0,1000,N2).astype(int)] = f[np.linspace(0,1000,N2).astype(int)]

with plt.xkcd():
    plt.title('Sampling of f')
    plt.plot(t,f, 'g', label = '$f(x)$');
    plt.plot(t,sub1, 'r')
    plt.plot(ts1, f[np.linspace(0, 1000, N1).astype(int)], 'ob', label = '$f(x_k)$')
    plt.legend()
    plt.show()

    plt.plot(ts1, f[np.linspace(0, 1000, N1).astype(int)], 'ob')
    plt.savefig('sampling.png')
    plt.show()

G1 = gaussian(x,0.5,5)
G1 /=np.sum(G1)
fconv1 = scp.convolve(f, G1, mode = 'same')
G2 = gaussian(x,1,5)
G2 /=np.sum(G2)
fconv1 = scp.convolve(f, G1, mode = 'same')
fconv2 = scp.convolve(f, G2, mode = 'same')
subconv1 = np.zeros(f.size)
subconv2 = np.zeros(f.size)
subconv1[np.linspace(0,1000,N1).astype(int)] = fconv1[np.linspace(0,1000,N1).astype(int)]
subconv2[np.linspace(0,1000,N2).astype(int)] = fconv2[np.linspace(0,1000,N2).astype(int)]
with plt.xkcd():
    plt.figure(2)
    plt.title('High resolution')
    plt.plot(t, f, 'g', label = 'f')
    plt.plot(t,fconv1, 'k', label = '$f*p_1$')
    plt.plot(t,subconv1, 'r')
    plt.plot(ts1, fconv1[np.linspace(0, 1000, N1).astype(int)], 'ob', label = '$(f*p_1)(x_m)$')
    plt.legend()

    plt.figure(1)
    plt.title('Low resolution')
    plt.plot(t, f, 'g', label = '$f$')
    plt.plot(t,fconv2, 'k', label = '$f*p_2$')
    plt.plot(t,subconv2, 'r')
    plt.plot(ts2, fconv2[np.linspace(0, 1000, N2).astype(int)], 'ob', label = '$(f*p_2)(x_n)$')
    plt.legend()
    plt.savefig('Function_sampling.png')
    plt.show()