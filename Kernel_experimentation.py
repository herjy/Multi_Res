import numpy as np
import matplotlib.pyplot as plt
import scarlet
import scipy.ndimage as scp



def Spline(x):
    return 1./12.*(np.abs(x-2)**3-4*np.abs(x-1)**3+6*np.abs(x)**3-4*np.abs(x+1)**3+np.abs(x+2)**3)
x = np.linspace(0,10,1000)
freq = np.fft.fftfreq(x.shape[-1])
f = np.sinc(x/10)**8
ffft = np.fft.fft(f)
plt.plot(x,f)
plt.title('function')
plt.show()

plt.plot(freq,np.abs(ffft)**2)
plt.title('fft')
plt.show()

def filter3(y, N, nu):
    assert N<nu
    K = -1./(nu*N**2)
    K2 = -1./((N-nu)**2*nu)
    f = np.copy(y)
    f[y<N] = K*y[y<N]**3+1
    f[y>=N] = K2*(y[y>=N]-nu)**3
    f[y>nu] = 0
    return f

def filtern(y, N, nu, n1,n2,C):
    assert N<nu
    K = n2/n1 * C/(N**n1*(1-n2/n1)-nu*N**(n1-1))
    K2 = C/((N-nu)**(n2-1)*(N*(1-n2/n1)-nu))
    f = np.copy(y)
    f[y<N] = K*y[y<N]**n1+C
    f[y>=N] = K2*(y[y>=N]-nu)**n2
    f[y>nu] = 0
    return f

n1 = 7
n2 = 7
nu = 2.
C = 3.

plt.subplot(121)
plt.title('filters')
plt.plot(x,filtern(x,0.2,nu, n1,n2,C), label = 'N = 0.2')
plt.plot(x,filtern(x,0.4,nu, n1,n2, C), label = 'N = 0.4')
plt.plot(x,filtern(x,0.5,nu, n1,n2, C), label = 'N = 0.5')
plt.plot(x,filtern(x,0.8,nu, n1,n2, C), label = 'N = 0.8')
plt.plot(x,filtern(x,1.2,nu, n1,n2, C), label = 'N = 1.2')
plt.plot(x,filtern(x,1.5,nu, n1,n2, C), label = 'N = 1.5')
plt.legend()


plt.subplot(122)
plt.title('filters FFT')
plt.plot(freq,np.log(np.fft.ifft(filtern(x,0.2,nu, n1,n2, C)**0.5)), label = 'N = 0.2')
plt.plot(freq,np.log(np.fft.ifft(filtern(x,0.4,nu, n1,n2, C)**0.5)), label = 'N = 0.4')
plt.plot(freq,np.log(np.fft.ifft(filtern(x,0.5,nu, n1,n2, C)**0.5)), label = 'N = 0.5')
plt.plot(freq,np.log(np.fft.ifft(filtern(x,0.8,nu, n1,n2, C)**0.5)), label = 'N = 0.8')
plt.plot(freq,np.log(np.fft.ifft(filtern(x,1.2,nu, n1,n2, C)**0.5)), label = 'N = 1.2')
plt.plot(freq,np.log(np.fft.ifft(filtern(x,1.5,nu, n1,n2, C)**0.5)), label = 'N = 1.5')
plt.legend()

plt.show()