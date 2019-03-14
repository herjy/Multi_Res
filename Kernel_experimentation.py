import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as scp

x = np.linspace(-10,10,1000)

def Spline(x):
    return 1./12.*(np.abs(x-2)**3-4*np.abs(x-1)**3+6*np.abs(x)**3-4*np.abs(x+1)**3+np.abs(x+2)**3)
freq = np.fft.fftfreq(x.shape[-1])
Splinefft = np.fft.fft(Spline(x)/np.max(Spline(x)))
Sinc4fft = np.fft.fft(np.sinc(x)**4)

plt.plot(x, np.sinc(x), 'r')
plt.plot(x, np.exp(-x**2),'g')
plt.plot(x, np.sinc(x)**4, 'b')
plt.plot(x, Spline(x)/np.max(Spline(x)), 'c')
plt.plot(freq,np.abs(Sinc4fft), 'm')
plt.show()