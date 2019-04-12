#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import tools
import matplotlib.pyplot as plt
import scipy.signal as scp
import tensorflow_hub as hub
import tensorflow as tf

import scarlet
from astropy.wcs import WCS
import astropy.io.fits as fits
import warnings
warnings.simplefilter("ignore")



# # PSF

# In[39]:


PSF_HSC = fits.open('../HSTC/PSF_HSC_52.fits')[0].data
PSF_HST = fits.open('../HSTC/PSF_HST_52.fits')[0].data


# # coordinates

# In[40]:



x0 = 10873  #9150 #7374 #
y0 = 10704 #9633 #10593 #

excess =61

n1,n2 = 100, 100
N1,N2 = 40, 40

x_HST = np.linspace(0, n1-1, n1)
y_HST = np.linspace(0, n2-1, n2)
x_HST, y_HST = np.meshgrid(x_HST, y_HST)
x_HST = x_HST.flatten()
y_HST = y_HST.flatten()

X_HSC = np.linspace(0, N1-1, N1)
Y_HSC = np.linspace(0, N2-1, N2)

X_HST = np.linspace(0, n1-1, N1)
Y_HST = np.linspace(0, n2-1, N2)
X_HST, Y_HST = np.meshgrid(X_HST, Y_HST)
X_HST = X_HST.flatten()
Y_HST = Y_HST.flatten()


# # Computing Low Resolution matrix

# In[41]:


mat_HSC = tools.make_mat2D_fft(x_HST, y_HST, X_HST, Y_HST, PSF_HSC)


# # Images

# In[48]:


Cut_HST = np.zeros((n1,n2))


x, y = np.meshgrid(np.linspace(10,n1-11,n1/10).astype(int),np.linspace(10,n2-11,n2/10).astype(int))

Cut_HST[x, y] = 1

print(Cut_HST.shape, PSF_HST.shape)
Cut_HST = scp.convolve(Cut_HST, PSF_HST, mode = 'same')

Cut_HSC = np.dot(Cut_HST.flatten(), mat_HSC).reshape(N1,N2)

plt.plot(x_HST,y_HST,'or')
plt.plot(X_HST,Y_HST, 'ob')
plt.show()

plt.subplot(121)
plt.imshow(Cut_HST, cmap = 'inferno', interpolation = 'nearest')
plt.colorbar()
plt.subplot(122)
plt.imshow(Cut_HSC, cmap = 'inferno', interpolation = 'nearest')
plt.colorbar()
plt.show()


# # FFT

# In[53]:


F_HST = np.fft.fftshift(np.fft.fftn(Cut_HST))
F_HSC = np.fft.fftshift(np.fft.fftn(Cut_HSC))

plt.subplot(121)
plt.imshow(np.log(np.abs(F_HST)), cmap = 'inferno', interpolation = 'nearest')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log(np.abs(F_HSC)), cmap = 'inferno', interpolation = 'nearest')
plt.colorbar()
plt.show()


# # Filter

# In[ ]:


F_HST = F_HST.reshape(n1*n2,1)
F_HSC = F_HSC.reshape(N1*N2,1)

compute =1
if compute == 1:
    inv = np.linalg.inv(np.dot(F_HST,F_HST.T))
    F = np.dot(F_HSC, np.dot(F_HST.T, inv))


    plt.imshow(np.log(np.abs(F))); plt.show()

    #hdus = fits.PrimaryHDU(F)
    #lists = fits.HDUList([hdus])
    #lists.writeto('./Filter_Test_'+str(n1)+'.fits', clobber=True)

#F = fits.open('./Filter_Test_'+str(n1)+'.fits')[0].data

R = np.real(np.fft.fftshift(np.fft.ifftn(F)))

Rec_HSC_FFT = np.real(np.fft.fftshift(np.fft.ifftn(np.dot(F, F_HST))))
Rec_HSC = np.dot(R, Cut_HST)

plt.subplot(231)
plt.imshow(Rec_HSC.reshape(N1,N2)); plt.colorbar()
plt.subplot(232)
plt.imshow(Cut_HSC); plt.colorbar()
plt.subplot(233)
plt.imshow(Cut_HSC-Rec_HSC.reshape(N1,N2)); plt.colorbar()
plt.subplot(234)
plt.imshow(Rec_HSC_FFT.reshape(N1,N2)); plt.colorbar()
plt.subplot(235)
plt.imshow(Cut_HSC); plt.colorbar()
plt.subplot(236)
plt.imshow(Cut_HSC-Rec_HSC_FFT.reshape(N1,N2)); plt.colorbar()
plt.show()
# In[ ]:




