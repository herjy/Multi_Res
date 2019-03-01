import numpy as np
import matplotlib.pyplot as plt
import tools
from astropy.wcs import WCS
import astropy.io.fits as fits
import warnings
warnings.simplefilter("ignore")


hdu_HST= fits.open('/Users/remy/Desktop/LSST_Project/Multi_Resolution/HS_TC/acs_I_030mas_029_sci.fits')
hdu_HSC= fits.open('/Users/remy/Desktop/LSST_Project/Multi_Resolution/HS_TC/HSC_Field.fits')
FHST = hdu_HST[0].data
FHSC = hdu_HSC[0].data

#HST coordinates
YY = np.linspace(172,222,51)
XX = np.linspace(5170,5220,51)
x,y = np.meshgrid(XX,YY)
x = x.flatten()
y = y.flatten()

WHST =WCS(hdu_HST[0].header)
Ra, Dec = WHST.wcs_pix2world(y, x,0)

#HSC coordinates
WHSC = WCS(hdu_HSC[0].header)
X, Y = WHSC.wcs_world2pix(Ra, Dec,0)


xpsf = np.array([5778, 5470, 4490, 9759, 3822])
ypsf = np.array([708, 907, 468, 365, 509])
Rapsf, Decpsf = WHST.wcs_pix2world(ypsf, xpsf,0)
Ypsf, Xpsf = WHSC.wcs_world2pix(Rapsf, Decpsf,0)


PSF_HST = tools.get_psf(FHST, xpsf, ypsf, 81)
PSF_HST[PSF_HST<0] = 0
PSF_HST/=np.sum(PSF_HST)
plt.imshow(np.log10(PSF_HST), interpolation = 'nearest'); plt.show()

hdus = fits.PrimaryHDU(PSF_HST)
lists = fits.HDUList([hdus])
lists.writeto('PSF_HST.fits', clobber=True)

PSF_HSC = tools.get_psf(FHSC, Xpsf, Ypsf, 41)
PSF_HSC[PSF_HSC<0] = 0
PSF_HSC_data = PSF_HSC/np.sum(PSF_HSC)
xx,yy = np.where(PSF_HSC*0 == 0)
r = np.sqrt((xx-20)**2+(yy-20)**2).reshape(41,41)

PSF_HSC_data[r>15]=0
hdus = fits.PrimaryHDU(PSF_HSC_data)
lists = fits.HDUList([hdus])
lists.writeto('PSF_HSC_Data.fits', clobber=True)

print(PSF_HSC_data.shape)

PSF_HSC = fits.open('/Users/remy/Desktop/LSST_Project/Multi_Resolution/HS_TC/PSF_HSC.fits')[0].data
xd = np.linspace(0,40,41)
yd = np.linspace(0,40,41)
xxd,yyd = np.meshgrid(xd,yd)
x0 = np.linspace(0,40,40)
y0 = np.linspace(0,40,40)
xx0,yy0 = np.meshgrid(x0,y0)

#PSF_HSC = tools.interp2D(xx.flatten(),yyd.flatten(),xx0.flatten(), yy0.flatten(), PSF_HSC.flatten()).reshape(41,41)

plt.subplot(131)
plt.imshow(np.log10(PSF_HSC_data), interpolation = 'nearest'); plt.colorbar()
plt.subplot(132)
plt.imshow(np.log10(PSF_HSC), interpolation = 'nearest'); plt.colorbar()
plt.subplot(133)
plt.imshow(np.log10(PSF_HSC-PSF_HSC_data), interpolation = 'nearest'); plt.colorbar()
plt.show()



cut_HST = FHST[np.min(x):np.max(x), np.min(y):np.max(y)]#[5170:5220,172:222]
plt.imshow(cut_HST, cmap = 'gist_stern', interpolation = 'nearest')
plt.show()
cut_HSC = hdu_HSC[2].data[np.min(X):np.max(X), np.min(Y):np.max(Y)]
plt.imshow(cut_HSC, cmap = 'gist_stern', interpolation = 'nearest')
plt.show()
