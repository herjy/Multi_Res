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
hdr_HST= hdu_HST[0].header
hdr_HSC= hdu_HSC[0].header


WHST =WCS(hdu_HST[0].header)
WHSC = WCS(hdu_HSC[0].header)


xpsf = np.array([5778, 5470, 4490, 9759, 3822])
ypsf = np.array([708, 907, 468, 365, 509])
Rapsf, Decpsf = WHST.wcs_pix2world(ypsf, xpsf,0)
Ypsf, Xpsf = WHSC.wcs_world2pix(Rapsf, Decpsf,0)


PSF_HST = tools.get_psf(FHST, xpsf, ypsf, 101)
PSF_HST[PSF_HST<0] = 0
PSF_HST/=np.sum(PSF_HST)

hdus = fits.PrimaryHDU(PSF_HST)
lists = fits.HDUList([hdus])
lists.writeto('PSF_HST.fits', clobber=True)

PSF_HSC = tools.get_psf(FHSC, Xpsf, Ypsf, 21)
PSF_HSC[PSF_HSC<0] = 0
PSF_HSC_data = PSF_HSC/np.sum(PSF_HSC)
xx,yy = np.where(PSF_HSC*0 == 0)
r = np.sqrt((xx-10)**2+(yy-10)**2).reshape(21,21)

xd = np.linspace(0,100,101)
yd = np.linspace(0,100,101)
xxd,yyd = np.meshgrid(xd,yd)
x0 = np.linspace(0,100,21)
y0 = np.linspace(0,100,21)
xx0,yy0 = np.meshgrid(x0,y0)

PSF_HSC_data[r>15]=0

PSF_HSC_data_HR = tools.interp2D(xxd.flatten(),yyd.flatten(), xx0.flatten(), yy0.flatten(), PSF_HSC_data.flatten()).reshape(101,101)

PSF_HSC_data_HR[PSF_HSC_data_HR<0] = 0
PSF_HSC_data_HR/=np.sum(PSF_HSC_data_HR)

hdus = fits.PrimaryHDU(PSF_HSC_data_HR)
lists = fits.HDUList([hdus])
lists.writeto('PSF_HSC_Data.fits', clobber=True)

plt.subplot(121)
plt.imshow(np.log10(PSF_HST), interpolation = 'nearest', cmap = 'gist_stern')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log10(PSF_HSC_data_HR), interpolation = 'nearest', cmap = 'gist_stern')
plt.colorbar()
plt.show()

#HST coordinates
YY = np.linspace(172,222,51)
XX = np.linspace(5170,5220,51)
x,y = np.meshgrid(XX,YY)
x_HST = x.flatten()
y_HST = y.flatten()
Ra_HST, Dec_HST = WHST.wcs_pix2world(y_HST, x_HST,0)
#HSC coordinates
Y0, X0 = WHSC.wcs_world2pix(Ra_HST, Dec_HST,0)
X = np.linspace(np.min(X0), np.max(X0), np.max(X0)-np.min(X0)+1)
Y = np.linspace(np.min(Y0), np.max(Y0), np.max(Y0)-np.min(Y0)+1)

X,Y = np.meshgrid(X,Y)
X_HSC = X.flatten()
Y_HSC = Y.flatten()
Ra_HSC, Dec_HSC = WHSC.wcs_pix2world(Y_HSC, X_HSC,0)  # type:


plt.plot(Ra_HST,Dec_HST,'or')
plt.plot(Ra_HSC,Dec_HSC, 'ob')
plt.show()

cut_HST = FHST[np.min(x):np.max(x), np.min(y):np.max(y)]#[5170:5220,172:222]
cut_HSC = FHSC[np.min(X):np.max(X), np.min(Y):np.max(Y)]

print(cut_HST.shape, cut_HSC.shape)
plt.subplot(121)
plt.imshow(cut_HST, cmap = 'gist_stern', interpolation = 'nearest')
plt.subplot(122)
plt.imshow(cut_HSC, cmap = 'gist_stern', interpolation = 'nearest')
plt.show()


hdus = fits.PrimaryHDU(cut_HST, header = hdr_HST)
lists = fits.HDUList([hdus])
lists.writeto('Cut_HST.fits', clobber=True)

hdus = fits.PrimaryHDU(cut_HSC, header = hdr_HSC)
lists = fits.HDUList([hdus])
lists.writeto('Cut_HSC.fits', clobber=True)

xp,yp = np.where(PSF_HST*0==0)

print('Computing Low resolution matrix')
mat_HST = tools.make_mat2D(x_HST, y_HST, x_HST, y_HST, PSF_HST, x_HST, y_HST, xp, yp)
print('Computing High resolution matrix')
mat_HST = tools.make_mat2D(x_HST, y_HST, x_HSC, y_HSC, PSF_HSC, x_HST, y_HST, xp, yp)