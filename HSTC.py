import numpy as np
import tools
import matplotlib.pyplot as plt
import scarlet
from astropy.wcs import WCS
import astropy.io.fits as fits
from multiprocess import Process
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


xpsf = np.array([ 4490, 9759])#np.array([5778, 5470, 4490, 9759, 3822])
ypsf = np.array([ 468, 365])#np.array([708, 907, 468, 365, 509])
Rapsf, Decpsf = WHST.all_pix2world(ypsf, xpsf,0)
Ypsf, Xpsf = WHSC.all_world2pix(Rapsf, Decpsf,0)


PSF_HST = tools.get_psf(FHST, xpsf, ypsf, 103, HST = True)
PSF_HST[PSF_HST<0] = 0
PSF_HST/=np.sum(PSF_HST)

hdus = fits.PrimaryHDU(PSF_HST)
lists = fits.HDUList([hdus])
lists.writeto('../HS_TC/PSF_HST.fits', clobber=True)

PSF_HSC = tools.get_psf(FHSC, Xpsf, Ypsf, 19)
PSF_HSC[PSF_HSC<0] = 0
PSF_HSC_data = PSF_HSC/np.sum(PSF_HSC)
xx,yy = np.where(PSF_HSC*0 == 0)
r = np.sqrt((xx-10)**2+(yy-10)**2).reshape(19,19)

xd = np.linspace(0,100,103)
yd = np.linspace(0,100,103)
xxd,yyd = np.meshgrid(xd,yd)
x0 = np.linspace(0,100,19)
y0 = np.linspace(0,100,19)
xx0,yy0 = np.meshgrid(x0,y0)

PSF_HSC_data[r>15]=0

PSF_HSC_data_HR = tools.interp2D(xxd.flatten(),yyd.flatten(), xx0.flatten(), yy0.flatten(), PSF_HSC_data.flatten()).reshape(103,103)

PSF_HSC_data_HR[PSF_HSC_data_HR<0] = 0
PSF_HSC_data_HR/=np.sum(PSF_HSC_data_HR)

hdus = fits.PrimaryHDU(PSF_HSC_data_HR)
lists = fits.HDUList([hdus])
lists.writeto('../HS_TC/PSF_HSC_Data.fits', clobber=True)

plt.subplot(121)
plt.imshow(np.log10(PSF_HST), interpolation = 'nearest', cmap = 'gist_stern')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log10(PSF_HSC_data_HR), interpolation = 'nearest', cmap = 'gist_stern')
plt.colorbar()
plt.show()


#HST coordinates
###########works
x0 = 5388-6 #9132
y0 = 14579#10825
excess =75
xstart =x0-excess
xstop =x0+excess
ystart =y0-excess
ystop =y0+excess

XX = np.linspace(xstart,  xstop,2*excess+1)#np.linspace(3810,3960,151)#np.linspace(10270,10340,71)#np.linspace(5170,5220,51)#XX = np.linspace(8485,8635,151)#
YY = np.linspace(ystart, ystop,2*excess+1)#np.linspace(7170,7320, 151)#np.linspace(4370,4440,71)#np.linspace(172,222,51)#YY = np.linspace(9500,9650, 151)
x,y = np.meshgrid(XX,YY)
x_HST = x.flatten().astype(int)
y_HST = y.flatten().astype(int)
Ra_HST, Dec_HST = WHST.wcs_pix2world(y_HST, x_HST,0)

#HSC coordinates
Y0, X0 = WHSC.wcs_world2pix(Ra_HST, Dec_HST,0)
X = np.linspace(np.min(X0), np.max(X0), np.max(X0)-np.min(X0)+1)
Y = np.linspace(np.min(Y0), np.max(Y0), np.max(Y0)-np.min(Y0)+1)

X,Y = np.meshgrid(X,Y)
X_HSC = X.flatten()
Y_HSC = Y.flatten()
Ra_HSC, Dec_HSC = WHSC.all_pix2world(Y_HSC, X_HSC,0)  # type:
Y_HST, X_HST = WHST.all_world2pix(Ra_HSC, Dec_HSC, 0)
X_HST = X_HST.astype(int)
Y_HST = Y_HST.astype(int)


xp_wcs = np.linspace(np.min(x_HST)-26, np.max(x_HST)+26, 103)
yp_wcs = np.linspace(np.min(y_HST)-26, np.max(y_HST)+26, 103)
xxp_wcs,yyp_wcs = np.meshgrid(xp_wcs,yp_wcs)
xp_wcs = xxp_wcs.flatten()
yp_wcs = yyp_wcs.flatten()

n1 = np.int(x_HST.size**0.5)
n2 = np.int(y_HST.size**0.5)

plt.plot(x_HST,y_HST,'or')
plt.plot(X_HST,Y_HST, 'ob')
#plt.plot(xp_wcs,yp_wcs, 'og')
plt.show()

cut_HST = FHST[x_HST-4, y_HST-6].reshape(n1,n2)#[np.int(np.min(x)):np.int(np.max(x)), np.int(np.min(y)):np.int(np.max(y))]#[5170:5220,172:222]
cut_HSC = FHSC[X_HSC.astype(int),Y_HSC.astype(int)]#[np.int(np.min(X)):np.int(np.max(X)), np.int(np.min(Y))+1:np.int(np.max(Y))]
N1 = np.int(X_HSC.size**0.5)
N2 = np.int(Y_HSC.size**0.5)

cut_HSC = cut_HSC.reshape(N1,N2)


cut_HST /= np.sum(cut_HST)/(n1*n2)
cut_HSC /= np.sum(cut_HSC)/(N1*N2)


print(n1,n2,N1,N2)

plt.subplot(121)
plt.imshow(cut_HST, cmap = 'gist_stern', interpolation = 'nearest')
plt.colorbar()
plt.subplot(122)
plt.imshow(cut_HSC, cmap = 'gist_stern', interpolation = 'nearest')
plt.colorbar()
plt.show()


hdus = fits.PrimaryHDU(cut_HST, header = hdr_HST)
lists = fits.HDUList([hdus])
lists.writeto('../HS_TC/Cut_HST.fits', clobber=True)

hdus = fits.PrimaryHDU(cut_HSC, header = hdr_HSC)
lists = fits.HDUList([hdus])
lists.writeto('../HS_TC/Cut_HSC.fits', clobber=True)

xp,yp = np.where(PSF_HST*0 == 0)

compute = 0
if compute:
    print('Computing Low Resolution matrix')
    mat_HSC = tools.make_mat2D_fft(x_HST, y_HST, X_HST, Y_HST, PSF_HSC_data_HR)

    hdus = fits.PrimaryHDU(mat_HSC)
    lists = fits.HDUList([hdus])
    lists.writeto('../HS_TC/Mat_HSC_'+str(n1)+'.fits', clobber=True)

    print('Computing High Resolution matrix')
    mat_HST = tools.make_mat2D_fft(x_HST, y_HST, x_HST, y_HST, PSF_HST)

    hdus = fits.PrimaryHDU(mat_HST)
    lists = fits.HDUList([hdus])
    lists.writeto('../HS_TC/Mat_HST_'+str(n1)+'.fits', clobber=True)

if np.abs(compute-1):
    mat_HSC = fits.open('../HS_TC/Mat_HSC_'+str(n1)+'.fits')[0].data
    mat_HST = fits.open('../HS_TC/Mat_HST_'+str(n1)+'.fits')[0].data

def filter_HR(x):
    return scarlet.transformation.LinearFilter(mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)).dot(x)
def filter_HRT(x):
    return scarlet.transformation.LinearFilter(mat_HST[np.int(n1*n1/2),:].reshape(n1,n2).T).dot(x)

niter = 50
Sall, SHR, SLR = tools.Combine2D_filter(cut_HST, cut_HSC.flatten(), filter_HR, filter_HRT, mat_HSC, niter, verbosity = 1)
#Sall, SHR, SLR = tools.Combine2D(cut_HST.flatten(), cut_HSC.flatten(), mat_HST, mat_HSC, niter, verbosity = 1)

hdus = fits.PrimaryHDU(Sall.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HS_TC/HSTC_Sall_patch1.fits', clobber=True)
hdus = fits.PrimaryHDU(SHR.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HS_TC/HSTC_SHR_patch1.fits', clobber=True)
hdus = fits.PrimaryHDU(SLR.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HS_TC/HSTC_SLR_patch1.fits', clobber=True)

if 1:
    font = 25
    plt.figure(0)
    plt.imshow(Sall.reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('Joint Reconstruction', fontsize = font)
    plt.axis('off')
    plt.savefig('Joint_Reconstruction.png')
    plt.figure(1)
    plt.imshow(SLR.reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('HSC deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('HSC-deconvolution.png')
    plt.figure(2)
    plt.imshow(SHR.reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('HST deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('HST_deconvolution.png')
    plt.figure(3)
    plt.imshow(np.dot(Sall, mat_HST).reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('HST joint model', fontsize = font)
    plt.axis('off')
    plt.savefig('HST_joint_model.png')
    plt.figure(4)
    plt.imshow(np.dot(Sall, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('HSC joint model', fontsize = font)
    plt.axis('off')
    plt.savefig('HSC_joint_model.png')
    plt.figure(6)
    plt.title('HST image', fontsize = font)
    plt.imshow(cut_HST, interpolation = 'none', cmap = 'CMRmap')
    plt.savefig('HST_image.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(7)
    plt.title('HSC image', fontsize = font)
    plt.imshow(cut_HSC, interpolation = 'none', cmap = 'CMRmap')
    plt.savefig('HSC_image.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(8)
    plt.title('HST deconvolution model', fontsize = font)
    plt.imshow(np.dot(SHR, mat_HST).reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.savefig('HST_deconvolution_model.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(9)
    plt.title('HSC deconvolution model', fontsize = font)
    plt.imshow(np.dot(SLR, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('HSC_deconvolution_model.png')
    plt.figure(10)
    plt.title('Residual joint HST', fontsize = font)
    plt.imshow(cut_HST - np.dot(Sall, mat_HST).reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Residual_joint_HST.png')
    plt.figure(11)
    plt.title('Residual joint HSC', fontsize = font)
    plt.imshow(cut_HSC - np.dot(Sall, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Residual_joint_HSC.png')
    plt.show()