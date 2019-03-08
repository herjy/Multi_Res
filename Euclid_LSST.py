import numpy as np
import matplotlib.pyplot as plt
import SLIT
import tools
import scarlet.display
import sep
import astropy.io.fits as fits
from astropy.wcs import WCS
import warnings
warnings.simplefilter("ignore")


HST = fits.open('../IMAGES_diff_surveys/CANDELS_V_60mas.fits')
Euclid = fits.open('../IMAGES_diff_surveys/Euclid_VI_100mas.fits')
LSST = fits.open('../IMAGES_diff_surveys/LSST_V_200mas_full.fits')

HST0 = HST[0].data.astype(float)
WHST = WCS(HST[0].header)
Euclid0 = Euclid[0].data.astype(float)
WEuclid = WCS(Euclid[0].header)
LSST0 = LSST[0].data.astype(float)
WLSST = WCS(LSST[0].header)


#PSF
sigma_HR = SLIT.tools.MAD(Euclid0)
sigma_HHR = SLIT.tools.MAD(HST0)
sigma_LR = SLIT.tools.MAD(LSST0)

np1, np2 = 61,61#PSF_Euclid0.shape
Np1, Np2 = 31,31

Xpsf, Ypsf = [94], [36]
xpsf, ypsf = [188], [71]

PSF_Euclid = tools.get_psf(Euclid0, xpsf, ypsf, np1)
PSF_LSST = tools.get_psf(LSST0, Xpsf, Ypsf, Np1)


xd = np.linspace(0,np1-1,np1)
yd = np.linspace(0,np2-1,np2)
xxd,yyd = np.meshgrid(xd,yd)
x0 = np.linspace(0,np1-1,Np1)
y0 = np.linspace(0,np2-1,Np2)
xx0,yy0 = np.meshgrid(x0,y0)


PSF_LSST = tools.interp2D(xxd.flatten(),yyd.flatten(), xx0.flatten(), yy0.flatten(), PSF_LSST.flatten()).reshape(np1,np2)

plt.subplot(121)
plt.imshow(np.log10(PSF_Euclid), interpolation = 'none', cmap = 'gist_stern')
plt.subplot(122)
plt.imshow(np.log10(PSF_LSST), interpolation = 'none', cmap = 'gist_stern')
plt.show()

#Data
n1, n2 = 101., 101.
N1, N2 = 51., 51.

x0 = 195
y0 = 245
excess =100
xstart =x0-excess
xstop =x0+excess
ystart =y0-excess
ystop =y0+excess

XX = np.linspace(xstart,  xstop,2*excess+1)#np.linspace(3810,3960,151)#np.linspace(10270,10340,71)#np.linspace(5170,5220,51)#XX = np.linspace(8485,8635,151)#
YY = np.linspace(ystart, ystop,2*excess+1)#np.linspace(7170,7320, 151)#np.linspace(4370,4440,71)#np.linspace(172,222,51)#YY = np.linspace(9500,9650, 151)
x,y = np.meshgrid(XX,YY)
x_Euclid = x.flatten().astype(int)
y_Euclid = y.flatten().astype(int)
Ra_Euclid, Dec_Euclid = WEuclid.wcs_pix2world(y_Euclid, x_Euclid,0)


#HSC coordinates
Y0, X0 = WLSST.wcs_world2pix(Ra_Euclid, Dec_Euclid,0)
X = np.linspace(np.min(X0), np.max(X0), np.max(X0)-np.min(X0)+1)
Y = np.linspace(np.min(Y0), np.max(Y0), np.max(Y0)-np.min(Y0)+1)

X,Y = np.meshgrid(X,Y)
X_LSST = X.flatten()
Y_LSST = Y.flatten()
Ra_LSST, Dec_LSST = WLSST.all_pix2world(Y_LSST, X_LSST,0)  # type:
Y_Euclid, X_Euclid = WEuclid.all_world2pix(Ra_LSST, Dec_LSST, 0)
X_Euclid = X_Euclid.astype(int)
Y_Euclid = Y_Euclid.astype(int)

Cut_Euclid = Euclid0[x0-n1/2:x0+n1/2, y0-n2/2:y0+n2/2]
Cut_LSST = LSST0[x0/2-N1/2+1:x0/2+N1/2+1, y0/2-N2/2+1:y0/2+N2/2+1]

print(Cut_Euclid.shape, Cut_LSST.shape)

hdus = fits.PrimaryHDU(Cut_Euclid)
lists = fits.HDUList([hdus])
lists.writeto('Image_Euclid.fits', clobber=True)
hdus = fits.PrimaryHDU(Cut_LSST)
lists = fits.HDUList([hdus])
lists.writeto('Image_LSST.fits', clobber=True)

plt.subplot(121)
plt.imshow(Cut_Euclid, interpolation = 'none', cmap = 'gist_stern')
plt.subplot(122)
plt.imshow(Cut_LSST, interpolation = 'none', cmap = 'gist_stern')
plt.show()



xxp = np.linspace(-np1/2,np1/2+1,np1)
yyp = np.linspace(-np2/2, np2/2+1,np2)
xp,yp = np.meshgrid(xxp,yyp)
xp=xp.flatten()
yp=yp.flatten()

XXp = np.linspace(-np1/2,np1/2+1,Np1)
YYp = np.linspace(-np2/2,np2/2+1,Np2)
Xp,Yp = np.meshgrid(XXp,YYp)
Xp=Xp.flatten()
Yp=Yp.flatten()

hdus = fits.PrimaryHDU(PSF_Euclid)
lists = fits.HDUList([hdus])
lists.writeto('PSF_Euclid.fits', clobber=True)
hdus = fits.PrimaryHDU(PSF_LSST)
lists = fits.HDUList([hdus])
lists.writeto('PSF_LSST.fits', clobber=True)



print('building operator:')

print('Low resolution operator')
mat_LR = tools.make_mat2D_fft(x_Euclid, y_Euclid, X_Euclid, Y_Euclid, PSF_LSST)

print('High resolution operator')
h = x_Euclid[1]-x_Euclid[0]
HR_filter = tools.make_vec2D_fft(x_Euclid, y_Euclid,x_Euclid[n1*n2/2], y_Euclid[n1*n2/2], PSF_Euclid, h)#.reshape(n1,n2)

print(HR_filter.shape)


print('Now lets solve this!')
def filter_HR(x):
    return scarlet.transformation.LinearFilter(HR_filter).dot(x)#mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)
def filter_HRT(x):
    return scarlet.transformation.LinearFilter(HR_filter.T).dot(x)#mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)

niter = 250
Sall, SHR, SLR = tools.Combine2D_filter(Cut_Euclid, Cut_LSST.flatten(), filter_HR, filter_HRT, mat_LR, niter, verbosity = 1)

if 1:
    font = 25
    plt.figure(0)
    plt.imshow(Sall.reshape(n1,n2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.title('Joint Reconstruction', fontsize = font)
    plt.axis('off')
    plt.savefig('Joint_Reconstruction_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(1)
    plt.imshow(SLR.reshape(n1,n2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.title('HSC deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('HSC-deconvolution_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(2)
    plt.imshow(SHR.reshape(n1,n2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.title('HST deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('HST_deconvolution_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(3)
    plt.imshow(filter_HR(Sall.reshape(n1,n2)), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.title('HST joint model', fontsize = font)
    plt.axis('off')
    plt.savefig('HST_joint_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(4)
    plt.imshow(np.dot(Sall, mat_LR).reshape(N1,N2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.title('HSC joint model', fontsize = font)
    plt.axis('off')
    plt.savefig('HSC_joint_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(6)
    plt.title('HST image', fontsize = font)
    plt.imshow(Cut_Euclid, interpolation = 'none', cmap = 'gist_stern')
    plt.savefig('HST_image_'+str(x0)+'_'+str(y0)+'.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(7)
    plt.title('HSC image', fontsize = font)
    plt.imshow(Cut_LSST, interpolation = 'none', cmap = 'gist_stern')
    plt.savefig('HSC_image_'+str(x0)+'_'+str(y0)+'.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(8)
    plt.title('HST deconvolution model', fontsize = font)
    plt.imshow(filter_HR(SHR.reshape(n1,n2)), interpolation = 'none', cmap = 'gist_stern')
    plt.savefig('HST_deconvolution_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(9)
    plt.title('HSC deconvolution model', fontsize = font)
    plt.imshow(np.dot(SLR, mat_LR).reshape(N1,N2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('HSC_deconvolution_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(10)
    plt.title('Residual joint HST', fontsize = font)
    plt.imshow(Cut_Euclid - filter_HR(Sall.reshape(n1,n2)), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Residual_joint_HST_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(11)
    plt.title('Residual joint HSC', fontsize = font)
    plt.imshow(Cut_LSST - np.dot(Sall, mat_LR).reshape(N1,N2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Residual_joint_HSC_'+str(x0)+'_'+str(y0)+'.png')
    plt.show()

stop
objects = sep.extract(LSST0, 5*sigma_LR, deblend_cont = 0.05)

ne1,ne2 = np.shape(Euclid0)
nl1,nl2 = np.shape(LSST0)
Solution_all = np.zeros((ne1,ne2))
Solution_HR = np.zeros((ne1,ne2))
Solution_LR = np.zeros((ne1,ne2))
Sall_cube = np.zeros((np.size(objects), n1,n2))
SHR_cube = np.zeros((np.size(objects), n1,n2))
SLR_cube = np.zeros((np.size(objects), n1,n2))
c = 0
for object in objects:

    y, x = object['x'], object['y']

    if (x >=N1/2) and (y>=N2/2) and (nl1-x>=N1/2) and (nl2-y>=N2/2):

        YHR =Euclid0[x*2-n1/2:x*2+n1/2,y*2-n2/2:y*2+n2/2].reshape(n1*n2)
        YLR =LSST0[x-N1/2:x+N1/2,y-N2/2:y+N2/2].reshape(N1*N2)

        Sall, SHR, SLR = Solve(YHR, YLR, mat_HR, mat_LR, 20000, verbosity = 0)

        Solution_all[x*2-n1/2:x*2+n1/2,y*2-n2/2:y*2+n2/2] = Sall.reshape(n1,n2)
        Solution_HR[x*2-n1/2:x*2+n1/2,y*2-n2/2:y*2+n2/2] = SHR.reshape(n1,n2)
        Solution_LR[x*2-n1/2:x*2+n1/2,y*2-n2/2:y*2+n2/2] = SLR.reshape(n1,n2)
        Sall_cube[c,:,:] = Sall.reshape(n1,n2)
        SHR_cube[c,:,:] = SHR.reshape(n1,n2)
        SLR_cube[c,:,:] = SLR.reshape(n1,n2)
        hdus = fits.PrimaryHDU(Sall_cube)
        lists = fits.HDUList([hdus])
        lists.writeto('All_cube.fits', clobber=True)
        hdus = fits.PrimaryHDU(SHR_cube)
        lists = fits.HDUList([hdus])
        lists.writeto('HR_cube.fits', clobber=True)
        hdus = fits.PrimaryHDU(SLR_cube)
        lists = fits.HDUList([hdus])
        lists.writeto('LR_cube.fits', clobber=True)
        c+=1

hdus = fits.PrimaryHDU(Solution_all)
lists = fits.HDUList([hdus])
lists.writeto('Euclid_Sall.fits', clobber=True)
hdus = fits.PrimaryHDU(Solution_HR)
lists = fits.HDUList([hdus])
lists.writeto('Euclid_SHR.fits', clobber=True)
hdus = fits.PrimaryHDU(Solution_LR)
lists = fits.HDUList([hdus])
lists.writeto('Euclid_SLR.fits', clobber=True)
