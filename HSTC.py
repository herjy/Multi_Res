import numpy as np
import tools
import matplotlib.pyplot as plt
import scarlet
from astropy.wcs import WCS
import astropy.io.fits as fits
import warnings
warnings.simplefilter("ignore")


hdu_HST= fits.open('/Users/remy/Desktop/LSST_Project/Multi_Resolution/HSTC/acs_I_030mas_029_sci.fits')
hdu_HSC= fits.open('/Users/remy/Desktop/LSST_Project/Multi_Resolution/HSTC/cutout-HSC-I-9813-pdr1_udeep-190227-231046.fits')
FHST = hdu_HST[0].data
FHSC = hdu_HSC[1].data
hdr_HST= hdu_HST[0].header
hdr_HSC= hdu_HSC[1].header

WHST =WCS(hdu_HST[0].header)
WHSC = WCS(hdu_HSC[1].header)

#PSF
np1, np2 = 152, 152
Np1,Np2 = 26, 26

xpsf = np.array([ 4490, 9759])#np.array([5778, 5470, 4490, 9759, 3822])
ypsf = np.array([ 468, 365])#np.array([708, 907, 468, 365, 509])
Rapsf, Decpsf = WHST.all_pix2world(ypsf, xpsf,0)
Ypsf, Xpsf = WHSC.all_world2pix(Rapsf, Decpsf,0)


PSF_HST = tools.get_psf(FHST, xpsf, ypsf, np1+1, HST = True)
PSF_HSC_data = tools.get_psf(FHSC, Xpsf, Ypsf, Np1+1)
xx,yy = np.where(PSF_HSC_data*0 == 0)
r = np.sqrt((xx-10)**2+(yy-10)**2).reshape(Np1+1,Np1+1)
PSF_HSC_data[r>15]=0

xd = np.linspace(0,np1,np1+1)
yd = np.linspace(0,np2,np2+1)
xxd,yyd = np.meshgrid(xd,yd)
x0 = np.linspace(0,np1,Np1+1)
y0 = np.linspace(0,np2,Np2+1)
xx0,yy0 = np.meshgrid(x0,y0)

PSF_HSC_data_HR = tools.interp2D(xxd.flatten(),yyd.flatten(), xx0.flatten(), yy0.flatten(), PSF_HSC_data.flatten()).reshape(np1+1,np2+1)
PSF_HSC_data_HR[PSF_HSC_data_HR<0] = 0
PSF_HSC_data_HR/=np.sum(PSF_HSC_data_HR)

# Get the target PSF to partially deconvolve the image psfs
#target_psf_HST = scarlet.psf_match.fit_target_psf(PSF_HST.reshape(1,np1+1,np2+1), scarlet.psf_match.moffat)
# Display the target PSF

# Match each PSF to the target PSF
#diff_kernels, psf_blend = scarlet.psf_match.build_diff_kernels(PSF_HST.reshape(1,np1+1,np2+1), target_psf_HST, l0_thresh=0.000001)
#PSF_HST = diff_kernels[0,:,:]

# Get the target PSF to partially deconvolve the image psfs
#target_psf_HSC = scarlet.psf_match.fit_target_psf(PSF_HSC_data_HR.reshape(1,np1+1,np2+1), scarlet.psf_match.moffat)
# Display the target PSF

# Match each PSF to the target PSF
#diff_kernels, psf_blend = scarlet.psf_match.build_diff_kernels(PSF_HSC_data_HR.reshape(1,np1+1,np2+1), target_psf_HSC, l0_thresh=0.000001)
#PSF_HSC_data_HR = diff_kernels[0,:,:]



hdus = fits.PrimaryHDU(PSF_HST)
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/PSF_HST.fits', clobber=True)

hdus = fits.PrimaryHDU(PSF_HSC_data_HR)
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/PSF_HSC_Data.fits', clobber=True)

plt.subplot(121)
plt.imshow(np.log10(PSF_HST), interpolation = 'nearest', cmap = 'gist_stern')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log10(PSF_HSC_data_HR), interpolation = 'nearest', cmap = 'gist_stern')
plt.colorbar()
plt.show()

#HST coordinates
#########large###big##big##faint#####works
def match_patches(x0,y0,excess):

    '''
    :param x0, y0: coordinates of the center of the patch in High Resolutions pixels
    :param excess: half size of the box
    :return:
    x_HST, y_HST: pixel coordinates of the grid for the High resolution patch
    X_HSC, Y_HSC: pixel coordinates of the grid for the Low resolution grid
    X_HST, Y_HST: pixel coordinates of the Low resolution grid in units of the High resolution patch
    '''

    xstart = x0 - excess
    xstop = x0 + excess
    ystart = y0 - excess
    ystop = y0 + excess

    XX = np.linspace(xstart, xstop,
                      2*excess + 1)  # np.linspace(3810,3960,151)#np.linspace(10270,10340,71)#np.linspace(5170,5220,51)#XX = np.linspace(8485,8635,151)#
    YY = np.linspace(ystart, ystop,
                      2*excess + 1)  # np.linspace(7170,7320, 151)#np.linspace(4370,4440,71)#np.linspace(172,222,51)#YY = np.linspace(9500,9650, 151)

    x, y = np.meshgrid(XX, YY)
    x_HST = x.flatten().astype(int) + 0.5
    y_HST = y.flatten().astype(int) + 0.5
    Ra_HST, Dec_HST = WHST.wcs_pix2world(y_HST, x_HST, 0)

    # HSC coordinates

    Y0, X0 = WHSC.all_world2pix(Ra_HST, Dec_HST, 0)

    X = np.linspace(np.min(X0), np.max(X0)-1, np.max(X0) - np.min(X0) )
    Y = np.linspace(np.min(Y0), np.max(Y0)-1, np.max(Y0) - np.min(Y0) )

    X, Y = np.meshgrid(X, Y)
    X_HSC = X.flatten()+0.5
    Y_HSC = Y.flatten()+0.5
    Ra_HSC, Dec_HSC = WHSC.all_pix2world(Y_HSC, X_HSC, 0)  # type:
    Y_HST, X_HST = WHST.all_world2pix(Ra_HSC, Dec_HSC, 0)
    X_HST = X_HST#.astype(int)
    Y_HST = Y_HST#.astype(int)

    return x_HST, y_HST, X_HSC, Y_HSC, X_HST, Y_HST

def make_patches(x_HR, y_HR, X_LR, Y_LR, Im_HR, Im_LR):
    '''
    :param x_HST, y_HR: Coordinates of the High resolution grid
    :param X_HSC, Y_HSC: Coordinates of the Low resolution grid
    :param Im_HST: High resolution FoV
    :param Im_HSC: Low resolution FoV
    :return: Patch_HST, Patch_HSC
    '''

    N1 = np.int(X_LR.size**0.5)
    N2 = np.int(Y_LR.size**0.5)
    n1 = np.int(x_HR.size**0.5)
    n2 = np.int(y_HR.size**0.5)

    cut_HR = Im_HR[x_HR.astype(int), y_HR.astype(int)].reshape(n1,n2)
    cut_LR = Im_LR[X_LR.astype(int)+1, Y_LR.astype(int)+1].reshape(N1,N2)

    cut_HR /= (n1 * n2)
    cut_LR /= (N1 * N2)

    return cut_HR, cut_LR

x0 = 6296#5388#2675 #1436 #5992#5388 -8  #9132 -6
y0 = 13632#14579#4888 #1109 #1926#14579 -4 #10825 -4
excess =71

x_HST, y_HST, X_HSC, Y_HSC, X_HST, Y_HST = match_patches(x0,y0,excess)

cut_HST, cut_HSC = make_patches(x_HST, y_HST, X_HSC, Y_HSC, FHST, FHSC)


n1, n2 = cut_HST.shape
N1, N2 = cut_HSC.shape

plt.plot(x_HST,y_HST,'or')
plt.plot(X_HST,Y_HST, 'ob')
plt.show()

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
lists.writeto('../HSTC/Cut_HST.fits', clobber=True)

hdus = fits.PrimaryHDU(cut_HSC, header = hdr_HSC)
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/Cut_HSC.fits', clobber=True)

xp,yp = np.where(PSF_HST*0 == 0)

compute = 1
if compute:
    print('Computing Low Resolution matrix')
    mat_HSC = tools.make_mat2D_fft(x_HST, y_HST, X_HST, Y_HST, PSF_HSC_data_HR)

    hdus = fits.PrimaryHDU(mat_HSC)
    lists = fits.HDUList([hdus])
    lists.writeto('../HSTC/Mat_HSC_'+str(n1)+'.fits', clobber=True)


h = x_HST[1]-x_HST[0]
HR_filter = tools.make_vec2D_fft(x_HST, y_HST,x_HST[np.int(n1*n2/2)], y_HST[np.int(n1*n2/2)], PSF_HST, h).reshape(n1,n2)

if np.abs(compute-1):
    mat_HSC = fits.open('../HSTC/Mat_HSC_'+str(n1)+'.fits')[0].data


def filter_HR(x):
    return scarlet.transformation.Convolution(HR_filter).dot(x)#mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)
def filter_HRT(x):
    return scarlet.transformation.Convolution(HR_filter).T.dot(x)#mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)



niter = 2000
nc = 2
Sall, SHR, SLR = tools.Combine2D_filter(cut_HST, cut_HSC.flatten(), filter_HR, filter_HRT, mat_HSC, niter, verbosity = 1)
#Sall = tools.Deblend2D_filter(cut_HST.flatten(), cut_HSC.flatten(), filter_HR, filter_HRT, mat_HSC, niter, nc, verbosity = 1)


#plt.imshow(Sall[0,:].reshape(n1,n2)); plt.show()
#plt.imshow(Sall[1,:].reshape(n1,n2)); plt.show()

hdus = fits.PrimaryHDU(Sall.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/HSTC_Sall_patch1.fits', clobber=True)
hdus = fits.PrimaryHDU(SHR.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/HSTC_SHR_patch1.fits', clobber=True)
hdus = fits.PrimaryHDU(SLR.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/HSTC_SLR_patch1.fits', clobber=True)

if 1:
    font = 25
    plt.figure(6)
    plt.imshow(Sall.reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('Joint Reconstruction', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/Joint_Reconstruction_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(12)
    plt.imshow(SLR.reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('HSC deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HSC-deconvolution_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(9)
    plt.imshow(SHR.reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('HST deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HST_deconvolution_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(5)
    plt.imshow(filter_HR(Sall.reshape(n1,n2)), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('HST joint model', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HST_joint_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(4)
    plt.imshow(np.dot(Sall, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.title('HSC joint model', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HSC_joint_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(2)
    plt.title('HST image', fontsize = font)
    plt.imshow(cut_HST, interpolation = 'none', cmap = 'CMRmap')
    plt.savefig('Images/HST_image_'+str(x0)+'_'+str(y0)+'.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(1)
    plt.title('HSC image', fontsize = font)
    plt.imshow(cut_HSC, interpolation = 'none', cmap = 'CMRmap')
    plt.savefig('Images/HSC_image_'+str(x0)+'_'+str(y0)+'.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(8)
    plt.title('HST deconvolution model', fontsize = font)
    plt.imshow(filter_HR(SHR.reshape(n1,n2)), interpolation = 'none', cmap = 'CMRmap')
    plt.savefig('Images/HST_deconvolution_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.colorbar()
    plt.axis('off')
    plt.figure(10)
    plt.title('HSC deconvolution model', fontsize = font)
    plt.imshow(np.dot(SLR, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/HSC_deconvolution_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(14)
    plt.title('Residual joint HST', fontsize = font)
    plt.imshow(cut_HST - filter_HR(Sall.reshape(n1,n2)), interpolation = 'none', cmap = 'CMRmap')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/Residual_joint_HST_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(13)
    plt.title('Residual joint HSC', fontsize = font)
    plt.imshow(cut_HSC - np.dot(Sall, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'CMRmap')

    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/Residual_joint_HSC_'+str(x0)+'_'+str(y0)+'.png')
    plt.show()