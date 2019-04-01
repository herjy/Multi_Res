import numpy as np
import tools
import matplotlib.pyplot as plt
import scipy as scp
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
np1, np2 = 170, 170

xpsf = np.array([ 4490, 9759])#np.array([5778, 5470, 4490, 9759, 3822])
ypsf = np.array([ 468, 365])#np.array([708, 907, 468, 365, 509])
Rapsf, Decpsf = WHST.all_pix2world(ypsf, xpsf,0)
Ypsf, Xpsf = WHSC.all_world2pix(Rapsf, Decpsf,0)


PSF_HST, PSF_HSC_data_HR = tools.get_psf(FHST, FHSC,xpsf,ypsf, WHST, WHSC, np1)

print(np.where(PSF_HSC_data_HR == np.max(PSF_HSC_data_HR)), PSF_HSC_data_HR.shape)
# Get the target PSF to partially deconvolve the image psfs
#target_psf_HST, r, t = scarlet.psf_match.fit_target_psf(PSF_HST.reshape(1,np1+1,np2+1), scarlet.psf_match.moffat)
# Display the target PSF

# Match each PSF to the target PSF
#diff_kernels, psf_blend = scarlet.psf_match.build_diff_kernels(PSF_HST.reshape(1,np1+1,np2+1), target_psf_HST, l0_thresh=0.000001)
#PSF_HST = diff_kernels[0,:,:]

# Get the target PSF to partially deconvolve the image psfs
#target_psf_HSC, r, t = scarlet.psf_match.fit_target_psf(PSF_HSC_data_HR.reshape(1,np1+1,np2+1), scarlet.psf_match.moffat)
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
plt.imshow(np.log10(PSF_HST), interpolation = 'nearest', cmap = 'inferno')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.log10(PSF_HSC_data_HR), interpolation = 'nearest', cmap = 'inferno')
plt.colorbar()
plt.show()

#HST coordinates
#########large###big##big##faint#####works


x0 = 6296#5388#6296#5388#2675 #1436 #5992#5388 -8  #9132 -6
y0 = 13632#14579 #13632#14579#4888 #1109 #1926#14579 -4 #10825 -4
excess =91

x_HST, y_HST, X_HSC, Y_HSC, X_HST, Y_HST = tools.match_patches(x0,y0,WHSC, WHST, excess)

cut_HST, cut_HSC = tools.make_patches(x_HST, y_HST, X_HSC, Y_HSC, FHST, FHSC)

n1, n2 = cut_HST.shape
N1, N2 = cut_HSC.shape

#cut_HST += np.random.randn(n1,n2)*tools.MAD(cut_HST)*5

plt.plot(x_HST,y_HST,'or')
plt.plot(X_HST,Y_HST, 'ob')
plt.show()

print(n1,n2,N1,N2)

plt.subplot(121)
plt.imshow(cut_HST, cmap = 'inferno', interpolation = 'nearest')
plt.colorbar()
plt.subplot(122)
plt.imshow(cut_HSC, cmap = 'inferno', interpolation = 'nearest')
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
HR_filter = tools.make_vec2D_fft(x_HST, y_HST,X_HST[np.int(N1*N2/2)], Y_HST[np.int(N1*N2/2)], PSF_HST, h).reshape(n1,n2)

if np.abs(compute-1):
    mat_HSC = fits.open('../HSTC/Mat_HSC_'+str(n1)+'.fits')[0].data


def filter_HR(x):
    return scarlet.transformation.Convolution(HR_filter).dot(x)#mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)
def filter_HRT(x):
    return scarlet.transformation.Convolution(HR_filter).T.dot(x)#mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)


reg_HR = np.sum(HR_filter**2)**0.5

print('Noise in HST:', tools.MAD(cut_HST))
print('Noise in HSC:', tools.MAD(cut_HSC))


niter = 400
nc = 2
Sall, SHR, SLR = tools.Combine2D_filter(cut_HST, cut_HSC.flatten(), filter_HR, filter_HRT, mat_HSC, niter, verbosity = 1, reg_HR = 0)
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
    plt.imshow(Sall.reshape(n1,n2), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.title('Joint Reconstruction', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/Joint_Reconstruction_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(12)
    plt.imshow(SLR.reshape(n1,n2), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.title('HSC deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HSC-deconvolution_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(9)
    plt.imshow(SHR.reshape(n1,n2), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.title('HST deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HST_deconvolution_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(5)
    plt.imshow(filter_HR(Sall.reshape(n1,n2)), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.title('HST joint model', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HST_joint_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(4)
    plt.imshow(np.dot(Sall, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.title('HSC joint model', fontsize = font)
    plt.savefig('Images/HSC_joint_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(2)
    plt.title('HST image', fontsize = font)
    plt.imshow(cut_HST, interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/HST_image_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(1)
    plt.title('HSC image', fontsize = font)
    plt.imshow(cut_HSC, interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/HSC_image_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(8)
    plt.title('HST deconvolution model', fontsize = font)
    plt.imshow(filter_HR(SHR.reshape(n1,n2)), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/HST_deconvolution_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(10)
    plt.title('HSC deconvolution model', fontsize = font)
    plt.imshow(np.dot(SLR, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/HSC_deconvolution_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(14)
    plt.title('Residual joint HST', fontsize = font)
    plt.imshow(cut_HST - filter_HR(Sall.reshape(n1,n2)), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/Residual_joint_HST_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(13)
    plt.title('Residual joint HSC', fontsize = font)
    plt.imshow(cut_HSC - np.dot(Sall, mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/Residual_joint_HSC_'+str(x0)+'_'+str(y0)+'.png')
    plt.show()