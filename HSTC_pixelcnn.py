import numpy as np
import tools
import matplotlib.pyplot as plt

import tensorflow_hub as hub
import tensorflow as tf

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
FHSC -= np.median(FHSC)


WHST =WCS(hdu_HST[0].header)
WHSC = WCS(hdu_HSC[1].header)

#PSF
make_PSF = 0
if make_PSF == 1:
    np1, np2 = 170, 170

    xpsf = np.array([4490, 9759])  # np.array([5778, 5470, 4490, 9759, 3822])
    ypsf = np.array([468, 365])
    Rapsf, Decpsf = WHST.wcs_pix2world(ypsf, xpsf,0)
    Ypsf, Xpsf = WHSC.wcs_world2pix(Rapsf, Decpsf,0)


    PSF_HST, PSF_HSC = tools.get_psf(FHST, FHSC,xpsf,ypsf, WHST, WHSC, np1)

    np1, np2 = PSF_HST.shape

    # Get the target PSF to partially deconvolve the image psfs
    target_psf, r, t = scarlet.psf_match.fit_target_psf(PSF_HST.reshape(1,np1,np2), scarlet.psf_match.moffat)
    # Display the target PSF

    # Match each PSF to the target PSF
    diff_kernels, psf_blend = scarlet.psf_match.build_diff_kernels(PSF_HSC.reshape(1,np1,np2), PSF_HST.reshape(1,np1, np2), l0_thresh=0.000001)
    PSF_HSC = diff_kernels[0,:,:]
    PSF_HSC /= np.sum(PSF_HSC)



    hdus = fits.PrimaryHDU(PSF_HST)
    lists = fits.HDUList([hdus])
    lists.writeto('../HSTC/PSF_HST_29.fits', clobber=True)

    hdus = fits.PrimaryHDU(PSF_HSC)
    lists = fits.HDUList([hdus])
    lists.writeto('../HSTC/PSF_HSC_29.fits', clobber=True)

    plt.subplot(121)
    plt.imshow(np.log10(PSF_HST), interpolation = 'nearest', cmap = 'inferno')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(np.log10(PSF_HSC), interpolation = 'nearest', cmap = 'inferno')
    plt.colorbar()
    plt.show()

else:
    PSF_HSC = fits.open('../HSTC/PSF_HSC_29.fits')[0].data

#HST coordinates
#########large###big##big##faint#####works


x0 = 12650 #6296  #6147 #6284  #12718#6726  #5388   #2675 #
y0 = 1080 #13632 #5964 #11195 #10187#13113 #14579  #4888 #
excess = 61

x_HST, y_HST, X_HSC, Y_HSC, X_HST, Y_HST = tools.match_patches(x0,y0,WHSC, WHST, excess)

cut_HST, cut_HSC = tools.make_patches(x_HST-6, y_HST-6, X_HSC, Y_HSC, FHST, FHSC)

n1, n2 = cut_HST.shape
N1, N2 = cut_HSC.shape

#cut_HST += np.random.randn(n1,n2)*tools.MAD(cut_HST)*5


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



compute = 1
if compute:
    print('Computing Low Resolution matrix')
    mat_HSC = tools.make_mat2D_fft(x_HST, y_HST, X_HST, Y_HST, PSF_HSC)

    hdus = fits.PrimaryHDU(mat_HSC)
    lists = fits.HDUList([hdus])
    lists.writeto('../HSTC/Mat_HSC_'+str(n1)+'.fits', clobber=True)


h = x_HST[1]-x_HST[0]
#HR_filter = tools.make_vec2D_fft(x_HST, y_HST,X_HST[np.int(N1*N2/2)], Y_HST[np.int(N1*N2/2)], PSF_HST, h).reshape(n1,n2)

if np.abs(compute-1):
    mat_HSC = fits.open('../HSTC/Mat_HSC_'+str(n1)+'.fits')[0].data


#def filter_HR(x):
#    return scarlet.transformation.Convolution(HR_filter).dot(x)#mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)
#def filter_HRT(x):
#    return scarlet.transformation.Convolution(HR_filter).T.dot(x)#mat_HST[:,np.int(n1*n1/2)].reshape(n1,n2)


#reg_HR = np.sum(HR_filter**2)**0.5

print('Noise in HST:', tools.MAD(cut_HST))
print('Noise in HSC:', tools.MAD(cut_HSC))


module_path='/Users/remy/Desktop/LSST_Project/scarlet_Pixelcnn/scarlet-pixelcnn/modules/pixelcnn_out'
pixelcnn = hub.Module(module_path)
sess= tf.Session()
sess.run(tf.global_variables_initializer())
x_nn = tf.placeholder(shape=(1,32,32,1), dtype=tf.float32)

out = pixelcnn(x_nn, as_dict=True)['grads']

def grad_nn(y):
    return sess.run(out, feed_dict={x_nn: y.reshape((1,32,32,1))})[0,:,:,0]

niter = 1000
nc = 2
Sall, SHR, SLR = tools.Combine2D_filter(cut_HST, cut_HSC, mat_HSC, niter, reg_nn = grad_nn, verbosity = 1, reg_HR =0)
#Sall = tools.Deblend2D_filter(cut_HST.flatten(), cut_HSC.flatten(), filter_HR, filter_HRT, mat_HSC, niter, nc, verbosity = 1)


#plt.imshow(Sall[0,:].reshape(n1,n2)); plt.show()
#plt.imshow(Sall[1,:].reshape(n1,n2)); plt.show()

hdus = fits.PrimaryHDU(Sall.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/HSTC_Sall_cnn.fits', clobber=True)
hdus = fits.PrimaryHDU(SHR.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/HSTC_SHR_cnn.fits', clobber=True)
hdus = fits.PrimaryHDU(SLR.reshape(n1,n2))
lists = fits.HDUList([hdus])
lists.writeto('../HSTC/HSTC_SLR_cnn.fits', clobber=True)

if 1:
    font = 25
    plt.figure(6)
    plt.imshow(Sall, interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.title('Joint Reconstruction', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/Joint_Reconstruction_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(12)
    plt.imshow(SLR, interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.title('HSC deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HSC-deconvolution_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(9)
    plt.imshow(SHR, interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.title('HST deconvolution', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HST_deconvolution_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(5)
    plt.imshow((Sall), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.title('HST joint model', fontsize = font)
    plt.axis('off')
    plt.savefig('Images/HST_joint_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(4)
    plt.imshow(np.dot(Sall.flatten(), mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'inferno')
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
    plt.imshow((SHR), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/HST_deconvolution_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(10)
    plt.title('HSC deconvolution model', fontsize = font)
    plt.imshow(np.dot(SLR.flatten(), mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/HSC_deconvolution_model_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(14)
    plt.title('Residual joint HST', fontsize = font)
    plt.imshow(cut_HST - (Sall), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/Residual_joint_HST_'+str(x0)+'_'+str(y0)+'.png')
    plt.figure(13)
    plt.title('Residual joint HSC', fontsize = font)
    plt.imshow(cut_HSC - np.dot(Sall.flatten(), mat_HSC).reshape(N1,N2), interpolation = 'none', cmap = 'inferno')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('Images/Residual_joint_HSC_'+str(x0)+'_'+str(y0)+'.png')
    plt.show()