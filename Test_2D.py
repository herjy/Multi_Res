import numpy as np
import matplotlib.pyplot as plt
import sep
import tools
import pyfits as pf
import scarlet
import warnings
warnings.simplefilter("ignore")

def G(x, y, x0, y0, sigma):
    G = 1./(sigma*np.sqrt(2*np.pi)) * \
            np.exp(-((x-x0)**2.+(y-y0)**2)/(2*sigma**2))
    return G

def S(a, b, n, XY, s0):
    f = 0
    for i in range(n_blobs):
        f = f+ G(a, b, XY[0,i], XY[1,i], s0[0,i])*(np.random.rand(1)*2+1)
    return f

def p1(a, b):
    return G(a, b, 0, 0, 1.5)  # /np.sum(G(x,3.))

def p2(a, b):
    return G(a, b, 0, 0, 4.)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
n1, n2 = 31, 31
N1, N2 = 15, 15
gridHR = np.zeros((n1, n2))
gridLR = np.zeros((N1, N2))
#x, y = np.where(gridHR == 0)
xx = np.linspace(-n1/2,n1/2,n1)
yy = np.linspace(-n2/2,n2/2,n2)
x, y = np.meshgrid(xx,yy)
x=x.flatten()#reshape(1,x.size)
y=y.flatten()#reshape(1,y.size)

#X, Y = np.where(gridLR == 0)
XX = np.linspace(-n1/2,n1/2,N1)
YY = np.linspace(-n2/2,n2/2,N2)
X,Y = np.meshgrid(XX,YY)
X=X.flatten()#reshape(1,X.size)
Y=Y.flatten()#reshape(1,Y.size)

xxp = np.linspace(-n1/4,n1/4,n1)
yyp = np.linspace(-n2/4,n2/4,n2)
xp, yp = np.meshgrid(xxp,yyp)
xp=xp.flatten()#reshape(1,x.size)
yp=yp.flatten()

n_blobs = 5
print('number of blobs: ', np.int(n_blobs))

xy = np.random.rand(2, n_blobs)*16-8
print('positions of blobs: ', xy)

sig = np.random.rand(1, n_blobs)*2.+0.5
print('width of blobs: ', sig)

def F(a,b):
    return S(a,b,n_blobs,xy,sig)

Im_HR = tools.Image(F, x, y)
Im_LR = tools.Image(F, X, Y)

Interp = tools.interp2D(x, y, X, Y, Im_LR.reshape(N1*N2)).reshape((n1, n2))

plt.subplot(221)
plt.imshow(Im_HR, interpolation=None, cmap='gist_stern',
           vmin=np.min(Im_HR), vmax=np.max(Im_HR))
plt.colorbar()
plt.subplot(222)
plt.imshow(Im_LR, interpolation=None, cmap='gist_stern',
           vmin=np.min(Im_HR), vmax=np.max(Im_HR))
plt.colorbar()
plt.subplot(223)
plt.imshow(Interp, interpolation=None, cmap='gist_stern',
           vmin=np.min(Im_HR), vmax=np.max(Im_HR))
plt.colorbar()
plt.subplot(224)
plt.imshow(Im_HR-Interp, interpolation=None, cmap='gist_stern')
plt.colorbar()
plt.show()

PSF_HR = tools.Image(p1,xp,yp)
PSF_LR = tools.Image(p2,xp,yp)
print(PSF_HR.shape)
plt.subplot(121)
plt.imshow(PSF_HR)
plt.subplot(122)
plt.imshow(PSF_LR)
plt.show()

xp0,yp0 = np.where(PSF_HR*0==0)

print('Low resolution operator')
#mat_LR = tools.make_mat2D(x, y, X, Y, PSF_LR, x, y, xp0, yp0)

#hdus = pf.PrimaryHDU(mat_LR)
#lists = pf.HDUList([hdus])
#lists.writeto('mat_LR2d_PSFHR.fits', clobber=True)

print('High resolution operator')
#mat_HR = tools.make_mat2D(x, y, x, y, PSF_HR, x, y, xp0, yp0)

#hdus = pf.PrimaryHDU(mat_HR)
#lists = pf.HDUList([hdus])
#lists.writeto('mat_HR2d_PSFHR.fits', clobber=True)


mat_LR = pf.open('mat_LR2d_PSFHR.fits')[0].data
mat_HR = pf.open('mat_HR2d_PSFHR.fits')[0].data

Y0 = tools.Image(F,x,y)

x1 = mat_HR[:,np.int(n1*n1/2)]#np.concatenate(( mat_LR[-1,:], mat_HR[0,1:]))#

print(x1.size)

n0 = x1.size
n = np.int(np.sqrt(x1.size))**2

#x1 = x1[(n0-n)/2:(n-n0)/2]

xx = scarlet.transformation.LinearFilter(x1.reshape(N1,N1))
res = xx.dot(Y0)

plt.plot(res); plt.show()
plt.imshow((res), cmap = 'gist_stern', interpolation = 'None'); plt.show()


print('Now lets solve this!')

YLR = np.dot(Y0.flatten(), mat_LR)
YHR = np.dot(Y0.flatten(), mat_HR)

plt.subplot(131)
plt.imshow(Y0, interpolation = None, cmap = 'gist_stern')
plt.subplot(132)
plt.imshow(YHR.reshape(n1,n2), interpolation = None, cmap = 'gist_stern')
plt.subplot(133)
plt.imshow(YLR.reshape(N1,N2)-res, interpolation = None, cmap = 'gist_stern')
plt.show()


SNR_HR = 5
SNR_LR = 1000
sigma_HR = np.sqrt(np.sum(YHR**2)/SNR_HR/(n1*n2))
sigma_LR = np.sqrt(np.sum(YLR**2)/SNR_LR/(N1*N2))

print('sigma: ', sigma_HR, sigma_LR)

YLR+=np.random.randn(YLR.size)*sigma_LR
YHR+=np.random.randn(YHR.size)*sigma_HR

niter = 10000

def filter_HR(x):
    return scarlet.transformation.LinearFilter(mat_HR[n1*n1/2,:].reshape(n1,n2)).dot(x)
def filter_LR(x):
    return scarlet.transformation.LinearFilter(mat_LR[n1*n1/2,:].reshape(N1,N2)).dot(x)
def filter_HRT(x):
    return scarlet.transformation.LinearFilter(mat_HR[:,n1*n1/2].reshape(n1,n2).T).dot(x)
def filter_LRT(x):
    return scarlet.transformation.LinearFilter(mat_LR[:,N1*N1/2].reshape(n1,n2).T).dot(x)

Sall, SHR, SLR = tools.Combine2D_filter(YHR, YLR, filter_HR, filter_HRT, filter_LR, filter_LRT, niter, verbosity = 0)
Sall, SHR, SLR = tools.Combine2D(YHR, YLR, mat_HR, mat_LR, niter, verbosity = 1)

Sall = Sall.reshape(n1,n2)
SLR = SLR.reshape(n1,n2)
SHR = SHR.reshape(n1,n2)

plt.subplot(336)
plt.imshow(Sall, interpolation = 'none', cmap = 'CMRmap')
plt.title('S all')
plt.axis('off')
plt.subplot(335)
plt.imshow(SLR, interpolation = 'none', cmap = 'CMRmap')
plt.title('S LR')
plt.axis('off')
plt.subplot(334)
plt.imshow(SHR, interpolation = 'none', cmap = 'CMRmap')
plt.title('S HR')
plt.axis('off')
plt.subplot(331)
plt.title('HR')
plt.axis('off')
plt.imshow(YHR.reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
plt.subplot(332)
plt.title('LR')
plt.axis('off')
plt.imshow(YLR.reshape(N1,N2), interpolation = 'none', cmap = 'CMRmap')
plt.subplot(333)
plt.imshow(Y0, interpolation = 'none', cmap = 'CMRmap')
plt.title('Truth')
plt.axis('off')
plt.subplot(338)
plt.imshow((YLR - np.dot(Sall.reshape(n1*n2), mat_LR)).reshape(N1,N2), interpolation = 'none', cmap = 'CMRmap')
plt.title('Residuals LR')
plt.axis('off')
plt.subplot(337)
plt.imshow((YHR - np.dot(Sall.reshape(n1*n2), mat_HR)).reshape(n1,n2), interpolation = 'none', cmap = 'CMRmap')
plt.title('Residuals HR')
plt.axis('off')
plt.show()

objects_truth = sep.extract(Y0, sigma_LR/2)#, deblend_cont = 0.01)
objects_HR = sep.extract(SHR, sigma_LR/2)#, deblend_cont = 0.01)
objects_LR = sep.extract(SLR, sigma_LR/2)#, deblend_cont = 0.01)
objects_all = sep.extract(Sall, sigma_LR/2)#, deblend_cont = 0.01)

xc,yc = np.meshgrid(np.linspace(0,n1-1,n1),np.linspace(0,n2-1,n2))
plt.imshow(Y0, interpolation = None, cmap = 'gray')
plt.contour(xc, yc, Sall, 5, cmap = 'Reds')
plt.contour(xc, yc, SLR, 5, cmap = 'Greens')
plt.contour(xc, yc, SHR, 5, cmap = 'Blues')
plt.plot(xy[0,:]+15, xy[1,:]+15, 'xk', label = 'truth', ms = 10, mew = 5)
plt.plot(objects_truth['x'],objects_truth['y'], 'xc', label = 'Original', ms = 10, mew = 5)
plt.plot(objects_HR['x'],objects_HR['y'], 'xb', label = 'HR', ms = 10, mew = 5)
plt.plot(objects_LR['x'],objects_LR['y'], 'xg', label = 'LR', ms = 10, mew = 5)
plt.plot(objects_all['x'],objects_all['y'], 'xr', label = 'All', ms = 10, mew = 5)
plt.legend()
plt.axis([0,30,0,30])
plt.show()


stop
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
def s1(a, b):
    return G(a, b, -1, -3, 2)

def s2(a, b):
    return G(a, b, 3, 2, 2)

def f1(a, b):
    return 0.1*s1(a, b)+0.8*s2(a, b)

def f2(a, b):
    return 0.3*s1(a, b)+0.15*s2(a, b)

def f3(a, b):
    return 0.6*s1(a,b)+0.05*s2(a, b)


Y_LR = np.zeros((3, X.size))
Y_LR[0, :] = np.dot(f1(x, y), mat_LR)
Y_LR[1, :] = np.dot(f2(x, y), mat_LR)
Y_LR[2, :] = np.dot(f3(x, y), mat_LR)

Y_HR = np.zeros((3, x.size))
Y_HR[0, :] = np.dot(f1(x, y), mat_HR)
Y_HR[1, :] = np.dot(f2(x, y), mat_HR)
Y_HR[2, :] = np.dot(f3(x, y), mat_HR)

plt.imshow(Y_HR[2,:].reshape(n1,n2)); plt.show()

A = np.zeros((3, 2))
A[0, :] = 0.1, 0.8
A[1, :] = 0.3, 0.15
A[2, :] = 0.6, 0.05

hdus = pf.PrimaryHDU(Y_HR.reshape(3,n1,n2))
lists = pf.HDUList([hdus])
lists.writeto('Y3d_HR.fits', clobber=True)
hdus = pf.PrimaryHDU(Y_LR.reshape(3,N1,N2))
lists = pf.HDUList([hdus])
lists.writeto('Y3d_LR.fits', clobber=True)

#Y_LR = pf.open('Y_LR.fits')[0].data
#Y_HR = pf.open('Y_HR.fits')[0].data

#Y_LR+=np.random.randn(3,X.size)*np.max(Y_LR)/100
#Y_HR+=np.random.randn(3,x.size)*np.max(Y_HR)/50
#print(mat_LR.shape)

#Display
# Use Asinh scaling for the images
norm = scarlet.display.Asinh(img=Y_HR.reshape(3,n1,n2), Q=20)
# Map i,r,g -> RGB
filter_indices = [2,1,0]
# Convert the image to an RGB image
img_colourHR = scarlet.display.img_to_rgb(colourHR, filter_indices=filter_indices, norm=norm)
img_colourLR = scarlet.display.img_to_rgb(colourLR, filter_indices=filter_indices, norm=norm)
plt.subplot(221)
plt.imshow(img_colourHR)
plt.title('Original')
plt.subplot(222)
plt.imshow(img_colourLR)
plt.title('Original LR')
plt.subplot(223)
img_HR = scarlet.display.img_to_rgb(Y_HR.reshape(3,n1,n2), filter_indices=filter_indices, norm=norm)
plt.imshow(img_HR)
plt.title('high resolution')
plt.subplot(224)
img_LR = scarlet.display.img_to_rgb(Y_LR.reshape(3,N1,N2), filter_indices=filter_indices, norm=norm)
plt.imshow(img_LR)
plt.title('Low resolution')
plt.show()



muA1 = 0.0051
muA2 = 0.0051
S = np.zeros((2, x.size))
As = A+np.random.randn(3, 2)/2.
As /= np.sum(As, axis=0)

for j in range(10):
    print(j)
    for i in range(niter):

        S = S + mu1 * \
            np.dot(np.dot(As.T, Y_HR-np.dot(As, np.dot(S, mat_HR))), mat_HR.T)
        S = S + mu2 * \
            np.dot(np.dot(As.T, Y_LR - np.dot(As, np.dot(S, mat_LR))), mat_LR.T)
        S[S < 0] = 0
    As = As + muA1 * \
        np.dot(np.dot(Y_LR - np.dot(As, np.dot(S, mat_LR)), mat_LR.T), S.T)
    As = As + muA2 * \
        np.dot(np.dot(Y_HR - np.dot(As, np.dot(S, mat_HR)), mat_HR.T), S.T)
    As[As < 0] = 0
    As /= np.sum(As, axis=0)

print(A, As)



plt.imshow(np.reshape(S[0, :], (n1, n2)))
plt.colorbar()
plt.show()

plt.imshow(np.reshape(S[1, :], (n1, n2)))
plt.colorbar()
plt.show()


plt.imshow(np.reshape(S[0, :], (n1, n2))-S1)
plt.colorbar()
plt.show()

plt.imshow(np.reshape(S[1, :], (n1, n2))-S2)
plt.colorbar()
plt.show()


plt.show()
