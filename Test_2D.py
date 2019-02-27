import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
import sep
import pyfits as pf
import scarlet.display
import warnings
warnings.simplefilter("ignore")

def G(x, y, x0, y0, sigma):
    G = 1./(sigma*np.sqrt(2*np.pi)) * \
            np.exp(-((x-x0)**2.+(y-y0)**2)/(2*sigma**2))
    return G

def linormmat(S, nit):
    """
      Estimates the maximal eigen value of a matrix A

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: maximal eigen value

       EXAMPLES

    """

    n1, n2 = np.shape(S)
    x0 = np.random.rand(1, n1)
    x0 = x0 / np.sqrt(np.sum(x0 ** 2))

    for i in range(nit):
        x = np.dot(x0, S)
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = np.dot( xp, S.T)
        yn = np.sqrt(np.sum(y ** 2))

        if yn < np.dot(y, x0.T):
            break
        x0 = y / yn

    return 1./xn

def S(a, b, n, XY, s0):
    f = 0
    for i in range(n_blobs):
        f = f+ G(a, b, XY[0,i], XY[1,i], s0[0,i])*(np.random.rand(1)*2+1)
    return f

def s1(a, b):
    return G(a, b, -1, -3, 2)

def s2(a, b):
    return G(a, b, 3, 2, 2)

def p1(a, b):
    return G(a, b, 0, 0, 1.5)  # /np.sum(G(x,3.))

def p2(a, b):
    return G(a, b, 0, 0, 4.)

def sinc(a):
    if a == 0:
        return 1.
    else:
        return np.sin(a)/a

def f1(a, b):
    return 0.1*s1(a, b)+0.8*s2(a, b)

def f2(a, b):
    return 0.3*s1(a, b)+0.15*s2(a, b)

def f3(a, b):
    return 0.6*s1(a,b)+0.05*s2(a, b)

def Image(f, a, b):
    n1 = np.sqrt(a.size)
    Img = np.zeros((n1, n1))
    xgrid, ygrid = np.where(np.zeros((n1, n1)) == 0)

    Img[xgrid, ygrid] = f(a, b)
    return Img

print('building operator:')

def conv(xp, yp, xk, yk, xm, ym, p, xpp, ypp, h):
    # x: numpy array, high resolution sampling
    # X: numpy array, low resolution sampling
    # p: numpy array, psf sampled at high resolution
    # xm: scalar, location of sampling in thelow resolution grid
    Fm = 0

    for i in range(np.size(xp)):
        Fm += sinc((xm-(xk+xp[i]))/h)*sinc((ym-(yk+yp[i]))/h)*p[xpp[i], ypp[i]]
    return Fm*h/np.pi

def make_vec(a, b, xm, ym, p, xp, yp, xpp, ypp, h):
    vec = np.zeros(a.size)

    for k in range(np.size(a)):

        vec[k] = conv(xp, yp, a[k], b[k], xm, ym, p, xpp, ypp, h)

    return vec.flatten()

def make_mat(a, b, A, B, p, xp, yp, xpp,ypp):
    mat = np.zeros((a.size, B.size))
    h = a[1]-a[0]
    assert h!=0
    c = 0
    for m in range(np.size(B)):
            mat[:, c] = make_vec(a, b, A[m], B[m], p, xp, yp, xpp, ypp, h)
            c += 1
    return mat

def make_filter(a, b, A, B, p, xp, yp, xpp, ypp):
    ## Low resolution sampling
    H=A[1]-A[0]
    h=a[1]-a[0]
    xm0=np.array(A[len(A)/2])
    ym0=np.array(B[len(B)/2])

    ma=np.min(a)
    Ma=np.max(a)
    mb=np.min(b)
    Mb=np.max(b)
    N=len(A)
    n=len(a)
    #Extension of the sampling to account for sinc edges
    xx=np.linspace(ma-n, Ma+n, 3*n+1)
    yy=np.linspace(mb-n, Mb+n, 3*n+1)

    # On essaie, mais je crois que c'est xpp ypp en vrai
    xxp,yyp = np.where(np.zeros(((3*n+1)**0.5,(3*n+1)**0.5))==0)

    return make_vec(xx, yy, xm0, ym0, p, xx, yy, xpp, ypp, h)*(1-np.float(len(a))/len(xx))

def make_mat_alt(a, b, A, B, p, xp, yp, xpp, ypp):
    vec = make_filter(a, b, A, B, p, xp, yp, xpp, ypp)

    mat = np.zeros((a.size, A.size))
    n = np.size(vec)
    h =A[1]-A[0]
    print(vec.shape, mat.shape, n)
    for k in range(A.size):
        mat[:,k] = vec[n+n/2-k*h : 2*n+n/2-k*h]/h
    return mat

def interp(a, b, A, B, Fm):
    # x: High resolution target grid
    # X: Input grid
    # Fm: Samples at positions X
    hx = np.abs(A[1]-A[0])
    hy = np.abs(B[np.sqrt(B.size)+1] - B[0])
    print(hx,hy)
    return np.array([Fm[k] * np.sinc((a-A[k])/(hx)) * np.sinc((b-B[k])/(hy)) for k in range(len(A))]).sum(axis=0)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
n1, n2 = 31., 31.
N1, N2 = 15., 15.
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



xxp = np.linspace(-n1/2,n1/2,n1)
yyp = np.linspace(-n2/2,n2/2,n2)
xp, yp = np.meshgrid(xxp,yyp)
xp=xp.flatten()#reshape(1,x.size)
yp=yp.flatten()

colourHR = np.zeros((3, n1,n2))
colourHR[0,:,:] = Image(f1, x, y)
colourHR[1,:,:] = Image(f2, x, y)
colourHR[2,:,:] = Image(f3, x, y)
colourLR = np.zeros((3, N1,N2))
colourLR[0,:,:] = Image(f1, X, Y)
colourLR[1,:,:] = Image(f2, X, Y)
colourLR[2,:,:] = Image(f3, X, Y)
#+np.random.randn(N1, N2)*np.max(Image(f1, x, y))/10

n_blobs = 3
print('number of blobs: ', np.int(n_blobs))

xy = np.random.rand(2, n_blobs)*14-7
print('positions of blobs: ', xy)

sig = np.random.rand(1, n_blobs)*2.+0.5
print('width of blobs: ', sig)

def F(a,b):
    return S(a,b,n_blobs,xy,sig)

Im_HR = Image(F, x, y)
Im_LR = Image(F, X, Y)

S1 = Image(s1,x, y)
S2 = Image(s2,x, y)

Interp = interp(x, y, X, Y, Im_LR.reshape(N1*N2)).reshape((n1, n2))

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

PSF_HR = Image(p1,xp,yp)
PSF_LR = Image(p2,xp,yp)
print(PSF_HR.shape)
plt.subplot(121)
plt.imshow(PSF_HR)
plt.subplot(122)
plt.imshow(PSF_LR)
plt.show()

xp0,yp0 = np.where(PSF_HR*0==0)

print('Low resolution operator')
#mat_LR = make_mat(x, y, X, Y, PSF_LR, x, y, xp0, yp0)

#hdus = pf.PrimaryHDU(mat_LR)
#lists = pf.HDUList([hdus])
#lists.writeto('mat_LR2d.fits', clobber=True)

print('High resolution operator')
#mat_HR = make_mat(x, y, x, y, PSF_HR, x, y, xp0, yp0)

#hdus = pf.PrimaryHDU(mat_HR)
#lists = pf.HDUList([hdus])
#lists.writeto('mat_HR2d.fits', clobber=True)


mat_LR = pf.open('mat_LR2d.fits')[0].data
mat_HR = pf.open('mat_HR2d.fits')[0].data


Y0 = Image(F,x,y)

print('Now lets solve this!')

YLR = np.dot(Y0.flatten(), mat_LR)
YHR = np.dot(Y0.flatten(), mat_HR)

plt.subplot(131)
plt.imshow(Y0, interpolation = None, cmap = 'gist_stern')
plt.subplot(132)
plt.imshow(YHR.reshape(n1,n2), interpolation = None, cmap = 'gist_stern')
plt.subplot(133)
plt.imshow(YLR.reshape(N1,N2), interpolation = None, cmap = 'gist_stern')
plt.show()

sigma_HR = np.max(YHR)/20
sigma_LR = np.max(YLR)/200

YLR+=np.random.randn(YLR.size)*sigma_LR
YHR+=np.random.randn(YHR.size)*sigma_HR

var_norm = 1./sigma_HR**2 + 1./sigma_LR**2
wvar_HR = (1./sigma_HR**2)*(1./var_norm)
wvar_LR = (1./sigma_LR**2)*(1./var_norm)


niter = 10000
mu1 = linormmat(mat_LR, 20)/10.
mu2 = linormmat(mat_HR, 20)/10.
print(mu1, mu2)
Sall = np.zeros((x.size))
SHR = np.zeros((x.size))
SLR = np.zeros((x.size))
vec = np.zeros(niter)
vec2 = np.zeros(niter)
vec3 = np.zeros(niter)
for i in range(niter):
    if i % 1000+1 == True:
        print(i)
    Sall += mu2 * np.dot(YLR - np.dot(Sall, mat_LR), mat_LR.T)*wvar_LR + \
        mu1 * np.dot(YHR-np.dot(Sall, mat_HR), mat_HR.T)*wvar_HR
    SLR += mu2 * np.dot(YLR - np.dot(SLR, mat_LR), mat_LR.T)
    SHR += mu1 * np.dot(YHR - np.dot(SHR, mat_HR), mat_HR.T)
    Sall[Sall < 0] = 0
    SLR[SLR < 0] = 0
    SHR[SHR < 0] = 0
    vec[i] = np.sum((YLR - np.dot(Sall, mat_LR))**2*wvar_LR) + np.sum((YHR-np.dot(Sall, mat_HR))**2*wvar_LR)
    vec2[i] = np.sum((YLR - np.dot(SLR, mat_LR))**2)
    vec3[i] = np.sum((YHR - np.dot(SHR, mat_HR))**2)

plt.plot(vec, 'b', label = 'All', linewidth = 2)
plt.plot(vec2, 'r', label = 'LR', linewidth = 3)
plt.plot(vec3, 'g', label = 'HR', linewidth = 4)
plt.show()

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

objects_truth = sep.extract(Im_HR, 3*sigma_HR, deblend_cont = 0.01)
objects_HR = sep.extract(SHR, 3*sigma_HR, deblend_cont = 0.01)
objects_LR = sep.extract(SLR, 3*sigma_HR, deblend_cont = 0.01)
objects_all = sep.extract(Sall, 3*sigma_HR, deblend_cont = 0.01)
print(objects_truth['x'],xy.T+15)
plt.plot(xy.T+15, 'kx', label = 'truth')
plt.plot(objects_truth['x'],objects_truth['y'], 'bx', label = 'Original')
plt.plot(objects_HR['x'],objects_HR['y'], 'cx', label = 'HR')
plt.plot(objects_LR['x'],objects_LR['y'], 'mx', label = 'LR')
plt.plot(objects_all['x'],objects_all['y'], 'rx', label = 'All')
plt.legend()
plt.axis([0,30,0,30])
plt.show()


stop
###################################################################################
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
