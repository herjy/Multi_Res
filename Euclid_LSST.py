import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scp
import scipy.ndimage as nd
import pyfits as pf
import SLIT
import MuSCADeT as wine
import scarlet.display
import sep
import pyfits as pf
import warnings
warnings.simplefilter("ignore")

def Image(f, a, b):
    n1 = np.sqrt(a.size)
    Img = np.zeros((n1, n1))
    xgrid, ygrid = np.where(np.zeros((n1, n1)) == 0)

    Img[xgrid, ygrid] = f(a, b)
    return Img

def linormA(S, nit):
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

def G(x, y, x0, y0, sigma):
    G = 1./(sigma*np.sqrt(2*np.pi)) * \
            np.exp(-((x-x0)**2.+(y-y0)**2)/(2*sigma**2))
    return G

def interp(a, b, A, B, Fm):
    # x: High resolution target grid
    # X: Input grid
    # Fm: Samples at positions X

    hx = np.abs(A[1]-A[0])
    hy = np.abs(B[np.sqrt(B.size)+1] - B[0])

    return np.array([Fm[k] * np.sinc((a-A[k])/(hx)) * np.sinc((b-B[k])/(hy)) for k in range(len(A))]).sum(axis=0)

HST0 = pf.open('IMAGES_diff_surveys/CANDELS_V_60mas.fits')[0].data
Euclid0 = pf.open('IMAGES_diff_surveys/Euclid_VI_100mas.fits')[0].data.astype(float)
LSST0 = pf.open('IMAGES_diff_surveys/LSST_V_200mas_full.fits')[0].data.astype(float)

sigma_HR = SLIT.tools.MAD(Euclid0)
sigma_HHR = SLIT.tools.MAD(HST0)
sigma_LR = SLIT.tools.MAD(LSST0)

PSF_Euclid0 = pf.open('PSFs_diff_surveys/f814w_flat_Tinytim_psf_Euclid_VI_100mas.fits')[0].data
ROT_Euclid = nd.rotate(PSF_Euclid0, -55)
PSF_LSST0 = pf.open('PSFs_diff_surveys/cos_f606w_psf_star1_LSST.fits')[0].data
PSF_HST = pf.open('PSFs_diff_surveys/f814w_flat_tinytim_psf_60mas.fits')[0].data

np1, np2 = 61,61#PSF_Euclid0.shape
Np1, Np2 = 31,31

xpsf_LR, ypsf_LR = 94, 36
xpsf_HR, ypsf_HR = 188, 71
psf_lsst = LSST0[xpsf_LR-np1/4:xpsf_LR+np1/4+1, ypsf_LR-np2/4:ypsf_LR+np2/4+1]
psf_euclid = Euclid0[xpsf_HR-np1/2:xpsf_HR+np1/2+1, ypsf_HR-np2/2:ypsf_HR+np2/2+1]


PSF_LSST, g = wine.MCA.mr_filter(psf_lsst, 20,5,sigma_LR, lvl = 4)
PSF_EUCLID, g = wine.MCA.mr_filter(psf_euclid, 20, 5, sigma_HR, lvl = 4)


plt.subplot(221)
plt.imshow(np.log10(psf_lsst))
plt.subplot(222)
plt.imshow(np.log10(PSF_LSST))
plt.subplot(223)
plt.imshow(np.log10(psf_euclid))
plt.subplot(224)
plt.imshow(np.log10(PSF_EUCLID))
plt.show()



def fg(x,y):
    return G(x, y, 29, 29, 3.)
#kernel = Image(fg, xp0, yp0)
#nd1, nd2 = np.shape(ROT_Euclid)[0]-np.shape(PSF_Euclid0)[0],np.shape(ROT_Euclid)[1]-np.shape(PSF_Euclid0)[1]

#PSF_Euclid = ROT_Euclid[nd1/2:-nd1/2, nd2/2:-nd2/2]
#PSF_LSST = scp.fftconvolve(PSF_Euclid, kernel, mode = "same")

n1, n2 = 30., 30.
N1, N2 = 15., 15.
xc, yc = 195, 245#306,197#332,318#133,367#195, 245#269,50#

Euclid = Euclid0[xc-n1/2:xc+n1/2, yc-n2/2:yc+n2/2]
LSST = LSST0[xc/2-N1/2+1:xc/2+N1/2+1, yc/2-N2/2+1:yc/2+N2/2+1]

hdus = pf.PrimaryHDU(Euclid)
lists = pf.HDUList([hdus])
lists.writeto('Image_HR.fits', clobber=True)
hdus = pf.PrimaryHDU(LSST)
lists = pf.HDUList([hdus])
lists.writeto('Image_LR.fits', clobber=True)

plt.subplot(121)
plt.imshow(Euclid, interpolation = 'none', cmap = 'gist_stern')
plt.subplot(122)
plt.imshow(LSST, interpolation = 'none', cmap = 'gist_stern')
plt.show()


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



PSF_Euclid = interp(xp,yp,xp+1,yp+1, PSF_EUCLID.reshape(np1*np2)).reshape(np1,np2)#+0.4-0.15
PSF_LSST = interp(xp,yp,Xp+2,Yp+1.5,PSF_LSST.reshape(Np1*Np2)).reshape(np1,np2)
#psf_LSST = G(xp,yp,0,0,10).reshape(np1,np2)
xp0, yp0 = np.where(np.zeros((np1,np2)) == 0)

PSF_LSST =PSF_LSST/np.sum(PSF_LSST)
PSF_Euclid /= np.sum(PSF_Euclid)

PSF_Euclid[PSF_Euclid<0] = 0
PSF_LSST[PSF_LSST<0] = 0

print('LSST max ', np.where(PSF_LSST == np.max(PSF_LSST)))
print('Euclid max ', np.where(PSF_Euclid == np.max(PSF_Euclid)))

hdus = pf.PrimaryHDU(PSF_Euclid)
lists = pf.HDUList([hdus])
lists.writeto('PSF_HR.fits', clobber=True)
hdus = pf.PrimaryHDU(PSF_LSST)
lists = pf.HDUList([hdus])
lists.writeto('PSF_LR.fits', clobber=True)



def img_func(img, x, y):
    return img[x,y]

def sinc(a):
    if a == 0:
        return 1.
    else:
        return np.sin(a)/a


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

print('Low resolution operator')
mat_LR = make_mat_alt(x, y, X, Y, PSF_LSST, xp, yp, xp0, yp0)

hdus = pf.PrimaryHDU(mat_LR)
lists = pf.HDUList([hdus])
lists.writeto('mat_LR_LSST_30_alt.fits', clobber=True)

mat_LR0 = pf.open('mat_LR_LSST_30_wave.fits')[0].data
plt.subplot(131)
plt.imshow(mat_LR0); plt.colorbar()
plt.subplot(132)
plt.imshow(mat_LR); plt.colorbar()
plt.subplot(133)
plt.imshow(mat_LR0-mat_LR); plt.colorbar()
plt.show()

print('High resolution operator')
mat_HR = make_mat_alt(x, y, x, y, PSF_Euclid, xp, yp, xp0, yp0)

hdus = pf.PrimaryHDU(mat_HR)
lists = pf.HDUList([hdus])
lists.writeto('mat_HR_LSST_30_alt.fits', clobber=True)

mat_HR0 = pf.open('mat_HR_LSST_30_wave.fits')[0].data

plt.subplot(131)
plt.imshow(mat_HR0); plt.colorbar()
plt.subplot(132)
plt.imshow(mat_HR); plt.colorbar()
plt.subplot(133)
plt.imshow(mat_HR0-mat_HR); plt.colorbar()
plt.show()



print('Now lets solve this!')


print(sigma_HR, sigma_LR, sigma_HHR)


def Solve(HR, LR, matHR, matLR, niter, verbosity = 0):

    n = HR.size
    N = LR.size
    sigma_HR = SLIT.tools.MAD(HR.reshape(n**0.5, n**0.5))
    sigma_HHR = SLIT.tools.MAD(LR.reshape(N**0.5, N**0.5))

    var_norm = 1./sigma_HR**2 + 1./sigma_LR**2
    wvar_HR = (1./sigma_HR**2)*(1./var_norm)
    wvar_LR = (1./sigma_LR**2)*(1./var_norm)

    mu1 = linormA(mat_HR, 10)/1.
    mu2 = linormA(mat_LR, 10)
    print(mu1, mu2)
    Sa = np.zeros((HR.size))
    SH = np.zeros((HR.size))
    SL = np.zeros((HR.size))
    vec = np.zeros(niter)
    vec2 = np.zeros(niter)
    vec3 = np.zeros(niter)
    for i in range(niter):
        if i % 1000+1 == True:
            print(i)
        Sa += mu2 * np.dot(LR - np.dot(Sa, matLR), matLR.T)*wvar_HR \
            + mu1 * np.dot(HR-np.dot(Sa, matHR), matHR.T)*wvar_LR
    #plt.imshow(Sall.reshape(n1,n2)); plt.savefig('fig'+str(i))

        SL += mu2 * np.dot(LR - np.dot(SL, matLR), matLR.T)
        if i < 10000:
            SH += mu1 * np.dot(HR - np.dot(SH, matHR), matHR.T)
            SH[SH < 0] = 0
        Sa[Sa < 0] = 0
        SL[SL < 0] = 0

        vec[i] = np.sum((LR - np.dot(Sa, matLR))**2*wvar_LR) + np.sum((HR-np.dot(Sa, matHR))**2*wvar_LR)
        vec2[i] = np.sum((LR - np.dot(SL, matLR))**2)
        vec3[i] = np.sum((HR - np.dot(SH, matHR))**2)
    #    plt.subplot(121)
    #    plt.imshow((LR - np.dot(Sall, matLR)).reshape(N1,N2))
    #    plt.subplot(122)
    #    plt.imshow((HR - np.dot(Sall, matHR)).reshape(n1,n2))
    #    plt.show()
    if verbosity == 1:
        plt.plot(vec, 'b', label = 'All', linewidth = 2)
        plt.plot(vec2, 'r', label = 'LR', linewidth = 3)
        plt.plot(vec3, 'g', label = 'HR', linewidth = 4)
        plt.show()

    return Sa, SH, SL

Y_LR = LSST.flatten()
Y_HR = Euclid.flatten()
Niter = 30000
Sall, SHR, SLR = Solve(Y_HR, Y_LR, mat_HR, mat_LR, Niter)

if 1:
    plt.subplot(331)
    plt.imshow(Sall.reshape(n1,n2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.title('S all')
    plt.axis('off')
    plt.subplot(332)
    plt.imshow(SLR.reshape(n1,n2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.title('S LR')
    plt.axis('off')
    plt.subplot(333)
    plt.imshow(SHR.reshape(n1,n2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.title('S HR')
    plt.axis('off')
    plt.subplot(334)
    plt.title('LR')
    plt.imshow(LSST, interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(335)
    plt.title('HR')
    plt.imshow(Euclid, interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(336)
    plt.title('Residual HR HR')
    plt.imshow((Y_HR - np.dot(SHR, mat_HR)).reshape(n1,n2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(337)
    plt.title('Residual LR')
    plt.imshow((Y_LR - np.dot(Sall, mat_LR)).reshape(N1,N2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(338)
    plt.title('Residual HR')
    plt.imshow((Y_HR - np.dot(Sall, mat_HR)).reshape(n1,n2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
    plt.subplot(339)
    plt.title('Residual LR LR')
    plt.imshow((Y_LR - np.dot(SLR, mat_LR)).reshape(N1,N2), interpolation = 'none', cmap = 'gist_stern')
    plt.colorbar()
    plt.axis('off')
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
        hdus = pf.PrimaryHDU(Sall_cube)
        lists = pf.HDUList([hdus])
        lists.writeto('All_cube.fits', clobber=True)
        hdus = pf.PrimaryHDU(SHR_cube)
        lists = pf.HDUList([hdus])
        lists.writeto('HR_cube.fits', clobber=True)
        hdus = pf.PrimaryHDU(SLR_cube)
        lists = pf.HDUList([hdus])
        lists.writeto('LR_cube.fits', clobber=True)
        c+=1

hdus = pf.PrimaryHDU(Solution_all)
lists = pf.HDUList([hdus])
lists.writeto('Euclid_Sall.fits', clobber=True)
hdus = pf.PrimaryHDU(Solution_HR)
lists = pf.HDUList([hdus])
lists.writeto('Euclid_SHR.fits', clobber=True)
hdus = pf.PrimaryHDU(Solution_LR)
lists = pf.HDUList([hdus])
lists.writeto('Euclid_SLR.fits', clobber=True)
