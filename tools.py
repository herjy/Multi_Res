import numpy as np
import matplotlib.pyplot as plt
import MuSCADeT as wine
import scipy.signal as scp
import time
import scipy.ndimage.filters as med
import warnings
warnings.simplefilter("ignore")


def interp1D(a, A, Fm):
    # x: High resolution target grid
    # X: Input grid
    # Fm: Samples at positions X

    h=np.abs(A[1]-A[0])
    return np.array([Fm[k] * np.sinc((a-A[k])/h) for k in range(len(A))]).sum(axis=0)


def conv1D(a, xk, xm, p, xp, h):

    # x: numpy array, high resolution sampling
    # X: numpy array, low resolution sampling
    # p: numpy array, psf sampled at high resolution
    # xm: scalar, location of sampling in thelow resolution grid
    Fm=0

    for i, xi in enumerate(xp):
        Fm += p(xi)*np.sinc((xm-xk-xi)/h)

    return Fm   #* h/np.pi


def make_vec1D(a, xm, p, xp, h):

    vec=np.zeros(a.size)
    for k, xk in enumerate(a):
        vec[k]=conv1D(a, xk, xm, p, xp, h)

    return vec


def make_filter1D(a, A, p, xp):
    ## Low resolution sampling
    H=A[1]-A[0]
    h=a[1]-a[0]
    xm0=A[len(A)/2]
    m=np.min(a)
    M=np.max(a)
    n=len(a)
    #Extension of the sampling to account for sinc edges
    xx=np.linspace(m-n, M+n, 3*n+1)


    return make_vec1D(xx, xm0, p, xx, H)*(1-np.float(len(a))/len(xx))



def make_mat_alt1D(a,A,p,xp):
    vec = make_filter1D(a, A, p, xp)
    mat = np.zeros((a.size, A.size))
    n = len(a)
    h =A[1]-A[0]

    for k in range(A.size):

        mat[:,k] = vec[n+n/2-k*h : 2*n+n/2-k*h]/h
    return mat

def make_mat(a, A, p, xp):
    mat=np.zeros((a.size, A.size))
    H=np.abs(a[1]-a[0])
    for m, xm in enumerate(A):
        mat[:, m]=make_vec1D(a, xm, p, xp, H)
    return mat

def Spline1D(x):
    return 1./12.*(np.abs(x-2)**3-4*np.abs(x-1)**3+6*np.abs(x)**3-4*np.abs(x+1)**3+np.abs(x+2)**3)
########################2D##################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
def sinc2D(x,y):
    return np.sinc(x)*np.sinc(y)#np.sinc(np.sqrt(x**2+y**2))

def Spline2D(x, y, x0, y0):
    return Spline1D(np.sqrt((x-x0)**2+(y-y0)**2))

def Image(f, a, b):
    n1 = np.int(np.sqrt(a.size))
    n2 = np.int(np.sqrt(b.size))
    Img = np.zeros((n1, n2))
    xgrid, ygrid = np.where(np.zeros((n1, n2)) == 0)

    Img[xgrid, ygrid] = f(a, b)
    return Img

def interp2D(a, b, A, B, Fm):
    # x: High resolution target grid
    # X: Input grid
    # Fm: Samples at positions X
    hx = np.abs(A[1]-A[0])
    hy = np.abs(B[np.int(np.sqrt(B.size))+1] - B[0])

    return np.array([Fm[k] * sinc2D((a-A[k])/(hx),(b-B[k])/(hy)) for k in range(len(A))]).sum(axis=0)#np.sinc((a-A[k])/(hx)) * np.sinc((b-B[k])/(hy))


def conv2D_fft(xk, yk, xm, ym, p, h):
    # x: numpy array, high resolution sampling
    # X: numpy array, low resolution sampling
    # p: numpy array, psf sampled at high resolution
    # xm: scalar, location of sampling in thelow resolution grid

    ker = np.zeros((np.int(xk.size**0.5), np.int(yk.size**0.5)))
    x,y = np.where(ker == 0)
    ker[x,y] = sinc2D((xm-xk)/h,(ym-yk)/h)#np.sinc((xm-(xk))/h)*np.sinc((ym-(yk))/h)

    return scp.fftconvolve(ker, p, mode = 'same')*h/np.pi

def make_vec2D_fft(a, b, xm, ym, p, h):

    vec = conv2D_fft(a, b, xm, ym, p, h)

    return vec.flatten()

def make_mat2D_fft(a, b, A, B, p):
    mat = np.zeros((a.size, B.size))
    h = a[1]-a[0]
    assert h!=0
    t0 = time.clock()
    for m in range(np.size(B)):
            #if (m % 1000+1 == True):
            #    print('Matrix line: ', m, ' out of ', np.size(B))
            #    print('time: ', time.clock()-t0)
            mat[:, m] = make_vec2D_fft(a, b, A[m], B[m], p, h)
            mat[:, m] /= np.sum(mat[:,m])

    return mat



def linorm2D(S, nit):
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
        y = np.dot(xp, S.T)
        yn = np.sqrt(np.sum(y ** 2))

        if yn < np.dot(y, x0.T):
            break
        x0 = y / yn

    return 1./xn

def MAD(x, shape = (0,0),n=3):
    ##DESCRIPTION:
    ##  Estimates the noise standard deviation from Median Absolute Deviation
    ##
    ##INPUTS:
    ##  -x: a 2D image for which we look for the noise levels.
    ##
    ##OPTIONS:
    ##  -n: size of the median filter. Default is 3.
    ##
    ##OUTPUTS:
    ##  -S: the source light profile.
    ##  -FS: the lensed version of the estimated source light profile

    if shape != (0,0):
        n = np.min(shape)
        x = x[:n, :n]

    xw = wine.wave_transform(x, np.int(np.log2(x.shape[0])))[0,:,:]
    meda = med.median_filter(xw,size = (n,n))
    medfil = np.abs(xw-meda)#np.median(x))
    sh = np.shape(xw)
    sigma = 1.48*np.median((medfil))
    return sigma


def get_psf(FHR, FLR,x0HR,y0HR, WHR, WLR, n):
    assert len(x0HR)==len(y0HR)

    PSFHR = 0.
    PSFLR = 0.

    for i in range(len(x0HR)):

        FHRcopy = np.copy(FHR*0)
        FHRcopy[x0HR[i]-25:x0HR[i]+25, y0HR[i]-25:y0HR[i]+25] = FHR[x0HR[i]-25:x0HR[i]+25, y0HR[i]-25:y0HR[i]+25]
        xm, ym = np.where(FHRcopy == np.max(FHRcopy))

        Ra0, Dec0 = WHR.all_pix2world(ym, xm, 0)
        y0LR, x0LR = WLR.all_world2pix(Ra0, Dec0, 0)
        x0LR, y0LR = np.int(x0LR), np.int(y0LR)

        FLRcopy = np.copy(FLR*0)
        FLRcopy[x0LR-25:x0LR+25, y0LR-25:y0LR+25] = FLR[x0LR-25:x0LR+25, y0LR-25:y0LR+25]
        Xm, Ym = np.where(FLRcopy == np.max(FLRcopy))

        xHR, yHR, XLR, YLR, XHR, YHR = match_patches(xm, ym, WLR, WHR, np.int(n/2)+1)

        n1 = np.int(xHR.size**0.5)
        N1 = np.int(XLR.size**0.5)

        Star_LR = FLR[np.int(Xm-N1/2):np.int(Xm+N1/2),np.int(Ym-N1/2):np.int(Ym+N1/2)]
        Star_HR = FHR[np.int(xm - n1 / 2):np.int(xm + n1 / 2), np.int(ym - n1 / 2):np.int(ym + n1 / 2)]


        sigmaLR = MAD(Star_LR, n=3)
        sigmaHR = MAD(Star_HR, n=3)
        PSFtLR = wine.MCA.mr_filter(Star_LR, 20, 5, sigmaLR, lvl = np.int(np.log2(N1-1)))[0]
        PSFtHR = wine.MCA.mr_filter(Star_HR, 20, 5, sigmaHR, lvl=np.int(np.log2(n1)))[0]

        PSFtLR = interp2D(xHR.flatten(),yHR.flatten(), XHR.flatten()-1, YHR.flatten()-1, PSFtLR.flatten()).reshape(n1,n1)

        PSFLR += PSFtLR / np.sum(PSFtLR)
        PSFHR += PSFtHR / np.sum(PSFtHR)

    PSFHR[PSFHR < 0] = 0
    PSFLR[PSFLR < 0] = 0
    return PSFHR/np.sum(PSFHR), PSFLR/np.sum(PSFLR)


def linorm2D_filter(filter, filterT, shape, nit):
    """
      Estimates the maximal eigen value of a matrix A

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: maximal eigen value

       EXAMPLES

    """

    n1 = shape
    x0 = np.random.rand(1, n1)
    x0 = x0 / np.sqrt(np.sum(x0 ** 2))

    for i in range(nit):
        x = filter(x0)
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = filterT(xp)
        yn = np.sqrt(np.sum(y ** 2))

        if yn < np.dot(y, x0.T):
            break
        x0 = y / yn

    return 1./xn



def Combine2D_filter(HR, LR, matLR, niter, reg_nn = None, verbosity = 0, reg_HR = 0):


    n1, n2 = np.shape(HR)
    N1, N2 = np.shape(LR)
    sigma_HR = MAD(HR.reshape(n1,n2), shape = (n1,n2))
    sigma_LR = MAD(LR.reshape(N1,N2), shape = (N1,N2))

    reg_LR = np.mean(np.sum(matLR ** 2, axis=1) ** 0.5)

    var_norm = 1./sigma_HR**2 + 1./sigma_LR**2
    wvar_HR = (1./sigma_HR**2)*(1./var_norm)
    wvar_LR = (1./sigma_LR**2)*(1./var_norm)

    mu1 = 1./1.#linorm2D_filter(filter_HR, filter_HRT, HR.size, 10)
    mu2 = linorm2D(matLR, 10)
    mu = 2./(1./mu1+1./mu2)
    print(mu1, mu2, mu)

    if reg_HR >0:

        thresh = sigma_HR*reg_HR + sigma_LR*reg_LR

    LR = LR.flatten()
    HR = HR.flatten()

    Sa = np.ones((n1,n2))*sigma_HR
    SH = np.ones((n1,n2))*sigma_HR
    SL = np.ones((n1,n2))*sigma_HR
    vec = np.zeros(niter)
    vec2 = np.zeros(niter)
    vec3 = np.zeros(niter)
    t0 =time.clock()
    for i in range(niter):
        if (i % 100+1 == True) and (verbosity == 1):
            print('Current iteration: ', i, ', time: ', time.clock()-t0)

            plt.imshow((LR - np.dot(Sa.flatten(), matLR)).reshape(N1,N2))

            plt.savefig('Residuals_HSC_'+str(i)+'.png')
        if reg_nn != None:
            S = np.sum(Sa)
            if i == 0:
                xm, ym = n1/2, n2/2
            else:
                xm, ym = np.where(Sa==np.max(Sa))
                xm = np.min([np.max([xm, 17]), n1 - 17])
                ym = np.min([np.max([ym, 17]), n2 - 17])
            Sa_nn = reg_nn(Sa[np.int(xm-16):np.int(xm+16), np.int(ym-16):np.int(ym+16)]/S)
        Sa += mu * (np.dot(LR - np.dot(Sa.flatten(), matLR), matLR.T)*wvar_LR + mu*(HR-Sa.flatten())*wvar_HR).reshape(n1,n2)#mu * filter_HRT(HR-filter_HR(Sa.reshape(n1,n2))).reshape(n)*wvar_HR
        if reg_nn != None:

            Sa[np.int(xm-16):np.int(xm+16), np.int(ym-16):np.int(ym+16)] -= 1e-6*Sa_nn*S
            Sa[:np.int(xm - 16), :] = 0
            Sa[np.int(xm + 16):,:] = 0
            Sa[:, :np.int(ym - 16)] = 0
            Sa[:, np.int(ym + 16):] = 0


        if reg_HR >0:

            Sa, jk = wine.MCA.mr_filter(Sa, 20, 5, thresh)
        if reg_nn != None:
            S = np.sum(SL)
            if i == 0:
                xm, ym = n1/2, n2/2
            else:
                xm, ym = np.where(SL == np.max(SL))
                xm = np.min([np.max([xm, 17]), n1 - 17])
                ym = np.min([np.max([ym, 17]), n2 - 17])

            SL_nn = reg_nn(SL[np.int(xm-16):np.int(xm+16), np.int(ym-16):np.int(ym+16)]/S)
        SL += mu2 * np.dot(LR - np.dot(SL.flatten(), matLR), matLR.T).reshape(n1,n2)
        if reg_nn != None:

            SL[np.int(xm-16):np.int(xm+16), np.int(ym-16):np.int(ym+16)] -= 1e-6 * SL_nn*S
            SL[:np.int(xm - 16), :] = 0
            SL[np.int(xm + 16):, :] = 0
            SL[:,:np.int(ym - 16)] = 0
            SL[:, np.int(ym + 16):] = 0


        if i < 10000:
            if reg_nn != None:
                S = np.sum(SH)
                if i == 0:
                    xm, ym = n1 / 2, n2 / 2
                else:
                    xm, ym = np.where(SH == np.max(SH))
                xm = np.min([np.max([xm,17]),n1-17])
                ym = np.min([np.max([ym, 17]), n2 - 17])
                SH_nn = reg_nn(SH[np.int(xm - 16):np.int(xm + 16), np.int(ym - 16):np.int(ym + 16)]/S)

            SH += mu1 * (HR-SH.flatten()).reshape(n1,n2)
            if reg_nn != None:

                SH[np.int(xm - 16):np.int(xm + 16), np.int(ym - 16):np.int(ym + 16)] -= 1e-6 * SH_nn*S
                SH[:np.int(xm - 16), :] = 0
                SH[np.int(xm + 16):,:] = 0
                SH[:, :np.int(ym - 16)] = 0
                SH[:, np.int(ym + 16):] = 0
            #SH[SH < 0] = 0
        Sa[Sa < 0] = 0
        SL[SL < 0] = 0



        #Sa /= np.sum(Sa) / (n1*n2)
        #SL /= np.sum(SL) / (n1 * n2)
        #SH /= np.sum(SH) / (n1 * n2)

        vec[i] = np.std(LR - np.dot(Sa.flatten(), matLR))**2/2./sigma_LR**2 + np.std(HR-Sa.flatten())**2*wvar_LR/2./sigma_HR**2
        vec2[i] = np.std(LR - np.dot(SL.flatten(), matLR))**2/sigma_LR**2
        vec3[i] = np.std(HR - (SH.flatten()))**2/sigma_HR**2
    #    plt.subplot(121)
    #    plt.imshow((LR - np.dot(Sall, matLR)).reshape(N1,N2))
    #    plt.subplot(122)
    #    plt.imshow((HR - np.dot(Sall, matHR)).reshape(n1,n2))
    #    plt.show()
    if verbosity == 1:
        plt.show()
        plt.plot(vec3, 'b', label='HR', linewidth=4)
        plt.plot(vec2, 'g', label='LR', linewidth=3)
        plt.plot(vec, 'r', label = 'All', linewidth = 2)


        plt.show()

    return Sa, SH, SL



def Deblend_cnn(HR, LR, matLR, coord, niter, reg_nn, n_patch = (32,32), verbosity = 0):


    n1, n2 = np.shape(HR)
    N1, N2 = np.shape(LR)


    y0,x0 = coord
    n = x0.size
    assert n == y0.size

    sigma_HR = MAD(HR.reshape(n1,n2), shape = (n1,n2))
    sigma_LR = MAD(LR.reshape(N1,N2), shape = (N1,N2))

    var_norm = 1./sigma_HR**2 + 1./sigma_LR**2
    wvar_HR = (1./sigma_HR**2)*(1./var_norm)
    wvar_LR = (1./sigma_LR**2)*(1./var_norm)

    mu1 = 1./100.#linorm2D_filter(filter_HR, filter_HRT, HR.size, 10)
    mu2 = linorm2D(matLR, 10)/1.
    mu = 2./(1./mu1+1./mu2)
    print(mu1, mu2, mu)

    LR = LR.flatten()
    HR = HR.flatten()

    np1, np2 = n_patch
    Sa = np.random.randn(n,n1,n2)*sigma_HR
    SHR = np.random.randn(n,n1,n2)*sigma_HR
    SLR = np.random.randn(n, n1, n2) * sigma_HR

    masks = np.zeros((n,n1,n2))

    for i in range(x0.size):
        masks[i,np.int(x0[i]-np1/2):np.int(x0[i]+np1/2), np.int(y0[i]-np2/2):np.int(y0[i]+np2/2)]=1

    vec = np.zeros(niter)
    vec2 = np.zeros(niter)
    vec3 = np.zeros(niter)

    t0 =time.clock()
    for i in range(niter):
        if (i % 100+1 == True) and (verbosity == 1):
            print('Current iteration: ', i, ', time: ', time.clock()-t0)

            #plt.imshow((LR - np.dot(np.sum(Sa, axis = 0).flatten(), matLR)).reshape(N1,N2))

            #plt.savefig('Residuals_HSC_'+str(i)+'.png')


        #Resolution combination problem
        S = np.sum(Sa.reshape(n, n1*n2),axis = 1)
        R = mu * (np.dot(LR - np.dot(np.sum(Sa, axis=0).flatten(), matLR), matLR.T) * wvar_LR + mu * (HR - np.sum(Sa, axis=0).flatten()) * wvar_HR).reshape(n1, n2)
        for j in range(n):
            for k in range(1):
                #plt.imshow(Sa[0]); plt.show()
                Sa_nn = reg_nn(Sa[j,np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)]/S[j])

                Sa[j, np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)] += R[np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)] - 1e-6*Sa_nn*S[j]


        #High resolution problem
        S = np.sum(SHR.reshape(n, n1 * n2), axis=1)
        R =  mu1 * (HR - np.sum(SHR, axis=0).flatten()).reshape(n1, n2)
        for j in range(n):
            for k in range(1):
                #plt.imshow(Sa[0]); plt.show()
                SHR_nn = reg_nn(SHR[j,np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)]/S[j])

                SHR[j, np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)] += R[np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)] - 1e-6*SHR_nn*S[j]

        #Low resolution problem
        S = np.sum(SLR.reshape(n, n1 * n2), axis=1)
        R = mu * (np.dot(LR - np.dot(np.sum(SLR, axis=0).flatten(), matLR), matLR.T) ).reshape(n1, n2)
        for j in range(n):
            for k in range(1):
                #plt.imshow(Sa[0]); plt.show()
                SLR_nn = reg_nn(SLR[j,np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)]/S[j])

                SLR[j, np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)] += R[np.int(x0[j]-np1/2):np.int(x0[j]+np1/2), np.int(y0[j]-np2/2):np.int(y0[j]+np2/2)] - 1e-6*SLR_nn*S[j]


        SLR[SLR < 0] = 0
        SHR[SHR < 0] = 0
        Sa[Sa < 0] = 0


        #Sa /= np.sum(Sa) / (n1*n2)
        #SL /= np.sum(SL) / (n1 * n2)
        #SH /= np.sum(SH) / (n1 * n2)

        vec[i] = np.std(LR - np.dot(np.sum(Sa, axis = 0).flatten(), matLR))**2/2./sigma_LR**2 + np.std(HR-np.sum(Sa, axis = 0).flatten())**2*wvar_LR/2./sigma_HR**2
        vec2[i] = np.std(LR - np.dot(np.sum(SLR, axis = 0).flatten(), matLR))**2/sigma_LR**2
        vec3[i] = np.std(HR - np.sum(SHR, axis = 0).flatten()) ** 2 / sigma_LR ** 2


    if verbosity == 1:
        plt.plot(vec3, 'b', label='LR', linewidth=3)
        plt.plot(vec2, 'g', label = 'LR', linewidth = 3)
        plt.plot(vec, 'r', label='All', linewidth=2)
        plt.show()

    return Sa*masks, SHR*masks, SLR*masks

def linorm2D_filterA(filter, filterT, A, shape, nit):
    """
      Estimates the maximal eigen value of a matrix A

      INPUTS:
          A: matrix
          nit: number of iterations

      OUTPUTS:
          xn: maximal eigen value

       EXAMPLES

    """

    n1,n2 = shape
    x0 = np.random.rand(n1,n2)
    x0 = x0 / np.sqrt(np.sum(x0 ** 2))

    for i in range(nit):
        x = np.dot(A,filter(x0).reshape(1,n1*n2))
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = filterT(np.dot(A.T,xp).reshape(n1,n2))
        yn = np.sqrt(np.sum(y ** 2))

        if yn < np.dot(y.reshape(1,n1*n2), x0.reshape(1,n1*n2).T):
            break
        x0 = y / yn

    return 1./xn

def linorm2D_A(S, A, nit):
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
        x = np.dot(np.dot(A,x0), S)
        xn = np.sqrt(np.sum(x ** 2))
        xp = x / xn
        y = np.dot(A.T,np.dot( xp, S.T))
        yn = np.sqrt(np.sum(y ** 2))

        if yn < np.dot(y, x0.T):
            break
        x0 = y / yn

    return 1./xn

def Deblend2D_filter(HR, LR, filter_HR, filter_HRT, matLR, niter, nc, verbosity = 0):
    # HR: high resolution data
    # LR: low resolution data
    # filter_HR(T): filter for the high resolution PSF convolution (and its transpose)
    # matLR: Operator for the downsampling and PSF convolution
    # niter: number of iterations
    # nc: number of colour components to extract
    n = HR.size
    n1 = np.int(HR.size**0.5)
    n2 = np.int(HR.size ** 0.5)
    N = LR.size
    sigma_HR = MAD(HR.reshape(np.int(n**0.5), np.int(n**0.5)))
    sigma_LR = MAD(LR.reshape(np.int(N**0.5), np.int(N**0.5)))

    var_norm = 1./sigma_HR**2 + 1./sigma_LR**2
    wvar_HR = (1./sigma_HR**2)*(1./var_norm)
    wvar_LR = (1./sigma_LR**2)*(1./var_norm)



    AHR = np.random.rand(nc,1)
    AHR /= np.sum(AHR)
    ALR = np.random.rand(nc,1)
    ALR /= np.sum(ALR)

    mu_HR = linorm2D_filterA(filter_HR, filter_HRT, AHR, (n1, n2), 10) #/ 10.
    mu_LR = linorm2D_A(matLR, ALR, 10) / 1.
    mu = (mu_HR + mu_LR) / 2

    muALR = 1.
    muAHR = 1.

    Sa = np.zeros((nc,HR.size))
    vec = np.zeros(niter)
    t0 =time.clock()
    for i in range(niter):
        if (i % 100+1 == True) and (verbosity == 1):
            print('Current iteration: ', i, ', time: ', time.clock()-t0)

        Sa += mu_LR * np.dot(ALR,np.dot( LR - np.dot(np.dot(ALR.T,Sa), matLR), matLR.T))*wvar_LR + mu_HR * np.dot(AHR,filter_HRT((HR-filter_HR(np.dot(AHR.T,Sa))).reshape(n1,n2)).reshape(1,n1*n2))*wvar_HR

        Sa[Sa < 0] = 0


        ALR = ALR + muALR * \
             np.dot(np.dot(LR - np.dot(np.dot(ALR.T, Sa), matLR), matLR.T), Sa.T).T#*wvar_LR
        AHR = AHR + muAHR * \
             np.dot(filter_HRT(HR - filter_HR(np.dot(AHR.T, Sa).reshape(n1,n2)).reshape(1,n1*n2)), Sa.T).T#*wvar_HR

        AHR[AHR<0] = 0
        ALR[ALR < 0] = 0

        for j in range(nc):

            AHR[j,:] = AHR[j,:]/(AHR[j,:]+ALR[j,:])
            ALR[j,:] = ALR[j,:] / (AHR[j,:] + ALR[j,:])
        print(AHR)
        print(ALR)
        vec[i] = np.std( LR - np.dot(np.dot(ALR.T,Sa), matLR))**2/2./sigma_LR**2 + np.std((HR-filter_HR(np.dot(AHR.T,Sa))).reshape(n1,n2))**2*wvar_LR/2./sigma_HR**2


    if verbosity == 1:
        plt.plot(vec, 'r', label = 'All', linewidth = 2)
        plt.show()

    return Sa, AHR, ALR




def match_patches(x0,y0,WLR, WHR, excess):

    '''
    :param x0, y0: coordinates of the center of the patch in High Resolutions pixels
    :param WLR, WHR: WCS of the Low and High resolution fields respectively
    :param excess: half size of the box
    :return:
    x_HR, y_HR: pixel coordinates of the grid for the High resolution patch
    X_LR, Y_LR: pixel coordinates of the grid for the Low resolution grid
    X_HR, Y_HR: pixel coordinates of the Low resolution grid in units of the High resolution patch
    '''

    xstart = x0 - excess
    xstop = x0 + excess
    ystart = y0 - excess
    ystop = y0 + excess

    XX = np.linspace(xstart, xstop, xstop-xstart + 1)
    YY = np.linspace(ystart, ystop, xstop-xstart + 1)

    x, y = np.meshgrid(XX, YY)
    x_HR = x.flatten().astype(int) + 0.5
    y_HR = y.flatten().astype(int) + 0.5

    # LR coordinates

    Ramin, Decmin = WHR.wcs_pix2world(ystart, xstart, 0)
    Ramax, Decmax = WHR.wcs_pix2world(ystop, xstop, 0)
    Ymin, Xmin = WLR.wcs_world2pix(Ramin, Decmin, 0)
    Ymax, Xmax = WLR.wcs_world2pix(Ramax, Decmax, 0)

    X = np.linspace(np.int(Xmin), np.int(Xmax), np.int(Xmax)-np.int(Xmin)+1)
    Y = np.linspace(np.int(Ymin), np.int(Ymax), np.int(Ymax)-np.int(Ymin)+1)

    X, Y = np.meshgrid(X, Y)
    X_LR = X.flatten().astype(int) + 0.5
    Y_LR = Y.flatten().astype(int) + 0.5
    Ra_LR, Dec_LR = WLR.wcs_pix2world(Y_LR, X_LR, 0)  # type:
    Y_HR, X_HR = WHR.wcs_world2pix(Ra_LR, Dec_LR, 0)

    return x_HR, y_HR, X_LR, Y_LR, X_HR, Y_HR

def make_patches(x_HR, y_HR, X_LR, Y_LR, Im_HR, Im_LR):
    '''
    :param x_HR, y_HR: Coordinates of the High resolution grid
    :param X_LR, Y_LR: Coordinates of the Low resolution grid
    :param Im_HR: High resolution FoV
    :param Im_LR: Low resolution FoV
    :return: Patch_HR, Patch_LR
    '''

    x_HR = x_HR.astype(int)
    y_HR = y_HR.astype(int)
    X_LR = X_LR.astype(int)
    Y_LR = Y_LR.astype(int)


    N1 = np.max(X_LR)-np.min(X_LR)+1
    N2 = np.max(Y_LR)-np.min(Y_LR)+1

    n1 = np.max(x_HR)-np.min(x_HR)+1
    n2 = np.max(y_HR)-np.min(y_HR)+1

    cut_HR = Im_HR[x_HR, y_HR].reshape(n2,n1)
    cut_LR = Im_LR[X_LR, Y_LR].reshape(N2,N1)

    cut_HR /= np.sum(cut_HR)/(n1 * n2)
    cut_LR /= np.sum(cut_LR)/(N1 * N2)

    return cut_HR, cut_LR