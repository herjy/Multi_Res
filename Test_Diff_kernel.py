
# Import Packages and setup
import os
import logging
from astropy.table import Table as ApTable

import numpy as np

import scarlet
import scarlet.display

import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")
# use a better colormap and don't interpolate the pixels
matplotlib.rc('image', cmap='gist_stern')
matplotlib.rc('image', interpolation='none')



# Load a real HSC-Cosmos blend with a different PSF in each band
datapath = '../real_data/hsc_cosmos'
files = os.listdir(datapath)
data = np.load(os.path.join(datapath, files[0]))
images = data["images"]
psfs = data["psfs"]
peaks = data["peaks"]
weights = data["weights"]

# Estimate the background RMS
bg_rms = np.sqrt(np.std(images, axis=(1,2))**2 + np.median(images, axis=(1,2))**2)

# Use Asinh scaling for the images
vmin, vmax = scarlet.display.zscale(images, fraction=.75)
norm = scarlet.display.Asinh(Q=50, vmin=vmin, vmax=vmax)

# Map i,r,g -> RGB
filter_indices = [3,2,1]
# Convert the image to an RGB image
img_rgb = scarlet.display.img_to_rgb(images, filter_indices=filter_indices, norm=norm)
plt.figure(figsize=(8,8))
plt.imshow(img_rgb)
for peak in peaks:
    plt.plot(peak[0], peak[1], "rx", mew=2)
plt.show()



import imp
imp.reload(scarlet)
import scarlet.psf_match
imp.reload(scarlet.psf_match)


target_psf = scarlet.psf_match.fit_target_psf(psfs, scarlet.psf_match.gaussian)

diff_kernels, psf_blend = scarlet.psf_match.build_diff_kernels(psfs, target_psf, l0_thresh=0.0001)

rgb_map = [3,2,1]
model = psf_blend.get_model()
for b, component in enumerate(psf_blend.components):
    fig = plt.figure(figsize=(15,2.5))
    ax = [fig.add_subplot(1,4,n+1) for n in range(4)]
    # Display the psf
    ax[0].set_title("psf")
    _img = ax[0].imshow(psfs[b])
    fig.colorbar(_img, ax=ax[0])
    # Display the model
    ax[1].set_title("modeled psf")
    _model = np.ma.array(model[b], mask=model[b]==0)
    _img = ax[1].imshow(_model)
    fig.colorbar(_img, ax=ax[1])
    # Display the difference kernel
    ax[2].set_title("difference kernel")
    _img = ax[2].imshow(np.ma.array(diff_kernels[b], mask=diff_kernels[b]==0))
    fig.colorbar(_img, ax=ax[2])
    # Display the residual
    ax[3].set_title("residual")
    residual = psfs[b]-model[b]
    vabs = np.max(np.abs(residual))
    _img = ax[3].imshow(residual, vmin=-vabs, vmax=vabs, cmap='seismic')
    fig.colorbar(_img, ax=ax[3])
    plt.show()


sources = []
images = data["images"]
bg_rms = np.sqrt(np.std(images, axis=(1,2))**2 + np.median(images, axis=(1,2))**2)
print(bg_rms)
for n,peak in enumerate(peaks):
    try:
        result = scarlet.ExtendedSource((peak[1], peak[0]), images, bg_rms, psf=diff_kernels)
        sources.append(result)
    except:
        print("No flux in peak {0} at {1}".format(n, peak))

blend_diff = scarlet.Blend(sources).set_data(images, bg_rms=bg_rms)
blend_diff.fit(100, e_rel=.015)
print("scarlet ran for {0} iterations".format(blend_diff.it))

def display_model_residual(images, blend, norm):
    """Display the data, model, and residual for a given result
    """
    model = blend.get_model()
    residual = images-model
    print("Data range: {0:.3f} to {1:.3f}\nresidual range: {2:.3f} to {3:.3f}\nrms: {4:.3f}".format(
        np.min(images),
        np.max(images),
        np.min(residual),
        np.max(residual),
        np.sqrt(np.std(residual)**2+np.mean(residual)**2)
    ))
    # Create RGB images
    img_rgb = scarlet.display.img_to_rgb(images, filter_indices=filter_indices, norm=norm)
    model_rgb = scarlet.display.img_to_rgb(model, filter_indices=filter_indices, norm=norm)
    residual_norm = scarlet.display.Linear(img=residual)
    residual_rgb = scarlet.display.img_to_rgb(residual, filter_indices=filter_indices, norm=residual_norm)

    # Show the data, model, and residual
    fig = plt.figure(figsize=(15,5))
    ax = [fig.add_subplot(1,3,n+1) for n in range(3)]
    ax[0].imshow(img_rgb)
    ax[0].set_title("Data")
    ax[1].imshow(model_rgb)
    ax[1].set_title("Model")
    ax[2].imshow(residual_rgb)
    ax[2].set_title("Residual")
    for k,component in enumerate(blend.components):
        y,x = component.center
        px, py = peaks[k]
        ax[0].plot(x, y, "gx")
        ax[0].plot(px, py, "rx")
        ax[1].text(x, y, k, color="r")
    plt.show()



display_model_residual(images, blend_diff, norm)
# display the total residual in each band
model = blend_diff.get_model()
residuals = images-model
fig = plt.figure(figsize=(15,10))
ax = [fig.add_subplot(2,3,n+1) for n in range(len(model))]
for b in range(len(model)):
    vabs = np.max(np.abs(residuals[b]))
    _img = ax[b].imshow(residuals[b], vmin=-vabs, vmax=vabs, cmap='seismic')
    fig.colorbar(_img, ax=ax[b])
plt.show()

