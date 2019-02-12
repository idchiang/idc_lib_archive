import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.convolution import convolve_fft

FWHM2sigma = 0.5 / np.sqrt(2*np.log(2))


def Gaussian_Kernel_C1(FWHM_in, FWHM_out=25):
    # Converting scales
    sigma_sq = (FWHM_out**2 - FWHM_in**2) * FWHM2sigma**2
    # Generating grid points. Ref Anaino total dimension ~729", half 364.5"
    num_pix = 50
    x, y = np.meshgrid(np.arange(-num_pix, num_pix + 1),
                       np.arange(-num_pix, num_pix + 1))
    result = np.exp(-0.5 * (x**2 / sigma_sq + y**2 / sigma_sq))
    return result / np.sum(result)


sigma_p = 2.0
sigma_q = 1.0
mean_p = -0.01
mean_q = 0.01
corr_pq = 0.3
n_pixel = 300
cov = np.array([[sigma_p**2, sigma_p * sigma_q * corr_pq],
                [sigma_p * sigma_q * corr_pq, sigma_q**2]])
print('Input mean:', mean_p, mean_q)
print('Input cov:\n', cov)

"""
Fake noise generation
"""
temp = np.random.multivariate_normal([mean_p, mean_q], cov,
                                     size=[n_pixel, n_pixel])
image_p0 = temp[:, :, 0]
image_q0 = temp[:, :, 1]
print('Raw mean:', np.mean(image_p0), np.mean(image_q0))
print('Raw cov:\n', np.cov(image_p0.flatten(), image_q0.flatten()))
mean_p0 = np.mean(image_p0)
mean_q0 = np.mean(image_q0)
cov0 = np.cov(image_p0.flatten(), image_q0.flatten())

g0 = Gaussian_Kernel_C1(0.00, 2.0)
image_p1 = convolve_fft(image_p0, g0, quiet=True, allow_huge=True)
image_q1 = convolve_fft(image_q0, g0, quiet=True, allow_huge=True)
print('first mean:', np.mean(image_p1), np.mean(image_q1))
print('first cov:\n', np.cov(image_p1.flatten(), image_q1.flatten()))
print('')

means_p = [np.mean(image_p1)]
means_q = [np.mean(image_q1)]
covs = [np.cov(image_p1.flatten(), image_q1.flatten())]
ress = [1.0]
means_p = [np.mean(image_p1)]
means_q = [np.mean(image_q1)]
covs = [np.cov(image_p1.flatten(), image_q1.flatten())]
ress = [2.0]
for r in np.logspace(np.log10(4.0), np.log10(n_pixel)):
    g = Gaussian_Kernel_C1(ress[0], r)
    ress.append(r)
    image_pi = convolve_fft(image_p1, g, quiet=True, allow_huge=True)
    image_qi = convolve_fft(image_q1, g, quiet=True, allow_huge=True)
    means_p.append(np.mean(image_pi))
    means_q.append(np.mean(image_qi))
    covs.append(np.cov(image_pi.flatten(), image_qi.flatten()))

plt.ioff()
plt.figure()
plt.semilogx(ress, means_p, label='mean p', color='b', marker='o')
plt.semilogx(ress, [mean_p0] * len(ress), 'b--', alpha=0.5,
             label='mean p:input')
plt.semilogx(ress, means_q, label='mean_q', color='r', marker='o')
plt.semilogx(ress, [mean_q0] * len(ress), 'r--', alpha=0.5,
             label='mean q:input')
plt.legend()
plt.xlabel('FWHM (pixel)')
plt.ylabel('mean noise')
plt.savefig('output/means.png')

covs = np.array(covs)
plt.figure()
plt.loglog(ress, covs[:, 0, 0], label=r'$\sigma_p^2$', marker='o', color='b')
plt.loglog(ress, [cov0[0, 0]] * len(ress), 'b--', label=r'$\sigma_p^2$:input',
           alpha=0.5)
plt.loglog(ress, covs[:, 1, 1], label=r'$\sigma_q^2$', marker='o', color='r')
plt.loglog(ress, [cov0[1, 1]] * len(ress), 'r--', label=r'$\sigma_q^2$:input',
           alpha=0.5)
plt.loglog(ress, covs[:, 0, 1], label='cov(p,q)', marker='o', color='c')
plt.loglog(ress, [cov0[0, 1]] * len(ress), 'c--', label='cov(p,q):input',
           alpha=0.5)
plt.legend()
plt.xlabel('FWHM (pixel)')
plt.ylabel('Covaraince matrix element values')
plt.savefig('output/covs.png')

ress = np.array(ress)
plt.figure()
plt.plot(1 / ress, np.sqrt(covs[:, 0, 0]), label=r'$\sigma_p$', marker='o',
         color='b')
plt.plot(1 / ress, np.sqrt(covs[:, 1, 1]), label=r'$\sigma_q$', marker='o',
         color='r')
plt.legend()
plt.xlabel('1 / Resolution (pixel)')
plt.ylabel('Standard deviation')
plt.savefig('output/standard_deviation.png')

plt.figure()
plt.semilogx(ress, covs[:, 0, 1] / np.sqrt(covs[:, 0, 0] * covs[:, 1, 1]),
             marker='o')
plt.xlabel('FWHM (pixel)')
plt.ylabel('Correlation')
plt.savefig('output/corr.png')
plt.close('all')
