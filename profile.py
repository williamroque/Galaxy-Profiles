import math

import numpy as np

from astropy.io import fits
from scipy.signal import fftconvolve

kappa = 2.5 / math.log(10)


class Profile:
    def __init__(self, psf_file, m0, pix2sec, limmag, adderror, axis, bounds=None):
        self.psf = np.genfromtxt(psf_file, unpack=True, usecols=[1])
        self.psf /= np.sum(self.psf)

        self.magzpt = m0 # 20.08
        self.image_scale = pix2sec # arcsecs per pixel
        self.lim_mag = limmag # 30
        self.add_error = adderror
        self.axis = axis
        self.bounds = bounds

        self.radii = []
        self.SB_values = []
        self.SB_std = []

        self.outer_cutoff = None

    def load(self, profile_file_name, profile_type):
        # Determine the center of the galaxy as the center of the image:

        self.ell_iraf = 0
        self.posang_iraf = 0

        if profile_type == 'iraf':
            # Load profile from iraf output
            for line in open(profile_file_name):
                if line.startswith('#'):
                    continue
                params = line.split()
                if len(params) <= 1:
                    continue

                if 'INDEF' in params[1]:
                    continue
                else:
                    dist = self.image_scale * float(params[1])

                if 'INDEF' in params[2]:
                    continue
                else:
                    flux = float(params[2]) / self.image_scale**2
                    mag = -2.5 * np.log10(flux) + self.magzpt

                if ('INDEF' in params[3]):
                    continue
                else:
                    flux_std = (float(params[3])+self.add_error) / self.image_scale**2  # 0.000740???
                    mag_std = flux_std * (2.5/np.log(10)) / flux

                if mag > self.lim_mag:
                    break

                self.radii.append(dist)
                self.SB_values.append(mag)
                self.SB_std.append(mag_std)
                if params[8] != 'INDEF':
                    self.posang_iraf = float(params[8])
                if params[9] != 'INDEF':
                    self.ell_iraf = 0  # float(params[6])
        else:
            for line in open(profile_file_name):
                if line.startswith('#'):
                    continue
                params = line.split()
                if len(params) <= 1:
                    continue

                dist = self.image_scale * float(params[0])
                flux = float(params[1]) / self.image_scale**2
                mag = -2.5 * np.log10(flux) + self.magzpt
                flux_std = (float(params[2])+self.add_error) / self.image_scale**2
                mag_std = flux_std * (2.5/np.log(10)) / flux

                if mag > self.lim_mag:
                    break

                self.radii.append(dist)
                self.SB_values.append(mag)
                self.SB_std.append(mag_std)
                self.posang_iraf = 0
                self.ell_iraf = 0

        self.radii = np.array(self.radii)
        self.SB_values = np.array(self.SB_values)
        self.SB_std = np.array(self.SB_std)

        if self.outer_cutoff is None:
            self.radii[-1]

        self.range_low_lim = 0
        self.range_up_lim = None

        low = self.range_low_lim
        if self.range_up_lim is None:
            up = self.radii[-1]
        else:
            up = self.range_up_lim

        self.good_inds = np.where(
            (self.radii > low) *
            (self.radii < up)
        )

    def reduce_by(self, percentage):
        indices = np.random.choice(
            np.arange(0, len(self.radii)),
            int((100 - percentage) * len(self.radii) / 100),
            False
        )
        indices = np.sort(indices)

        self.radii = self.radii[indices]
        self.SB_values = self.SB_values[indices]
        self.SB_std = self.SB_std[indices]

    def fit(self, fitter_type, plot=True):
        fitter = fitter_type(
            self.radii,
            self.SB_values,
            self.SB_std,
            self.psf,
            self.magzpt,
            self.axis,
            self.bounds,
            plot
        )
        return fitter.fit()

    def generate_random(self):
        data_count = np.random.randint(80, 160)
        self.radii = np.sort(
            np.random.rand(data_count) * 100
        )

        sersic_index = 3
        m0 = 20

        mu0_d = 1# fix using https://arxiv.org/abs/0810.1953
        mu0_b = np.random.rand() * 2 # vary according to above

        h_d = (1 + np.random.rand()) * 10 * self.radii.max() # 1/10 to 1

        half_light = np.random.rand() * 5 # .01 to 1 of self.radii.max()

        b_n = 2 * sersic_index - 1/3 + 4 / (405 * sersic_index)
        disk = 10**(0.4 * (m0 - (mu0_d + kappa * self.radii / h_d)))
        bulge = 10**(0.4 * (m0 - (mu0_b +
                                  kappa * b_n * (self.radii/half_light) ** (1/sersic_index))))

        bulge_threshold = np.where(bulge < 1e-5)[0]

        if len(bulge_threshold):
            bulge[bulge_threshold[0]:] = 1e-5

        total = disk + bulge

        total_symmetric = np.append(total[::-1], total)
        total_convolved = fftconvolve(
            total_symmetric,
            self.psf,
            mode='same'
        )[len(total):]

        profile = -2.5 * np.log10(total_convolved) + m0

        self.SB_values = profile
        self.SB_std = np.zeros_like(self.radii)

        return profile
