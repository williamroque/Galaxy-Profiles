import numpy as np
from scipy.signal import fftconvolve, savgol_filter
from scipy.optimize import curve_fit
from scipy.stats import linregress

import matplotlib.pyplot as plt


class BulgeDiskAutoFitter:
    def __init__(self, radii, sb_values, sb_std, psf, ZP, axis):
        self.radii = radii
        self.sb_values = sb_values
        self.sb_std = sb_std
        self.psf = psf
        self.ZP = ZP
        self.axis = axis


    def evaluate(self, r, mu0_b, re_b, n_b, mu0_d, re_d):
        """Sersic + exponential profile

        Parameters
        ----------
        r
            radius (independent variable)
        mu0_b
            central brightness according to the bulge model
        re_b
            scale_length according to bulge model
        n_b
            sersic index
        mu0_d
            central brightness according to the disk model
        re_d
            scale_length according to disk model
        """

        b_n = 2*n_b - 1/3 + 4/(405*n_b)

        bulge = mu0_b * np.exp(-b_n * (r/re_b)**(1/n_b))
        disk = mu0_d * np.exp(-b_n * (r/re_d))

        bulge_threshold = np.where(bulge < 1e-5)[0]

        if len(bulge_threshold):
            bulge[bulge_threshold[0]:] = 1e-5

        total = bulge + disk

        total_symmetric = np.append(total[::-1], total)
        total_convolved = fftconvolve(
            total_symmetric,
            self.psf,
            mode='same'
        )[len(total):]

        return total_convolved


    def find_linear(self):
        sb_values = savgol_filter(self.sb_values, 51, 2)

        self.axis.plot(self.radii, sb_values, 'r.')

        for i in range(len(self.radii) - 10, -1, -1):
            slope, intercept, rvalue, *_ = linregress(
                self.radii[i:],
                sb_values[i:]
            )

            if abs(rvalue) < 0.99:
                break

        return slope, intercept, self.radii[i]


    def fit(self):
        slope, intercept, radius = self.find_linear()

        I_0 = 10**((self.ZP - intercept)/2.5)
        h_R = 2.5/(slope*np.log(10))

        sb_values = slope*self.radii + intercept

        self.axis.plot(self.radii, sb_values)

        return I_0, h_R, radius
