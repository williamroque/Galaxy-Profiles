import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve, savgol_filter
from scipy.stats import linregress


class BulgeDiskAutoFitter:
    RVALUE_THRESHOLD = 0.99

    def __init__(self, radii, sb_values, sb_std, psf, ZP, axis, bounds, plot):
        self.radii = radii
        self.sb_values = sb_values
        self.sb_std = sb_std
        self.psf = psf
        self.ZP = ZP
        self.axis = axis
        self.bounds = bounds
        self.plot = plot


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
        if self.plot:
            self.axis.plot(self.radii, self.sb_values, 'b-', linewidth=1)

        window_size = 41

        if len(self.sb_values) <= window_size:
            window_size = (len(self.sb_values) // 6) * 2 + 1

        spline = CubicSpline(self.radii, self.sb_values)
        domain = np.linspace(
            self.radii.min(),
            self.radii.max(),
            len(self.radii)
        )

        sb_values = spline(domain)
        sb_values = savgol_filter(sb_values, window_size, 1)

        if self.plot:
            self.axis.plot(domain, sb_values, 'r--')

        if self.bounds is None:
            for i in range(len(domain) - 2, -1, -1):
                slope, intercept, rvalue, *_ = linregress(
                    domain[i:],
                    sb_values[i:]
                )

                if abs(rvalue) < self.__class__.RVALUE_THRESHOLD:
                    break

            return slope, intercept, domain[i]

        inner_index = np.where(domain < self.bounds[0])[0]
        if len(inner_index):
            inner_index = inner_index[0]
        else:
            inner_index = 0

        outer_index = np.where(domain > self.bounds[1])[0]
        if len(outer_index):
            outer_index = outer_index[0]
        else:
            outer_index = len(domain) - 1

        if inner_index > outer_index:
            inner_index, outer_index = outer_index, inner_index
        elif inner_index == outer_index:
            inner_index -= 1
            outer_index += 1

        slope, intercept, *_ = linregress(
            domain[inner_index:outer_index + 1],
            sb_values[inner_index:outer_index + 1]
        )

        return slope, intercept, domain[inner_index]


    def fit(self):
        slope, intercept, radius = self.find_linear()

        I_0 = 10**((self.ZP - intercept)/2.5)
        h_R = 2.5/(slope*np.log(10))

        sb_values = slope*self.radii + intercept

        if self.plot:
            self.axis.plot(self.radii, sb_values, 'k')

        return I_0, h_R, radius, slope, radius, self.radii, sb_values
