#!/usr/bin/env python

import argparse
import warnings

from joblib import Parallel, delayed

import csv

import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

from libs.fitters import BulgeDiskAutoFitter
from profile import Profile

warnings.filterwarnings('ignore')


def main(args, profile_path):
    fig, ax = plt.subplots()

    ax.set_xlabel(r'$r$ (arcsec)')
    ax.set_ylabel(r'$\mu$ (mag arcsec$^{-2}$)')

    basename = os.path.basename(
        os.path.dirname(profile_path)
    )
    output_directory = os.path.join(args.outputdir, basename)
    input_directory = os.path.dirname(profile_path)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    fit_input_path = os.path.join(input_directory, 'fit.csv')

    try:
        with open(fit_input_path, 'r') as f:
            reader = csv.reader(f)
            data = dict(zip(*[*reader][:2]))

            if 'inner' in data and 'outer' in data:
                bounds = (
                    float(data['inner']),
                    float(data['outer'])
                )
            else:
                bounds = None
    except FileNotFoundError:
        bounds = None

    profile = Profile(
        args.psf,
        args.ZP,
        args.pix2sec,
        args.SBlim,
        args.adderr,
        ax,
        bounds
    )

    profile.load(
        profile_path,
        args.profile_type
    )

    ax.set_ylim([
        np.nanmin(profile.SB_values) - 0.5,
        profile.lim_mag,
    ])
    ax.invert_yaxis()

    fitters = {
        'bulge-disk-auto': BulgeDiskAutoFitter
    }
    fitter_type = fitters[args.fitter]

    bounds_final = None
    domain_final = None
    sb_final = None

    I_0_values = []
    h_R_values = []

    sb_bins = []
    bin_size = 20

    for i in range(args.realizations):
        if len(profile.radii) < 10 or len(profile.SB_values) < 10:
            break

        I_0, h_R, bounds, domain, sb_values = profile.fit(
            fitter_type, i == 0
        )

        for j in range(len(domain) // bin_size):
            start = j * bin_size
            end = (j + 1) * bin_size

            if j >= len(sb_bins):
                sb_bins.append([np.mean(sb_values[start:end])])
            else:
                sb_bins[j].append(np.mean(sb_values[start:end]))

        if i == 0:
            bounds_final = bounds
            domain_final = domain
            sb_final = sb_values

        I_0_values.append(I_0)
        h_R_values.append(h_R)

        profile.reduce_by(args.mask)

    I_0 = I_0_values[0]
    h_R = h_R_values[0]

    I_0_std = np.std(I_0_values)
    h_R_std = np.std(h_R_values)

    sb_std = np.array(
        [np.std(sb_bin) for sb_bin in sb_bins for _ in range(bin_size)]
    )

    ax.fill_between(
        domain_final,
        sb_final - np.abs(sb_std),
        sb_final + np.abs(sb_std),
        alpha = 0.5,
        edgecolor='#222',
        facecolor='#111'
    )

    figure_path = os.path.join(output_directory, 'plot.png')
    data_path = os.path.join(output_directory, 'fit.csv')

    fig.savefig(figure_path)
    plt.close(fig)

    with open(data_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'inner',
            'outer',
            'scale_length',
            'scale_length_std',
            'central_SB',
            'central_SB_std'
        ])
        writer.writerow([
            *bounds_final,
            h_R,
            h_R_std,
            I_0,
            I_0_std
        ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outputdir')
    parser.add_argument('psf', help = 'Path to a text file with PSF')
    parser.add_argument(
        'profiles',
        nargs = '+'
    )
    parser.add_argument(
        '--realizations',
        help = 'Optional: The number of bootstrapping realizations.',
        type = int,
        default = 50
    )
    parser.add_argument(
        '--mask',
        help = 'Optional: Percentage of points to be randomly removed during every bootstrapping realization.',
        type = float,
        default = 20
    )
    parser.add_argument(
        '--profile-type',
        help = 'Optional: Type of the input file with the profile. [iraf|azim]',
        type = str,
        default = 'iraf'
    )
    parser.add_argument(
        '--fitter',
        help = 'Optional: Type of fitter to use. [disk-auto]',
        type = str,
        default = 'disk-auto'
    )
    parser.add_argument(
        '--ZP',
        help = 'Optional: Input the zero-point',
        type = float,
        default = 20.08
    )
    parser.add_argument(
        '--pix2sec',
        help = 'Optional: Input the pixelscale',
        type = float,
        default = 1.5
    )
    parser.add_argument(
        '--SBlim',
        help = 'Optional: Input the limit of the surface brightnes (the faintest value to be shown)',
        type = float,
        default = 30.0
    )
    parser.add_argument(
        '--adderr',
        help = 'Optional: Input additional error',
        type = float,
        default = 0.000740
    )

    args = parser.parse_args()

    Parallel(n_jobs=-1)(
        delayed(main)(args, profile) for profile in args.profiles
    )
