#!/usr/bin/env python

import argparse
import warnings

from joblib import Parallel, delayed

import csv

import os

import numpy as np
import matplotlib.pyplot as plt

from libs.fitters import BulgeDiskAutoFitter
from profile import Profile

warnings.filterwarnings('ignore')


def main(args, profile_path):
    fig, ax = plt.subplots()

    profile = Profile(
        args.psf,
        args.ZP,
        args.pix2sec,
        args.SBlim,
        args.adderr,
        ax
    )

    profile.load(
        profile_path,
        args.profile_type,
        'Galex_images/PGC3377_sky_subtr_n.fits'
    )

    ax.plot(profile.radii, profile.SB_values)

    ax.set_ylim([
        np.nanmin(profile.SB_values) - 0.5,
        profile.lim_mag,
    ])
    ax.invert_yaxis()

    ax.errorbar(
        profile.radii,
        profile.SB_values,
        profile.SB_std
    )

    fitters = {
        'bulge-disk-auto': BulgeDiskAutoFitter
    }
    fitter_type = fitters[args.fitter]

    I_0, h_R, inner_cutoff = profile.fit(fitter_type)

    basename = os.path.basename(
        os.path.dirname(profile_path)
    )
    directory = os.path.join(args.outputdir, basename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    figure_path = os.path.join(directory, 'plot.png')
    data_path = os.path.join(directory, 'fit.csv')

    fig.savefig(figure_path)
    plt.close(fig)

    with open(data_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['inner', 'outer', 'scale_length', 'central_SB'])
        writer.writerow([inner_cutoff, profile.outer_cutoff, h_R, I_0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outputdir')
    parser.add_argument('profile')
    parser.add_argument('psf', help = 'Path to a text file with PSF')
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
        nargs = '?',
        const = 1,
        help = 'Optional: Input the zero-point',
        type = float,
        default = 20.08
    )
    parser.add_argument(
        '--pix2sec',
        nargs = '?',
        const = 1,
        help = 'Optional: Input the pixelscale',
        type = float,
        default = 1.5
    )
    parser.add_argument(
        '--SBlim',
        nargs = '?',
        const = 1,
        help = 'Optional: Input the limit of the surface brightnes (the faintest value to be shown)',
        type = float,
        default = 30.0
    )
    parser.add_argument(
        '--adderr',
        nargs = '?',
        const = 1,
        help = 'Optional: Input additional error',
        type = float,
        default = 0.000740
    )
    parser.add_argument(
        'profiles',
        nargs = argparse.REMAINDER
    )
    
    args = parser.parse_args()

    Parallel(n_jobs=-1)(
        delayed(main)(args, profile) for profile in args.profiles
    )
