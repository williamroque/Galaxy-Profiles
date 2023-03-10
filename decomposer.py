#!/usr/bin/env python

import argparse
from pathlib import Path
import os
import math
import shutil
import json
import warnings
from string import Template

from scipy.ndimage import zoom
from scipy.optimize import fmin
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MPLEllipse

from astropy.io import fits
from astropy.modeling import models, fitting


from libs.fitters import SimpleDiskFitter
from libs.fitters import BrokenDiskFitter
from libs.fitters import DoubleBrokenDiskFitter
from libs.fitters import BulgeSimpleDiskFitter
from libs.fitters import BulgeBrokenDiskFitter
from libs.fitters import BulgeDoubleBrokenDiskFitter
from libs.fitters import make_n_break_fitter
from libs.fitters import make_n_break_fitter_with_bulge

warnings.filterwarnings("ignore")

matplotlib.rcParams['keymap.fullscreen'].remove("f")
matplotlib.rcParams['keymap.save'].remove("s")

matplotlib.rcParams['text.usetex'] = False

workdir = Path("workdir")
kappa = 2.5 / math.log(10)
colors = ["b", "g", "k"]


def fit_2dgaussian(data):
    y_size, x_size = data.shape
    p_init = models.Gaussian2D(amplitude=np.max(data), x_mean=x_size/2, y_mean=y_size/2,
                               x_stddev=3, y_stddev=3)
    fit_p = fitting.LevMarLSQFitter()
    y, x = np.mgrid[:y_size, :x_size]
    p = fit_p(p_init, x, y, data)
    return p


def find_gauss_center(x, y, make_plot=False):
    y -= np.nanmin(y)

    def f(params):
        a = params[0]
        x0 = params[1]
        sigma = params[2]
        chi_sq = np.nansum((y - a * np.exp(-(x-x0)**2 / (2*sigma**2)))**2)
        return chi_sq
    a_guess = max(y)
    x0_guess = (x[0] + x[-1]) / 2
    sigma_guess = x0_guess - x[0]  # FIXME
    initials = a_guess, x0_guess, sigma_guess
    best_params = fmin(f, x0=initials, disp=False)
    if make_plot:
        plt.plot(x, y, "ko")
        plt.plot(x, best_params[0] * np.exp(-(x-best_params[1])**2 / (2*best_params[2]**2)))
        plt.show()
    return best_params[1]


class Decomposer(object):
    def __init__(self, profile_file_name, image_name, psf_file, profile_type, m0, pix2sec, limmag, adderror):
        self.psf = np.genfromtxt(psf_file, unpack=True, usecols=[1])
        # self.psf = zoom(self.psf, image_scale)
        self.psf /= np.sum(self.psf) * np.nan

        self.magzpt = m0 #20.08
        self.image_scale = pix2sec  # arcsecs per pixel
        self.lim_mag = limmag#30
        self.add_error = adderror

        # Determine the center of the galaxy as the center of the image:
        hdulist = fits.open(image_name)
        inframe = hdulist[0].data
        ny, nx = np.shape(inframe)
        XC = nx/2.
        YC = ny/2.

        self.horizonthal_distances = []
        self.horizonthal_slice = []
        self.horizonthal_std = []
        self.ell_iraf = 0
        self.posang_iraf = 0
        self.x_cen_iraf = []
        self.y_cen_iraf = []

        if profile_type == 'iraf':
            # Load profile from iraf output
            for line in open(profile_file_name):
                if line.startswith("#"):
                    continue
                params = line.split()
                if len(params) <= 1:
                    continue
                if "INDEF" in params[1]:
                    continue
                else:
                    dist = self.image_scale * float(params[1])
                if "INDEF" in params[2]:
                    continue
                else:
                    flux = float(params[2]) / self.image_scale**2
                    mag = -2.5 * np.log10(flux) + self.magzpt
                if ("INDEF" in params[3]):
                    continue
                else:
                    flux_std = (float(params[3])+self.add_error) / self.image_scale**2  # 0.000740???
                    mag_std = flux_std * (2.5/np.log(10)) / flux
                if mag > self.lim_mag:
                    break
                self.horizonthal_distances.append(dist)
                self.horizonthal_slice.append(mag)
                self.horizonthal_std.append(mag_std)
                if params[8] != "INDEF":
                    self.posang_iraf = float(params[8])
                if params[9] != "INDEF":
                    self.ell_iraf = 0  # float(params[6])
                if params[10] != "INDEF":
                    self.x_cen_iraf.append(float(params[10]))
                if params[12] != "INDEF":
                    self.y_cen_iraf.append(float(params[12]))
        else:
            # Load profile from azimProfile output
            for line in open(profile_file_name):
                if line.startswith("#"):
                    continue
                params = line.split()
                if len(params) <= 1:
                    continue

                dist = self.image_scale * float(params[0])
                flux = float(params[1]) / self.image_scale**2
                mag = -2.5 * np.log10(flux) + self.magzpt
                flux_std = (float(params[2])+self.add_error) / self.image_scale**2  # math.sqrt(((float(params[2]))**2+(0.000870)**2)) / self.image_scale**2 ### 0.000740???
                mag_std = flux_std * (2.5/np.log(10)) / flux
                if mag > self.lim_mag:
                    break
                self.horizonthal_distances.append(dist)
                self.horizonthal_slice.append(mag)
                self.horizonthal_std.append(mag_std)
                self.posang_iraf = 0
                self.ell_iraf = 0
                self.x_cen_iraf.append(XC)
                self.y_cen_iraf.append(YC)

        self.horizonthal_distances = np.array(self.horizonthal_distances)
        self.horizonthal_slice = np.array(self.horizonthal_slice)
        self.horizonthal_std = np.array(self.horizonthal_std)
        self.x_cen_iraf = np.median(self.x_cen_iraf)
        self.y_cen_iraf = np.median(self.y_cen_iraf)

        # Load image
        data = fits.getdata(image_name)
        y_size, x_size = data.shape
        max_r = min(1.1*self.horizonthal_distances[-1] / self.image_scale,
                    self.x_cen_iraf, self.y_cen_iraf, x_size-self.x_cen_iraf, y_size-self.y_cen_iraf)
        self.orig_image = data[int(self.y_cen_iraf-max_r): int(self.y_cen_iraf+max_r+1),
                               int(self.x_cen_iraf-max_r): int(self.x_cen_iraf+max_r+1)]
        self.orig_image_cen = self.orig_image.shape[0] / 2

        # Setup fitting
        self.models_list = [SimpleDiskFitter, BrokenDiskFitter, DoubleBrokenDiskFitter,
                            make_n_break_fitter(3), make_n_break_fitter(4),
                            BulgeSimpleDiskFitter, BulgeBrokenDiskFitter, BulgeDoubleBrokenDiskFitter,
                            make_n_break_fitter_with_bulge(3), make_n_break_fitter_with_bulge(4)]
        self.best_model_idx = None  # Best model by the BIC value
        self.saved_fits = {}
        self.saved_surf_bris = {}
        self.horizonthal_fitter = None
        self.selected_model_idx = -1
        self.ref_points = []
        self.requesting = False
        self.request_question = ""
        self.range_low_lim = 0
        self.range_up_lim = None

        # Initialize plot
        self.messages = []
        self.fig = plt.figure(figsize=(18, 8))
        gs = self.fig.add_gridspec(nrows=4, ncols=4)
        # Original galaxy image
        self.ax_orig = self.fig.add_subplot(gs[2:4, 0])
        low = np.percentile(self.orig_image, 1)
        high = np.percentile(self.orig_image, 99)
        self.ax_orig.imshow(self.orig_image, vmin=low, vmax=high)
        self.ellipses_plot_instances = []
        # self.ax_orig.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        # self.ax_orig.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
        # Horizonthal slice
        self.ax_horizonthal = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_horizonthal.set_ylim([np.nanmin(self.horizonthal_slice)-0.5, self.lim_mag])
        self.horizonthal_slice_plot_instance = None
        self.ax_horizonthal.invert_yaxis()
        self.ax_horizonthal.set_xlabel("r ['']")
        self.horizonthal_fit_plot_instances = []
        self.ref_points_plot_instances = []
        self.range_low_lim_plot_instance = None
        self.range_up_lim_plot_instance = None

        # Help panel
        self.ax_help = self.fig.add_subplot(gs[:, 2:])
        self.ax_help.set_xticks([])
        self.ax_help.set_yticks([])
        self.models_plot_instances = []
        self.fitres_plot_instances = []
        self.requests = None
        self.request_plot_instance = None
        self.messages_plot_instances = []
        # Add permanent help info
        self.ax_help.text(x=0.05, y=0.08, s="f: fit current model using reference points",
                          transform=self.ax_help.transAxes, fontsize=12)
        self.ax_help.text(x=0.5, y=0.08, s="r: reset current model",
                          transform=self.ax_help.transAxes, fontsize=12)
        self.ax_help.text(x=0.05, y=0.04, s="s: save results",
                          transform=self.ax_help.transAxes, fontsize=12)
        self.ax_help.text(x=0.5, y=0.04, s="space: mark the model as the best one",
                          transform=self.ax_help.transAxes, fontsize=12)
        plt.tight_layout()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        self.make_plot()
        # fits.PrimaryHDU(data=self.image_data_rot).writeto(workdir / "rot.fits")
        # fits.PrimaryHDU(data=self.cutout_data).writeto(workdir / "cut.fits")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(workdir)

    def reset(self):
        self.horizonthal_fitter.clean()
        del self.saved_fits[self.horizonthal_fitter.name.lower()]
        del self.saved_surf_bris[self.horizonthal_fitter.name.lower()]
        self.ref_points = []
        self.requests = self.models_list[self.selected_model_idx].ref_points.copy()
        self.request_question = self.requests.pop(0)
        self.requesting = True
        self.ref_points = []
        self.make_plot()

    def on_click(self, event):
        if (event.inaxes == self.ax_horizonthal) and (event.button == 3):
            self.range_low_lim = event.xdata
            self.make_plot()
            return
        if (event.inaxes == self.ax_horizonthal) and (self.requesting):
            # Click on slice
            self.ref_points.append((event.xdata, event.ydata))
            if self.requests:
                self.request_question = self.requests.pop(0)
            else:
                self.request_question = None
                self.requesting = False
                self.range_up_lim = event.xdata
        self.make_plot()

    def on_press(self, event):
        try:
            idx = int(event.key)
            self.change_model(idx)
            self.make_plot()
        except ValueError:
            pass
        if event.key == "f":
            self.fit_slices()
        if event.key == "s":
            self.save_results()
        if event.key == "a":
            self.fit_slices(auto=True)
        if event.key == "d":
            self.fit_all_models()
        if event.key == " ":
            # Mark current model as best
            self.best_model_idx = self.selected_model_idx
            self.make_plot()
        if event.key == "i":
            self.prepare_imfit_run()
            self.messages.append("Imfit config created")
            self.make_plot()
        if event.key == "r":
            self.reset()
            self.make_plot()

    def change_model(self, idx):
        if idx >= len(self.models_list):
            return
        self.selected_model_idx = idx
        self.horizonthal_fitter = self.models_list[idx](self.horizonthal_distances,
                                                        self.horizonthal_slice,
                                                        self.horizonthal_std,
                                                        self.psf)
        self.requests = self.models_list[idx].ref_points.copy()
        self.request_question = self.requests.pop(0)
        self.requesting = True
        self.ref_points = []
        if self.horizonthal_fitter.name.lower() in self.saved_fits.keys():
            self.horizonthal_fitter.restore(self.saved_fits[self.horizonthal_fitter.name.lower()]['radial'])

    def fit_slices(self, auto=False):
        # Set initial parameters to the fitter
        if auto is False:
            self.horizonthal_fitter.compute_initials(self.ref_points)
            self.horizonthal_fitter.fit(self.range_low_lim, self.range_up_lim)
        else:
            self.horizonthal_fitter.fit_auto()
        name = self.horizonthal_fitter.name.lower()
        self.saved_fits[name] = {"radial": self.horizonthal_fitter.get_all_params()}
        self.saved_surf_bris[name] = {"radial": self.horizonthal_fitter.get_all_data()}
        self.find_best_model()
        self.make_plot()

    def fit_all_models(self):
        # Perform fitting of all models
        for idx in range(len(self.models_list)):
            print(f"Fitting model {self.models_list[idx].name}")
            self.change_model(idx)
            self.fit_slices(auto=True)
            self.make_plot()
        self.find_best_model()
        self.change_model(self.best_model_idx)
        self.make_plot()

    def find_best_model(self):
        """
        Compare BIC values of the fitted models to find the best one
        """
        best_bic = 1e10
        best_n_pars = 0
        for idx, model in enumerate(self.models_list):
            name = model.name.lower()
            if name in self.saved_fits:
                # Fit done
                bic = self.saved_fits[name]["radial"]["BIC"]
                if bic < best_bic:
                    if model.n_free_params < best_n_pars:
                        # Model has lower (i.e. better) BIC for lower number of
                        # parameters is an absolute winner
                        self.best_model_idx = idx
                        best_bic = bic
                        best_n_pars = model.n_free_params
                    else:
                        # Model has lower BIC but with the cost of bigger number
                        # parameters
                        delta_bic = best_bic - bic
                        if delta_bic > 3.5:
                            # Suppose that 3 is big enough to consider the model as a
                            # better one
                            self.best_model_idx = idx
                            best_bic = bic
                            best_n_pars = model.n_free_params

    def make_plot(self):
        # plot horizonthal slice
        if self.horizonthal_slice_plot_instance is not None:
            self.horizonthal_slice_plot_instance.remove()
        self.horizonthal_slice_plot_instance = None
        p = self.ax_horizonthal.errorbar(x=self.horizonthal_distances, y=self.horizonthal_slice,
                                         yerr=self.horizonthal_std, fmt="ro")
        self.horizonthal_slice_plot_instance = p
        self.ax_horizonthal.set_xlim([self.horizonthal_distances[0],
                                      self.horizonthal_distances[-1]*1.025])

        # Plot range low lim
        if self.range_low_lim_plot_instance is not None:
            self.range_low_lim_plot_instance.remove()
        self.range_low_lim_plot_instance = None
        if self.range_low_lim > 0:
            p = self.ax_horizonthal.axvline(x=self.range_low_lim)
            self.range_low_lim_plot_instance = p
        if self.range_up_lim_plot_instance is not None:
            self.range_up_lim_plot_instance.remove()
        self.range_up_lim_plot_instance = None
        if self.range_up_lim is not None:
            p = self.ax_horizonthal.axvline(x=self.range_up_lim)
            self.range_up_lim_plot_instance = p

        # Plot horizonthal fit
        while self.horizonthal_fit_plot_instances:
            self.horizonthal_fit_plot_instances.pop().remove()
        if self.horizonthal_fitter is not None:
            horizonthal_values = self.horizonthal_fitter.evaluate()
            low = self.range_low_lim
            if self.range_up_lim is None:
                up = self.horizonthal_distances[-1]
            else:
                up = self.range_up_lim
            good_inds = np.where((self.horizonthal_distances > low) * (self.horizonthal_distances < up))
            if not self.horizonthal_fitter.is_complex:
                # One component: just show it
                if horizonthal_values is not None:
                    p = self.ax_horizonthal.plot(self.horizonthal_distances[good_inds],
                                                 horizonthal_values[good_inds], zorder=3, color="k")
                    self.horizonthal_fit_plot_instances.append(p[0])
            else:
                horizonthal_values = self.horizonthal_fitter.evaluate(unpack=True)
                if horizonthal_values is not None:
                    for idx, v in enumerate(horizonthal_values):
                        p = self.ax_horizonthal.plot(self.horizonthal_distances[good_inds], v[good_inds],
                                                     zorder=3, color=colors[idx])
                        self.horizonthal_fit_plot_instances.append(p[0])

        # Requests for points
        if self.requesting:
            if self.request_plot_instance is not None:
                self.request_plot_instance.remove()
                self.request_plot_instance = None
            if self.request_question:
                self.request_plot_instance = self.ax_help.text(x=0.05, y=0.91-len(self.models_list)*0.03,
                                                               s=f"Select: {self.request_question}",
                                                               transform=self.ax_help.transAxes,
                                                               fontsize=14)
            else:
                if self.request_plot_instance is not None:
                    self.request_plot_instance.remove()
                    self.request_plot_instance = None
        else:
            if self.request_plot_instance is not None:
                self.request_plot_instance.remove()
                self.request_plot_instance = None

        # Plot ellipses on the image
        while self.ellipses_plot_instances:
            p = self.ellipses_plot_instances.pop()
            p.remove()
        if self.range_low_lim is not None:
            p = self.ax_orig.add_patch(MPLEllipse((self.orig_image_cen, self.orig_image_cen),
                                                  width=2*self.range_low_lim / self.image_scale,
                                                  height=2*self.range_low_lim*(1-self.ell_iraf) / self.image_scale,
                                                  angle=self.posang_iraf,
                                                  edgecolor="b", facecolor="none"))
            self.ellipses_plot_instances.append(p)
        if self.range_up_lim is not None:
            p = self.ax_orig.add_patch(MPLEllipse((self.orig_image_cen, self.orig_image_cen),
                                                  width=2*self.range_up_lim / self.image_scale,
                                                  height=2*self.range_up_lim*(1-self.ell_iraf) / self.image_scale,
                                                  angle=self.posang_iraf,
                                                  edgecolor="b", facecolor="none"))
            self.ellipses_plot_instances.append(p)

        # Plot help
        # models
        while self.models_plot_instances:
            self.models_plot_instances.pop().remove()
        self.models_plot_instances.append(self.ax_help.text(x=0.05, y=0.95, s="Available models",
                                                            transform=self.ax_help.transAxes,
                                                            fontsize=14))
        for idx, model in enumerate(self.models_list):
            if idx == self.selected_model_idx:
                color = "g"
            else:
                color = "k"
            if model.name.lower() in self.saved_fits.keys():
                s = f"{idx}: {model.name} (BIC: {self.saved_fits[model.name.lower()]['radial']['BIC']: 1.2f})"
            else:
                s = f"{idx}: {model.name}"
            if (self.best_model_idx is not None) and (idx == self.best_model_idx):
                s += " <- Best"
            self.models_plot_instances.append(self.ax_help.text(x=0.05, y=0.925-idx*0.025, s=s,
                                                                transform=self.ax_help.transAxes,
                                                                fontsize=14, color=color))

        # Ref points
        while self.ref_points_plot_instances:
            self.ref_points_plot_instances.pop().remove()
        for p in self.ref_points:
            self.ref_points_plot_instances.append(self.ax_horizonthal.scatter(p[0], p[1], zorder=3,
                                                                              color="g", marker="P", s=50))

        # Fit results
        while self.fitres_plot_instances:
            to_remove = self.fitres_plot_instances.pop()
            if isinstance(to_remove, list):
                while to_remove:
                    to_remove.pop().remove()
            else:
                try:
                    to_remove.remove()
                except:
                    pass
        if (self.horizonthal_fitter is not None) and (self.horizonthal_fitter.fit_done is True):
            fitted_params = self.horizonthal_fitter.get_all_params()
            p = self.ax_help.text(x=0.05, y=0.85-len(self.models_list)*0.025, s="Fit results:",
                                  transform=self.ax_help.transAxes, fontsize=14)
            self.fitres_plot_instances.append(p)
            idx = 0
            hidx = 0
            for name, value in fitted_params.items():
                if value is None:
                    continue
                if name.startswith("h"):
                    dx = 0.5
                    hidx += 1
                    y = 0.85-hidx*0.03-len(self.models_list)*0.03
                else:
                    dx = 0
                    idx += 1
                    y = 0.85-idx*0.03-len(self.models_list)*0.03
                self.fitres_plot_instances.append(self.ax_help.text(x=0.075+dx, y=y, s=f"{name}: {value:1.2f}",
                                                                    transform=self.ax_help.transAxes, fontsize=14))
                if "break" in name:
                    y = self.horizonthal_slice[np.argmin(np.abs(self.horizonthal_distances-value))]
                    p = self.ax_horizonthal.vlines(x=[value], ymin=y-0.75, ymax=y+0.75, linestyles="dashed", color="k")
                    self.fitres_plot_instances.append(p)
                    p = self.ax_orig.add_patch(MPLEllipse((self.orig_image_cen, self.orig_image_cen),
                                                          width=2*value / self.image_scale,
                                                          height=2*value*(1-self.ell_iraf) / self.image_scale,
                                                          angle=90-self.posang_iraf,
                                                          edgecolor="k", facecolor="none",
                                                          linewidth=2))
            self.ellipses_plot_instances.append(p)

        # Show messages
        while self.messages_plot_instances:
            self.messages_plot_instances.pop().remove()
        for idx, msg in enumerate(self.messages):
            y = 0.3-idx*0.03
            self.fitres_plot_instances.append(self.ax_help.text(x=0.375, y=y, s=msg, transform=self.ax_help.transAxes,
                                                                fontsize=14))
        self.messages = []
        plt.draw()

    def save_results(self):
        all_results = {}
        all_results["lower_limit"] = self.range_low_lim
        # Save slices
        all_results["radial_distances['']"] = list(self.horizonthal_distances.astype(float))
        all_results["radial_surf_bri"] = list(self.horizonthal_slice.astype(float))
        all_results["radial_surf_std"] = list(self.horizonthal_std.astype(float))
        # Save fit params
        # 1) A dictionary for all fit params
        all_results["fit_params"] = self.saved_fits
        # 2) A dictionary for all fitted surface brightnesses
        all_results["surf_brightnesses"] = self.saved_surf_bris
        # 3) Name of the best model
        all_results["best_model"] = self.models_list[self.best_model_idx].name.lower()
        # Save json
        filename = "Slice_fit.json"
        fout = open(filename, "w")
        json.dump(all_results, fout)
        fout.close()
        self.make_plot()

    def prepare_imfit_run(self):
        raise NotImplementedError
        """
        Make a directory with imfit run
        """
        self.imfit_run_path = Path("imfit_run")
        if not self.imfit_run_path.exists():
            os.makedirs(self.imfit_run_path)
        fits.PrimaryHDU(data=self.psf_fitted).writeto(self.imfit_run_path / "psf.fits", overwrite=True)
        header = fits.Header({"MAGZPT": self.magzpt})
        fits.PrimaryHDU(data=self.cutout_data, header=header).writeto(self.imfit_run_path / "image.fits",
                                                                      overwrite=True)
        fits.PrimaryHDU(data=self.cutout_mask).writeto(self.imfit_run_path / "mask.fits", overwrite=True)
        imfit_config = open(self.imfit_run_path / "config.imfit", "w")
        imfit_config.write(f"GAIN {self.gain}\n")
        imfit_config.write(f"READNOISE {self.readnoise}\n")
        try:
            z0 = np.nanmedian([vf.par_values["z0_d"] for vf in self.add_vertical_fitters])
        except TypeError:
            z0 = 3
        func_part = self.horizonthal_fitter.to_imfit(self.x_cen, self.y_cen, self.magzpt, self.image_scale)
        pars = {}
        func_part = Template(func_part).substitute(pars)
        imfit_config.write(func_part)
        imfit_config.close()


def setup():
    if workdir.exists():
        shutil.rmtree(workdir)
    os.makedirs(workdir)


def main(args):
    setup()
    # Manual intervention, no remote results
    with Decomposer(args.profile, args.image, args.psf, args.profile_type, args.ZP, args.pix2sec, args.SBlim, args.adderr) as d:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("profile")
    parser.add_argument("image")
    parser.add_argument("psf", help="Path to a text file with PSF")
    parser.add_argument("--profile-type", help="Optional: Type of the input file with the profile: iraf or azim", type=str, default='iraf')
    parser.add_argument("--ZP", nargs='?', const=1, help="Optional: Input the zero-point",type=float,default=20.08) 
    parser.add_argument("--pix2sec", nargs='?', const=1, help="Optional: Input the pixelscale",type=float,default=1.5)
    parser.add_argument("--SBlim", nargs='?', const=1, help="Optional: Input the limit of the surface brightnes (the faintest value to be shown)",type=float,default=30.0)
    parser.add_argument("--adderr", nargs='?', const=1, help="Optional: Input additional error ",type=float,default=0.000740)
    
    args = parser.parse_args()
    main(args)

# python3 /home/byu.local/mosav/Programs/decomposer/decomposer.py azim_model_PGC2440.txt PGC2440_depr.fits /home/byu.local/mosav/Programs/decomposer/psf_azim_model.txt --profile-type azim
