###################################################################################################
#
# ASTR 620 - Galaxies
#
# obs_sdss.py: Routines related to SDSS data
#
# (c) Benedikt Diemer, University of Maryland (based on similar code by Andrey Kratsov)
#
###################################################################################################

import numpy as np
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import urllib
from PIL import Image

from astropy.io import fits
from colossus.cosmology import cosmology

from routines import common as cmn
from routines import obs_utils

###################################################################################################

sdss_img_dir = cmn.data_dir + 'sdss_images/'
sdss_spec_dir = cmn.data_dir + 'sdss_spectra/'

# The solid angle in radians covered by the spectroscopic survey
solid_angle = 9274.0 / (180.0 / np.pi)**2

# The limiting magnitude for our sample (Petrosian magnitude in r-band)
m_r_limit = 17.77

# The natural scale of arcsecond / pixel for the SDSS camera; 1/scale = 2.52 pixels / arcsec
sdss_pixel_scale = 0.396127

# Colors in which to plot ugriz filters
filter_colors = 'bgrmk'

# Approximate diameter of spectral fibers in arcseconds
sdss_fiber_size = 3.0

# Spectra were not taken for all galaxies above the limiting magnitude because of so-called
# fiber collisions, meaning that galaxies were too close on the sky to fit the spectroscopic
# fibers next to each other. This happened for about 7% of galaxies (Bernardi et al. 2010).
spectroscopic_completeness = 0.93

###################################################################################################

# Load the SDSS spectroscopic sample. There are a few galaxies with extremely low redshifts that
# can cause numerical issues when computing luminosity distances etc., so we remove those.

def loadSdssSpecSample():

	hdu = fits.open(cmn.data_dir + 'sdss/sdss_specgals_dr8.fit')
	data = hdu[1].data

	mask = (data['z'] >= 1E-7)
	data = data[mask]
	
	return data

###################################################################################################

# Load the SDSS sample and extend the dataset by absolute magnitudes and colors including 
# K-corrections, distance modulus, effective radii, and so on.

def loadSdssSpecSampleExtra():
	
	data_all = loadSdssSpecSample()
	
	# Mask: no crazy colors (they will throw the K-correction fits off) and no high redshifts
	# (where the fitting function is not calibrated)
	mg_raw = data_all['modelMag_g'] - data_all['extinction_g']
	mr_raw = data_all['modelMag_r'] - data_all['extinction_r']
	gr_raw = mg_raw - mr_raw
	mask = (gr_raw > -0.5) & (gr_raw < 2.0) & (data_all['z'] <= 0.6) \
		 & (data_all['expRad_r'] > 0.0) & (data_all['deVRad_r'] > 0.0)
	data = data_all[mask]

	# Create dictionary. This will be temporary but it's easier to create the fields this way
	# and then add them to the main data array in the end.	
	data2 = {}

	# Compute observed magnitudes, using model mags for color since they are measured using the
	# same apertures. Then use those colors as input to K-corrections.
	mu_raw = data['modelMag_u'] - data['extinction_u']
	mg_raw = data['modelMag_g'] - data['extinction_g']
	mr_raw = data['modelMag_r'] - data['extinction_r']
	mi_raw = data['modelMag_i'] - data['extinction_i']
	mz_raw = data['modelMag_z'] - data['extinction_z']
	ur_raw = mu_raw - mr_raw
	gr_raw = mg_raw - mr_raw
	gi_raw = mg_raw - mi_raw
	rz_raw = mr_raw - mz_raw
	data2['K_u'] = obs_utils.kCorrection('u', data['z'], 'u-r', ur_raw)
	data2['K_g'] = obs_utils.kCorrection('g', data['z'], 'g-r', gr_raw)
	data2['K_r'] = obs_utils.kCorrection('r', data['z'], 'g-r', gr_raw)
	data2['K_i'] = obs_utils.kCorrection('i', data['z'], 'g-i', gi_raw)
	data2['K_z'] = obs_utils.kCorrection('z', data['z'], 'r-z', rz_raw)

	# Compute K-corrected colors
	mu_cor = mu_raw - data2['K_u']
	mg_cor = mg_raw - data2['K_g']
	mr_cor = mr_raw - data2['K_r']
	mi_cor = mi_raw - data2['K_i']
	mz_cor = mz_raw - data2['K_z']
	data2['color_ug'] = mu_cor - mg_cor
	data2['color_gr'] = mg_cor - mr_cor
	data2['color_ri'] = mr_cor - mi_cor
	data2['color_iz'] = mi_cor - mz_cor

	# Compute distance and distance modulus; distances are in Mpc (not Mpc/h)
	cosmo = cosmology.getCurrent()
	data2['DM'] = cosmo.distanceModulus(data['z'])
	data2['dL'] = cosmo.luminosityDistance(data['z']) / cosmo.h
	data2['dA'] = cosmo.angularDiameterDistance(data['z']) / cosmo.h

	# Compute absolute magnitudes; for the r-band, we store the individual exp/deV mags as well
	for f in 'ugriz':
		offset = data['extinction_%c' % f] + data2['K_%c' % f] + data2['DM']
		tpes = ['model', 'cmodel', 'petro', 'fiber']
		if f == 'r':
			tpes.extend(['exp', 'deV'])
		for tpe in tpes:
			data2['M_%s_%c' % (tpe, f)] = data['%sMag_%c' % (tpe, f)] - offset
			data2['m_%s_%c' % (tpe, f)] = data['%sMag_%c' % (tpe, f)] - data['extinction_%c' % f]

	# Find the radius corresponding to the better fit between exponential and de Vaucouleurs
	diff_exp = np.abs(data['modelMag_r'] - data['expMag_r'])
	diff_dev = np.abs(data['modelMag_r'] - data['devMag_r'])
	data2['mask_exp'] = (diff_exp < diff_dev)
	data2['Re_best'] = np.array(data['deVRad_r'])
	data2['Re_best'][data2['mask_exp']] = data['expRad_r'][data2['mask_exp']]
	kpc_factor = data2['dA'] * 1000.0 * np.pi / 180.0 / 3600.0
	data2['Re_best_kpc'] = data2['Re_best'] * kpc_factor
	
	# For the b/a factor, take the interpolated value between the exp and deV profiles
	data2['ab_best'] = data['fracdeV_r'] * data['deVAB_r'] + (1.0 - data['fracdeV_r']) * data['expAB_r']

	# Surface brightness. The factors of 2 in the logs comes from the fact that the half-light 
	# radius, by construction, contains half the total flux from the galaxy.
	data2['mu_petro_r'] = data2['m_petro_r'] + 2.5 * np.log10(2.0 * np.pi * data['petroR50_r']**2)
	data2['mu_cmodel_r'] = data2['m_cmodel_r'] + 2.5 * np.log10(2.0 * np.pi * data2['Re_best']**2)

	# Concentration
	data2['c_90_50_r'] = data['petroR90_r'] / data['petroR50_r']

	# Add inverse Vmax. Here we do not impose a limit, which means that some nearby galaxies will
	# have extremely small Vmax and thus extremely large 1/Vmax weightings. It is best to excluse 
	# such galaxies.
	data2['1/Vmax'] = inverseVmax(data2['M_petro_r'], m_limit = m_r_limit, z_max = None)

	# We combine the structured array data and the dictionary data2 manually. While there is a 
	# numpy.lib.recfunctions.append_fields() function, that function is incredibly slow compared
	# to the following solution which manually creates the fields and copies the data.
	dtypes = []
	for n in data.dtype.names:
		dtypes.append((n, data[n].dtype))
	for n in data2.keys():
		dtypes.append((n, data2[n].dtype))
	data_comb = np.zeros((len(data)), dtype = dtypes)
	for n in data.dtype.names:
		data_comb[n] = data[n]
	for n in data2.keys():
		data_comb[n] = data2[n]

	return data_comb

###################################################################################################

# Load the UPenn catalog of Meert et al. 2015. This function was adapted from Andrey Kravtsov's
# code. The photometric type determines which profile fits are loaded:
# 
# 1 = best fit, 2 = deVaucouleurs, 3 = Sersic, 4 = DeVExp, 5 = SerExp

def loadUPennCatalog(phot_type = 3): 

	def isSet(flag, bit):
		return (flag & (1 << bit)) != 0
	
	filenames = []
	filenames.append('UPenn_PhotDec_nonParam_rband.fits')
	filenames.append('UPenn_PhotDec_nonParam_gband.fits')
	filenames.append('UPenn_PhotDec_Models_rband.fits')
	filenames.append('UPenn_PhotDec_Models_gband.fits')
	filenames.append('UPenn_PhotDec_CAST.fits')
	filenames.append('UPenn_PhotDec_CASTmodels.fits')
	filenames.append('UPenn_PhotDec_H2011.fits')
	
	data_all = []
	names_in = []
	names_out = []
	dtypes = []
	for fn in filenames:
		file_path = cmn.data_dir + 'sdss_upenn/' + fn
		if fn.startswith('UPenn_PhotDec_Models_'):
			d = fits.open(file_path)[phot_type].data
		else:
			d = fits.open(file_path)[1].data
		data_all.append(d)
		names_in.append(d.dtype.names)
		names_out_file = []
		if 'rband' in fn:
			field_ext = '_r'
		elif 'gband' in fn:
			field_ext = '_g'
		elif 'H2011' in fn:
			field_ext = '_h11'
		else:
			field_ext = ''
		for f in d.dtype.names:
			if f == 'objid':
				f_out = 'objID'
			else:
				f_out = f + field_ext
			names_out_file.append(f_out)
			dtypes.append((f_out, d[f].dtype))
		names_out.append(names_out_file)
	
	# Append extra fields
	dtypes.append(('dL', float))
	dtypes.append(('dA', float))
	dtypes.append(('DM', float))
	dtypes.append(('M_r', float))
	dtypes.append(('M_petro_r', float))
	dtypes.append(('1/Vmax', float))
	dtypes.append(('color_gr', float))
	
	# Create structured array and transfer data
	data = np.zeros((len(data_all[0])), dtype = dtypes)
	for i in range(len(data_all)):
		d = data_all[i]
		for j in range(len(names_in[i])):
			data[names_out[i][j]] = d[names_in[i][j]]
	
	# Minimal quality cuts and flags recommended by Alan Meert; taken from Andrey Kravtsov's code
	fflag = data['finalflag_r']
	mask_valid = (data['petroMag'] > 0.0) & (data['petroMag'] < 100.0)
	mask_valid &= (data['kcorr_r'] > 0)
	mask_valid &= (data['m_tot_r'] > 0) & (data['m_tot_r'] < 100)
	mask_valid &= (isSet(fflag, 1) | isSet(fflag, 4) | isSet(fflag, 10) | isSet(fflag, 14))
	data = data[mask_valid]
	
	# Compute extra fields
	data['dL'] = cmn.cosmo.luminosityDistance(data['z']) / cmn.cosmo.h
	data['dA'] = cmn.cosmo.angularDiameterDistance(data['z']) / cmn.cosmo.h
	data['DM'] = cmn.cosmo.distanceModulus(data['z'])
	data['M_r'] = data['m_tot_r'] - data['extinction_r'] - data['DM'] - data['kcorr_r']
	data['M_petro_r'] = data['petroMag'] - data['extinction_r'] - data['DM'] - data['kcorr_r']
	data['1/Vmax'] = inverseVmax(data['M_r'], m_limit = m_r_limit, z_max = None)
	data['color_gr'] = (data['m_tot_g'] - data['extinction_g']) - (data['m_tot_r'] - data['extinction_r'])
	
	return data

###################################################################################################

# This function loads SDSS surface density profiles from a FITS file. The table is documented as
# follows:
# 
# bin      tinyint  1                        bin number (0..14)
# band	   tinyint  1                        u,g,r,i,z (0..4)
# profMean real     4  nanomaggies/arcsec^2	 Mean flux in annulus
# profErr  real	    4  nanomaggies/arcsec^2	 Standard deviation of mean pixel flux in annulus
# objID	   bigint   8                        links to the photometric object

def loadSDSSProfiles():

	hdu = fits.open(cmn.data_dir + 'sdss/sdss_profiles_dr8_m16.fit')
	data = hdu[1].data
	
	# Remove all galaxies where the count is either below 10 bins (averaged over filters)
	# or above 15 bins (the maximum)
	ids, inv, counts_allfilt = np.unique(data['objID'], return_inverse = True, return_counts = True)
	counts_bins = 0.2 * counts_allfilt
	idxs_weird = np.where((counts_bins < 10.0) | (counts_allfilt > 5 * 15))[0]
	print('Removing %d/%d galaxies from the profiles set.' % (len(idxs_weird), len(ids)))
	mask_keep = np.ones_like(data['objID'], bool)
	for idx in idxs_weird:
		mask_keep[inv == idx] = False
	data = data[mask_keep]
	
	# Rearrange into 3D array of [galaxy, filter, radial bin]
	ids, inv, counts_allfilt = np.unique(data['objID'], return_inverse = True, return_counts = True)
	n_gal = len(ids)
	ar_prf = np.ones((n_gal, 5, 15), float) * -1
	ar_err = np.ones((n_gal, 5, 15), float) * -1
	ar_prf[inv, data['band'], data['bin']] = data['profMean']
	ar_err[inv, data['band'], data['bin']] = data['profErr']
	
	return ids, ar_prf, ar_err

###################################################################################################

# This function avoids the comovingDistance function in Colossus because it is slower than the 
# other distances. We ignore any lower limit.

def inverseVmax(M_galaxies, m_limit = m_r_limit, z_max = None):
	
	dL = 10**(-5.0 + 0.2 * (m_limit - M_galaxies))
	z_lim = cmn.cosmo.luminosityDistance(dL * cmn.cosmo.h, inverse = True)
	if z_max is not None:
		z_lim = np.minimum(z_lim, z_max)
		dL = cmn.cosmo.luminosityDistance(z_lim)
	dC_max = dL / (1.0 + z_lim)
	Vmax = solid_angle / 3.0 * dC_max**3 / spectroscopic_completeness
	Vmax_inv_all = 1.0 / Vmax

	return Vmax_inv_all

###################################################################################################

# Compute the luminosity in solar units corresponding to an absolute magnitude definition.

def luminosity(data, mag_def = 'cmodel_r', evo_correction = True):

	M = data['M_%s' % mag_def]
	if evo_correction:
		M += 1.3 * data['z']
		
	filt = mag_def[-1]
	L = 10**(0.4 * (obs_utils.solar_mag[filt] - M))
	
	return L

###################################################################################################

# Function that checks whether a particular image cutout is already downloaded and does so if not.
# Technically, we would not need to add the object ID to the filename, but that way we can more
# easily identify which galaxy a cutout was meant to represent.
#
# The scale is the natural scale of SDSS in arcsec/pixel, according to 
# https://skyserver.sdss.org/dr2/en/tools/chart/chart.asp

def getSdssImage(obj_id, ra, dec, scale = sdss_pixel_scale, n_pix = 200, grid = False):

	if n_pix < 64:
		print('WARNING: SDSS image server does not produce images with fewer than 64 pixels.')
		n_pix = 64
		
	if not os.path.exists(sdss_img_dir):
		os.mkdir(sdss_img_dir)

	fn = sdss_img_dir + '%d_ra_%.4f_dec_%.4f_scale_%.4f_w_%d_h_%d.jpg' \
		% (obj_id, ra, dec, scale, n_pix, n_pix)

	if not os.path.exists(fn):
		opt_str = ''
		if grid:
			opt_str += 'G'
		url = 'http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?ra=%.8f&dec=%.8f&scale=%.2f&width=%i&height=%i' \
			% (ra, dec, scale, n_pix, n_pix)
		if opt_str != '':
			url += '&opt=%s' % opt_str
		urllib.request.urlretrieve((url), fn)
	
	im = Image.open(fn)
	
	return im

###################################################################################################

# Function that checks whether a particular spectrum is already downloaded and does so if not.
# Technically, we would not need to add the object ID to the filename, but that way we can more
# easily identify which galaxy a spectrum was meant to represent.
#
# The normalization constant was taken from https://classic.sdss.org/dr7/products/spectra/, since
# the units should be erg/s/cm2/A.
#
# This routine was inspired by its equivalent in AstroML.

def getSdssSpectrum(obj_id, plate, mjd, fiber):
	
	if not os.path.exists(sdss_spec_dir):
		os.mkdir(sdss_spec_dir)

	fn = sdss_spec_dir + '%d_plate_%d_mjd_%d_fiber_%d.fit' % (obj_id, plate, mjd, fiber)

	if not os.path.exists(fn):
		url = 'http://das.sdss.org/spectro/1d_26/%04i/1d/spSpec-%05i-%04i-%03i.fit' \
			% (plate, mjd, plate, fiber)
		urllib.request.urlretrieve((url), fn)

	hdu = fits.open(fn)
	spectrum = hdu[0].data[0] * 1E-17
	coeff0 = hdu[0].header['COEFF0']
	coeff1 = hdu[0].header['COEFF1']
	lam = 10**(coeff0 + coeff1 * np.arange(len(spectrum)))
	
	return lam, spectrum

###################################################################################################

def imageCollage(sdss_objs, n_rows, n_cols, n_pix = 150, scale = sdss_pixel_scale, panel_size = 1.5, 
				save = False, show = True):
	
	fig, axs = plt.subplots(n_rows, n_cols, figsize = (n_cols * panel_size, n_rows * panel_size))
	
	if 'jpg' not in fig.canvas.get_supported_filetypes():
		raise ValueError('Please make sure your matplotlib can show jpg images.')
	
	for obj_id_, ra_, dec_, ax in zip(sdss_objs['objID'], sdss_objs['ra'], sdss_objs['dec'], axs.flatten()):
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		im = getSdssImage(obj_id_, ra_, dec_, scale = scale, n_pix = n_pix)		
		ax.imshow(im, origin = 'upper')
		ax.set_aspect('auto')

	fig.subplots_adjust(hspace = 0.02, wspace = 0.02)
	if save:
		plt.savefig('image_collage.jpg', bbox_inches = 'tight')
	elif show:
		plt.show()
		
	return fig, axs

###################################################################################################

# Create an image collage that shows example galaxies along two axes. If an inverted axis is 
# desired (e.g., for magnitudes), pass the limits in the uninverted order (e.g., -22, -18) and 
# set invert_x / invert_y to True.
#
# The function also overplots contours of the sample, which are typically derived from a larger
# sample than the images (which should probably be of nearby galaxies). Thus, the user can pass
# two separate masks for images and contours. If they are None, the entire sample is used.

def imageCollageVariables(data_all, var_x, var_y, label_x, label_y, min_x, max_x, min_y, max_y, 
						invert_x = False, invert_y = False, n_bins_x = 10, n_bins_y = 10, 
						mask_images = None, image_size_kpc = 20.0, random_seed = 2023,
						plot_contours = True, mask_contours = None, 
						contour_levels = [0.99, 0.8, 0.6, 0.4, 0.2], contour_smoothing = 2.0,
						figsize = 8.0, save = False, show = True, fn_save = None):
	
	# ---------------------------------------------------------------------------------------------

	def pdf(x, hist, target):
		
		return np.sum(hist[hist > x]) - target
	
	# ---------------------------------------------------------------------------------------------

	def contourLabelFormat(x):
		
		idx = levels.index(x)
		lvl_pct = contour_levels[idx] * 100.0
		
		return r'%.0f\%%' % lvl_pct 

	# ---------------------------------------------------------------------------------------------
	
	# Create bins
	bin_edges_x = np.linspace(min_x, max_x, n_bins_x + 1)
	bin_edges_y = np.linspace(min_y, max_y, n_bins_y + 1)
	dx = np.abs(max_x - min_x) / n_bins_x
	dy = np.abs(max_y - min_y) / n_bins_y
	
	# Apply mask if necessary
	if mask_images is None:
		data_im = data_all
	else:
		data_im = data_all[mask_images]

	# Create figure	
	fig = plt.figure(figsize = (figsize, figsize))
	plt.subplots_adjust(left = 0.15, bottom = 0.15, right = 0.95, top = 0.95)
	ax = plt.gca()
	plt.xlabel(label_x)
	plt.ylabel(label_y)
	plt.xlim(min_x, max_x)
	plt.ylim(min_y, max_y)
	ax.tick_params(color = 'w', labelcolor = 'black', direction = 'in')
	ax.set_facecolor('k')
	if invert_x:
		ax.invert_xaxis()
	if invert_y:
		ax.invert_yaxis()
	if 'jpg' not in fig.canvas.get_supported_filetypes():
		raise ValueError('Please make sure your matplotlib can show jpg images.')

	# Go through bins
	np.random.seed(random_seed)
	for i in range(n_bins_x):
		mask_x = (data_im[var_x] >= bin_edges_x[i]) & (data_im[var_x] < bin_edges_x[i + 1])
		for j in range(n_bins_y):
			
			# Create sample for this bin and choose random galaxy
			mask_y = (data_im[var_y] >= bin_edges_y[j]) & (data_im[var_y] < bin_edges_y[j + 1])
			mask = mask_x & mask_y
			if np.count_nonzero(mask) == 0:
				continue
			idxs = np.where(mask)[0]
			idx = idxs[np.random.randint(0, len(idxs), 1)][0]

			# Get image of a particular size in kpc
			kpc_factor = data_im['dA'][idx] * 1000.0 * np.pi / 180.0 / 3600.0
			image_size_arcsec = image_size_kpc / kpc_factor
			image_size_pix = image_size_arcsec / sdss_pixel_scale
			image_size_pix = max(image_size_pix, 64)
			image_size_pix = min(image_size_pix, 256)
			image_size_pix = int(round(image_size_pix))
			im = getSdssImage(data_im['objID'][idx], data_im['ra'][idx], data_im['dec'][idx], n_pix = image_size_pix)		

			# Plot
			extent = []
			if invert_x:
				extent.extend([bin_edges_x[i + 1], bin_edges_x[i]])
			else:
				extent.extend([bin_edges_x[i], bin_edges_x[i + 1]])
			if invert_y:
				extent.extend([bin_edges_y[j + 1], bin_edges_y[j]])
			else:
				extent.extend([bin_edges_y[j], bin_edges_y[j + 1]])
			plt.imshow(im, extent = extent, origin = 'upper')
			
	plt.gca().set_aspect(dx / dy)

	# Plot density contours if desired. We weight the distribution by 1/Vmax.
	if plot_contours:
		if mask_contours is None:
			data_ct = data_all
		else:
			data_ct = data_all[mask_contours]

		# Note that we still need to transpose the result of histogram2d, but not invert the 
		# first axis as for imshow.
		hist, _, _ = np.histogram2d(data_ct[var_x], data_ct[var_y], bins = (50, 50), 
								range = [[min_x, max_x], [min_y, max_y]], weights = data_ct['1/Vmax'])
		hist /= np.sum(hist)
		hist = hist.T
		hist = scipy.ndimage.gaussian_filter(hist, contour_smoothing)
		
		# The levels are given as pdfs, as in, we want the contours that contain those fractions
		# of the sample. There does not seem to be an obvious way to do that with the contour()
		# function itself.
		levels = []
		for ct_level in contour_levels:  
			lvl = scipy.optimize.brentq(pdf, 0.0, 1.0, args = (hist, ct_level))   
			levels.append(lvl)
		
		# Plot contours and add labels. We need to trick the label function into not using the 
		# actual contour values but the cumulative values passed to the function.
		cts = ax.contour(hist, extent = [min_x, max_x, min_y, max_y], levels = sorted(levels), 
				linewidths = 0.6, colors = 'w', alpha = 0.3)
		ax.clabel(cts, inline = True, fmt = contourLabelFormat, fontsize = 8)

	# Finalize plot
	if save:
		if fn_save is None:
			fn_save = 'image_collage_vars.pdf'
		plt.savefig(fn_save)
	elif show:
		plt.show()
		
	return fig

###################################################################################################

def imageAndSpectrum(sdss_obj, n_pix = 150, scale = sdss_pixel_scale, color_def = 'fiber', 
					save = False, fn_out = None):

	if save:
		_, (ax0, ax1) = plt.subplots(1, 2, figsize = (9.0, 3.0))
		plt.subplots_adjust(wspace = 0.02)
	else:
		_, (ax0, ax1) = plt.subplots(1, 2, figsize = (12.0, 3.0))
		plt.subplots_adjust(wspace = -0.13)

	# Plot image	
	im = getSdssImage(sdss_obj['objID'], sdss_obj['ra'], sdss_obj['dec'], scale = scale, n_pix = n_pix)		
	ax0.axis('off')
	ax0.imshow(im, origin = 'upper')
	
	# Plot circle indicating the size of the spectral fiber
	r_fiber = sdss_fiber_size * 0.5 / scale
	circ = mpl.patches.Circle((n_pix * 0.5, n_pix * 0.5), r_fiber, color = 'w', fill = False, linestyle = 'solid', linewidth = 0.5)
	ax0.add_artist(circ)

	# Add text labels
	if color_def is not None:
		plt.text(0.95, 0.9, r'$(g-r)_{\rm %s} = %.2f$' % (color_def, sdss_obj['%sMag_g' % color_def] - sdss_obj['%sMag_r' % color_def]), 
			transform = ax0.transAxes, fontsize = 14, ha = 'right', color = 'w')
		
	# Download SDSS spectrum using plate number, epoch, and fiber ID
	# Normalizing by a high percentile is better than max in the presence of spikes.
	lbda, F = getSdssSpectrum(sdss_obj['objID'], sdss_obj['plate'], sdss_obj['mjd'], sdss_obj['fiberID'])
	F_plot = 0.5 * F / np.percentile(F, 99.5)
	
	plt.sca(ax1)
	plt.xlim(3000, 10500)
	plt.ylim(0, 0.52)
	plt.xlabel(r'$\lambda\ ({\rm \AA})$')
	plt.ylabel(r'$S(\lambda)\ \mathrm{or}\ F(\lambda)\ \mathrm{[arbitrary\ units]}$')

	for f, c, loc in zip('ugriz', filter_colors, [3500, 4650, 6150, 7500, 8750]):
		fn = cmn.data_dir + 'sdss/filter_%c.txt' % (f)
		if not os.path.exists(fn):
			raise ValueError('Could not find file %s.' % fn)
		filt = np.loadtxt(fn, unpack = True)
		plt.fill(filt[0], filt[2], ec = c, fc = c, alpha = 0.4)
		plt.text(loc, 0.04, f, color = c, ha = 'center', va = 'center', fontsize = 14)

	# Plot spectrum	
	ax1.plot(lbda, F_plot, '-', lw = 0.2, color = 'k')
	
	if save:
		if fn_out is None:
			fn_out = 'image_spec_%d.pdf' % (sdss_obj['objId'])
		plt.savefig(fn_out)
	else:
		plt.show()
	
	return

###################################################################################################
