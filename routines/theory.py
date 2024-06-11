###################################################################################################
#
# ASTR 620 - Galaxies
#
# theory.py: Routines related to large-scale structure, fluctuations, random fields
#
# (c) Benedikt Diemer, University of Maryland (based on similar code by Andrey Kratsov)
#
###################################################################################################

import numpy as np
import os
import pickle
import h5py
from scipy.interpolate import RegularGridInterpolator

from colossus.utils import utilities
from colossus.utils import constants
from colossus.cosmology import cosmology
from colossus.halo import mass_so

from routines import common as cmn

###################################################################################################

mu_ion = 0.59
mu_h = 1.4

###################################################################################################

def filterFourierSpace(kR, filt = 'tophat'):

    if filt == 'tophat':
        return 3.0 * (np.sin(kR) - kR * np.cos(kR)) / (kR)**3
    
    elif filt == 'gaussian':
        return np.exp(-0.5 * kR**2)
    
    else:
        print('Unknown filter, %s.' % filt)
        return

###################################################################################################

# Keyword args passed to this function are passed on to the power spectrum function

def gaussianRandomField(Lbox, N, z_ini, Pk_z0_func, **kwargs):

    cosmo = cosmology.getCurrent()

    # Set up grid coordinates in 3D
    N3 = N * N * N
    dx = Lbox / N
    dxi = 1.0 / dx
    #x1d = np.linspace(0, N - 1, N)
    #x, y, z = np.meshgrid(x1d, x1d, x1d)

    # Generate 3D array of Gaussian numbers with zero mean and unit dispersion
    delta = np.reshape(np.random.normal(0.0, 1.0, N3), (N, N, N))

    # Fourier transform delta field into delta_k, which now corresponds to a random
    # noise power spectrum in 3D
    delta_k = np.fft.fftn(delta)

    # We now want to weight each random component by the power spectrum, but we have
    # wave modes in kx, ky, and kz whereas the power spectrum is isotropic. So we 
    # comute the k value corresponding to each point in 3D k-space. The way the 
    # k-modes are arranged in the output of the FFT is not obvious, but the fftfreq()
    # function tells us that.
    k1d = np.fft.fftfreq(N) * 2.0 * np.pi
    kx, ky, kz = np.meshgrid(k1d, k1d, k1d)
    k = dxi * np.sqrt(kx**2 + ky**2 + kz**2)

    # The 0/0/0 corner needs to contain the mode that corresponds to one wave per
    # box size.
    k[0, 0, 0] = 2.0 * np.pi / Lbox

    # We now scale the modes with the power spectrum, or rather root(P)
    D = cosmo.growthFactor(z_ini)
    D2 = D**2
    Pk = Pk_z0_func(k, **kwargs) * D2
    delta_k *= np.sqrt(Pk)

    # The lowest mode in all dimensions is to the so-called "DC" mode, which corresponds
    # to a constant density throughout the box. We could set it to anything, but zero
    # makes sense.
    delta_k[0, 0, 0] = 0.0

    # Transform the Fourier field back into real space
    delta = np.real(np.fft.ifftn(delta_k))

    return k, delta, delta_k, D

###################################################################################################

# A fitting function to the results of Behroozi et al. 2013. Note that this function returns 
# stellar masses in units of Msun, not Msun / h. Mhalo must also be in Msun.

def mstarBehroozi13(Mhalo, z):

    def bfz(x, z):

        alpha_0 = -1.412
        alpha_a = 0.731
        delta_0 = 3.508
        delta_a = 2.608
        delta_z = -0.043
        gamma_0 = 0.316
        gamma_a = 1.319
        gamma_z = 0.279

        a = 1.0 / (1.0 + z)
        nu = np.exp(-4.0 * a**2)

        alpha = alpha_0 + nu * (alpha_a * (a - 1.0))
        delta = delta_0 + nu * (delta_a * (a - 1.0) + delta_z * z)
        gamma = gamma_0 + nu * (gamma_a * (a - 1.0) + gamma_z * z)

        f = -np.log10(10**(alpha * x) + 1.0) + delta * (np.log10(1.0 + np.exp(x)))**gamma / (1.0 + np.exp(10**(-x)))

        return f

    M1_0 = 11.514
    M1_a = -1.793
    M1_z = -0.251
    eps0 = -1.777
    eps_a = -0.006
    eps_z = 0.0
    eps_a2 = -0.119

    log_Mhalo = np.log10(Mhalo)
    a = 1.0 / (1.0 + z)
    nu = np.exp(-4.0 * a**2)
    log_M1 = M1_0 + nu * (M1_a * (a - 1.0) + z * M1_z)
    log_eps = eps0 + nu * (eps_a * (a - 1.0) + eps_z * z) + eps_a2 * (a - 1.0)

    log_Mstar = log_eps + log_M1 + bfz(log_Mhalo - log_M1, z) - bfz(0.0, z)

    return 10**log_Mstar

###################################################################################################

# A fitting function to the results of Kravtsov et al. 2018 at z = 0. Note that this function returns 
# stellar masses in units of Msun, not Msun / h. Mvir must also be in Msun.

def mstarKravtsov18(Mvir, scatter = False):

    if scatter:
        log_M1 = 11.39
        log_eps = -1.685
        alpha = -1.740
        delta = 4.335
        gamma = 0.531
    else:
        log_M1 = 11.43
        log_eps = -1.663
        alpha = -1.750
        delta = 4.290
        gamma = 0.595

    def f(x):
        
        t1 = -np.log10(10**(alpha * x) + 1.0)
        t2 = (np.log10(1.0 + np.exp(x)))**gamma
        t3 = 1.0 + np.exp(10**(-x))
        
        return t1 + delta * t2 / t3
    
    epsilon = 10**log_eps
    M1 = 10**log_M1
    log_Mstar = np.log10(epsilon * M1) + f(np.log10(Mvir / M1)) - f(0.0)

    return 10**log_Mstar

###################################################################################################

# Convenient wrapper for the mstarBehroozi13() function that creates the input array and restricts
# it to the mass range where the formula is defined.

def shmrBehroozi13():

    Mvir_beh = 10**np.linspace(np.log10(1E9), np.log10(1E16), 200)
    mstar_beh = mstarBehroozi13(Mvir_beh, 0.0)
    mask = (mstar_beh >= 10**7.25)
    Mvir_beh = Mvir_beh[mask]
    mstar_beh = mstar_beh[mask]
    
    return Mvir_beh, mstar_beh / Mvir_beh

###################################################################################################

# Convenient wrapper for the mstarKravtsov18() function that creates the input array and restricts
# it to the mass range where the formula is defined.

def shmrKravtsov18():

    Mvir_kra = 10**np.linspace(10.0, 15.2, 200)
    mstar_kra = mstarKravtsov18(Mvir_kra, scatter = True)
    
    return Mvir_kra, mstar_kra / Mvir_kra

###################################################################################################

# SHMR for dwarfs according to Munshi et al. 2021, including scatter.

def shmrMunshi21():

    Mh_crit = 1E10
    norm = 1E7
    Mvir_mun = 10**np.linspace(7.0, 11.0, 50)
    mstar_mun = np.zeros_like(Mvir_mun)
    scatter_mun = np.zeros_like(Mvir_mun)
    mask = (Mvir_mun > Mh_crit)
    mstar_mun[mask] = norm * (Mvir_mun[mask] / Mh_crit)**1.9
    scatter_mun[mask] = 0.3
    mask = np.logical_not(mask)
    mstar_mun[mask] = norm * (Mvir_mun[mask] / Mh_crit)**2.81
    scatter_mun[mask] = 0.3 - 0.39 * (np.log10(Mvir_mun[mask]) - 10.0)
    y_mun = mstar_mun / Mvir_mun
    y_mun_lo = y_mun / 10**scatter_mun
    y_mun_hi = y_mun * 10**scatter_mun
    
    return Mvir_mun, y_mun, y_mun_lo, y_mun_hi

###################################################################################################

def coolingInterpolator(Z):
    
    pickle_path = cmn.data_dir + 'tmp_files/coolrate_ps20_Z%.4f' % Z
    data_path = cmn.data_dir + 'cooling_tables/ploeckinger_schaye_20_z0.hdf5'
    
    if os.path.exists(pickle_path):
        pFile = open(pickle_path, 'rb')
        dic = pickle.load(pFile)
        pFile.close()

        interp = dic['interp']
        Tmin = dic['Tmin']
        Tmax = dic['Tmax']
        nmin = dic['nmin']
        nmax = dic['nmax']
    
    else:
        f = h5py.File(data_path, 'r')
        bins_Z = np.array(f['bins_metallicity'])
        bins_T = np.array(f['bins_temperature'])
        bins_n = np.array(f['bins_density'])
        cooling_primordial = np.array(f['cooling_primordial'])
        cooling_metal = np.array(f['cooling_metal'])
        heating_primordial = np.array(f['heating_primordial'])
        heating_metal = np.array(f['heating_metal'])
        f.close()
        
        # Total = Primordial + Metal, and cooling = Lambda - Gamma
        cool_rate = 10**cooling_primordial + 10**cooling_metal
        heat_rate = 10**heating_primordial + 10**heating_metal
        total_rate = cool_rate - heat_rate
        total_rate[total_rate <= 1E-40] = 1E-40
        total_rate = np.log10(total_rate)
        rgi1 = RegularGridInterpolator((bins_T, bins_Z, bins_n), total_rate, method = 'linear')
    
        # Convert to metallicity as defined in Ploeckinger & Schaye
        Z_use = np.log10(Z * cmn.Z_SOLAR / 0.0134)
        grid_T, grid_n = np.meshgrid(bins_T, bins_n, indexing = 'ij')
        grid_Z = np.ones_like(grid_T) * Z_use
        grid_T = grid_T.flatten()
        grid_n = grid_n.flatten()
        grid_Z = grid_Z.flatten()
        grid = np.stack((grid_T, grid_Z, grid_n)).T
    
        # Evaluate grid and create new interpolator; we stick with log-linear interpolation
        # because the cooling functions can contain sharp drops that might lead to spline
        # ringing.        
        grid_fixed_Z = rgi1(grid)
        grid_fixed_Z = grid_fixed_Z.reshape(len(bins_T), len(bins_n))
        interp = RegularGridInterpolator((bins_T, bins_n), grid_fixed_Z, method = 'linear')
    
        Tmin = 10**bins_T[0]
        Tmax = 10**bins_T[-1]
        nmin = 10**bins_n[0]
        nmax = 10**bins_n[-1]
    
        # Save to pickle
        dic = {}
        dic['interp'] = interp
        dic['Tmin'] = Tmin
        dic['Tmax'] = Tmax
        dic['nmin'] = nmin
        dic['nmax'] = nmax
        
        pickle_path_root = os.path.dirname(pickle_path)
        if not os.path.exists(pickle_path_root):
            os.mkdir(pickle_path_root)

        output_file = open(pickle_path, 'wb')
        pickle.dump(dic, output_file, pickle.HIGHEST_PROTOCOL)
        output_file.close()
    
    return Tmin, Tmax, nmin, nmax, interp
    
###################################################################################################

# This function returns the cooling rate Lambda, NOT Lambda / n^2!

def coolingRate(Z, T, n):
    
    _, _, _, _, interp = coolingInterpolator(Z)
    
    # Evaluate
    T_array, is_ar1 = utilities.getArray(T)
    n_array, is_ar2 = utilities.getArray(n)
    
    is_ar = (is_ar1 or is_ar2)
    if (len(T_array) != 1) and (len(n_array) != 1) and ((len(T_array) != len(n_array))):
        raise Exception('Different input dimensions')
    
    if len(T_array) != len(n_array):
        if len(T_array) == 1:
            T_array = np.ones_like(n_array) * T_array[0]
        elif len(n_array) == 1:
            n_array = np.ones_like(T_array) * n_array[0]
        else:
            raise Exception('Zero length input.')

    grid = np.stack((T_array, n_array)).T
    grid = np.log10(grid)
    Lambda = 10**interp(grid) * n_array**2

    if not is_ar: 
        Lambda = Lambda[0]
    
    return Lambda

###################################################################################################

# The virial density, including dark matter, in g/cm^3

def rhoVir(z):
    
    return mass_so.densityThreshold(z, 'vir') * constants.MSUN * cmn.cosmo.h**2 / constants.KPC**3

###################################################################################################

# Virial temperature, with Mvir given in Msun

def TvirFromMvir(Mvir, z, mu = mu_ion):
    
    Mvir_h = Mvir * cmn.cosmo.h
    Rvir_h = mass_so.M_to_R(Mvir_h, z, 'vir')
    Vvir = np.sqrt(constants.G * Mvir_h / Rvir_h) * 1E5
    T = mu * constants.M_PROTON / constants.KB / 3 * Vvir**2

    return T

###################################################################################################

# The mass corresponding to a given virial temperature

def MvirFromTvir(Tvir, z, mu = mu_ion):
    
    Mvir = (3 * constants.KB * Tvir / (mu * constants.M_PROTON * constants.G_CGS))**(3.0 / 2.0) \
        * np.sqrt(3 / (4 * np.pi * rhoVir(z))) / constants.MSUN
    
    return Mvir

###################################################################################################

# Cooling time times n; note that Lambda is the cooling rate, NOT Lambda/n^2. 

def n_tcool(Z, T, n):
    
    Lambda_T = coolingRate(Z, T, n)
    tcool = 3.0 * constants.KB * T / (2.0 * Lambda_T) * n**2
    
    return tcool

###################################################################################################

# The "virial cooling time" corresponding to the viral density

def tcoolFromMvir(Mvir, z, Z):

    Tvir = TvirFromMvir(Mvir, z)
    rho_vir = rhoVir(z)
    fb = cmn.cosmo.Ob(0.0) / cmn.cosmo.Om(0.0)
    n_vir = rho_vir * fb / (mu_h * constants.M_PROTON)
    tcool = n_tcool(Z, Tvir, n_vir) / n_vir

    return tcool

###################################################################################################

# The free-fall time from density, which should include DM.

def tFreeFall(rho):

    return np.sqrt(3.0 * np.pi / (32.0 * constants.G_CGS * rho))

###################################################################################################

def tFreeFallVir(z):
    
    rho_vir = rhoVir(z)
    tff = tFreeFall(rho_vir)
    
    return tff

###################################################################################################
