###################################################################################################
#
# ASTR 620 - Galaxies
#
# models.py: Semi-analytical modeling of galaxy evolution
#
# (c) Benedikt Diemer, University of Maryland (based on similar code by Andrey Kratsov)
#
###################################################################################################

import numpy as np
import scipy

from colossus.halo import mass_so

from routines import common as cmn

###################################################################################################

# Indices of the different components (halo, gas, stars, metals) in mass array
IH = 0
IG = 1
IS = 2
IZ = 3

###################################################################################################

class GalaxyModel():
    
    def __init__(self, 
                 # Initial conditions
                 z_ini, Mh_ini, 
                 # Parameters: metallicity
                 Z_in = 2E-5, yZ = 0.069,
                 # Parameters: MAH model
                 model_mah = 'neistein08', 
                 # Parameters: reionization model
                 model_reion = 'okamoto08', z_reion = 9.0, M_c1_early = 1E7, M_c1_0 = 2E10, alpha1 = 2.0, 
                 # Parameters: cooling model
                 model_cooling = 'cutoff', M_c2_0 = 1E12, alpha2 = 1.0, 
                 # Parameters: SFR model
                 model_sfr = 'depl_time', t_depl = 2.0, R_loss = 0.46,
                 # Parameters: wind/feedback model
                 model_wind = 'muratov15_star', eta_fixed = 1.0, eta_min = 2.0, eta_max = 300.0):
        
        # Cosmology
        self.cosmo = cmn.cosmo
        self.fb = self.cosmo.Ob(0.0) / self.cosmo.Om(0.0)

        # Initial conditions        
        self.z_ini = z_ini
        self.n_comp = 4
        self.Mx_ini = np.zeros((self.n_comp), float)
        self.Mx_ini[IH] = Mh_ini
        self.Mx_ini[IG] = Mh_ini * self.fb
        self.Mx_ini[IZ] = self.Mx_ini[IG] * Z_in
        
        # Parameters
        self.Z_in = Z_in
        self.yZ = yZ

        self.model_mah = model_mah

        self.model_reion = model_reion
        self.z_reion = z_reion
        self.M_c1_early = M_c1_early
        self.M_c1_0 = M_c1_0
        self.alpha1 = alpha1

        self.model_cooling = model_cooling
        self.M_c2_0 = M_c2_0
        self.alpha2 = alpha2
        
        self.model_sfr = model_sfr
        self.t_depl = t_depl
        self.R_loss = R_loss
        
        self.model_wind = model_wind
        self.eta_fixed = eta_fixed
        self.eta_min = eta_min
        self.eta_max = eta_max
        if self.eta_min >= self.eta_max:
            raise Exception('eta_min must be smaller than eta_max.')
        
        if self.model_mah == 'massfunction':
            self.createHaloEvolutionTable()
            
        return 

    # ---------------------------------------------------------------------------------------------

    def dD_dt(self, t):
        
        z = self.cosmo.age(t, inverse = True)
        dD_dz = self.cosmo.growthFactor(z, derivative = 1)
        dz_dt = self.cosmo.age(t, derivative = 1, inverse = True)
        
        return dD_dz * dz_dt

    # ---------------------------------------------------------------------------------------------

    def dMh_dt(self, t, Mx_cur):
        
        z = self.cosmo.age(t, inverse = True)
         
        if self.model_mah == 'neistein08':
    
            dDdt = self.dD_dt(t)
            D = self.cosmo.growthFactor(z)
            ret = 1.06E12 * (Mx_cur[IH] / 1E12)**1.14 * dDdt / D**2
        
        else:
            raise Exception('Unknown MAH model, %s.' % self.model_mah)
        
        return ret

    # ---------------------------------------------------------------------------------------------

    def eps_g1(self, t, Mx_cur):
                
        if self.model_reion == 'none':
            
            fg = np.ones_like(Mx_cur[0])
            
        elif self.model_reion == 'okamoto08':

            z = self.cosmo.age(t, inverse = True)
            if z > self.z_reion:
                M_c1 = self.M_c1_early
            else:
                M_c1 = self.M_c1_0 * np.exp(-0.63 * z)
            alpha = self.alpha1
            fg = (1.0 + (2.0**(alpha / 3) - 1) * (Mx_cur[IH] / M_c1)**-alpha)**(-3.0 / alpha)
            
        else:
            raise Exception('Unknown reionization model, %s.' % self.model_reion)
        
        return fg

    # ---------------------------------------------------------------------------------------------

    def eps_g2(self, t, Mx_cur):

        if self.model_cooling == 'none':
            
            eps = np.ones_like(Mx_cur[0])
            
        elif self.model_cooling == 'cutoff':
            
            z = self.cosmo.age(t, inverse = True)
            M_c2 = self.M_c2_0 * np.sqrt(self.cosmo.Ez(z)) * mass_so.deltaVir(z) / mass_so.deltaVir(0.0)
            alpha = self.alpha2
            eps = 1.0 - (1.0 + (2.0**(alpha / 3) - 1) * (Mx_cur[IH] / M_c2)**-alpha)**(-3.0 / alpha)
        
        else:
            raise Exception('Unknown cooling model, %s.' % self.model_cooling)
        
        return eps

    # ---------------------------------------------------------------------------------------------

    def sfr(self, t, Mx_cur):
 
        if self.model_sfr == 'depl_time':
                
            ret = Mx_cur[IG] / self.t_depl
            
        else:
            raise Exception('Unknown SFR model, %s.' % self.model_sfr)
       
        return ret

    # ---------------------------------------------------------------------------------------------

    def eta(self, t, Mx_cur):

        if self.model_wind == 'none':
            
            ret = np.zeros_like(Mx_cur[0])
            
        elif self.model_wind == 'fixed':
            
            ret = np.ones_like(Mx_cur[0]) * self.eta_fixed
            
        elif self.model_wind == 'muratov15_star':
        
            ret = np.maximum(0.0, 3.6 * (np.maximum(Mx_cur[IS], 1E-20) / 1E10)**-0.35 - 4.5)

        else:
            raise Exception('Unknown wind model, %s.' % self.model_wind)

        ret = np.maximum(ret, self.eta_min)
        ret = np.minimum(ret, self.eta_max)
        
        return ret

    # ---------------------------------------------------------------------------------------------

    # An array of dM / dt for all tracked components. This is the function that is integrated in 
    # the evolve() routine.

    def massRates(self, t, Mx_cur):
        
        # Halo accretion rate
        dMh_dt = self.dMh_dt(t, Mx_cur)
        
        # Efficiency factors
        eps_g1 = self.eps_g1(t, Mx_cur)
        eps_g2 = self.eps_g2(t, Mx_cur)
        eta = self.eta(t, Mx_cur)
        
        # SFR
        sfr = self.sfr(t, Mx_cur)
        
        # Effective gas accretion rate
        dMg_dt_in = dMh_dt * self.fb * eps_g1 * eps_g2
        dMg_dt_out = sfr * (1.0 - self.R_loss + eta)
        dMg_dt = dMg_dt_in - dMg_dt_out
        
        # Change in stellar mass
        dMs_dt = sfr * (1.0 - self.R_loss)
        
        # Change in metals
        dMZ_dt = dMg_dt_in * self.Z_in + sfr * ((1.0 - self.R_loss) * self.yZ \
                                        - (1.0 - self.R_loss + eta) * Mx_cur[IZ] / Mx_cur[IG])
    
        return [dMh_dt, dMg_dt, dMs_dt, dMZ_dt]

    # ---------------------------------------------------------------------------------------------

    # Evolve the model forward in time, starting at the initial halo mass set when the model was
    # created.

    def evolve(self, t_eval = None, z_eval = None, z_final = 0.0):

        t_ini = self.cosmo.age(self.z_ini)
        t_final = self.cosmo.age(z_final)
        
        if t_eval is not None and z_eval is not None:
            raise Exception('Can only take either t_eval or z_eval.')
        elif t_eval is not None:
            pass
        elif z_eval is not None:
            t_eval = self.cosmo.age(z_eval)
        else:
            t_eval = np.linspace(t_ini, t_final, 500)
        
        ret = scipy.integrate.solve_ivp(self.massRates, [t_ini, t_final], self.Mx_ini, t_eval = t_eval, 
                                            rtol = 1E-5, vectorized = False)
        t = ret['t']
        Mx = ret['y']
        z = self.cosmo.age(t, inverse = True)
        a = 1.0 / (1.0 + z)
        dMx = np.zeros_like(ret['y'])
        for i in range(len(t)):
            dMx[:, i] = self.massRates(t[i], Mx[:, i])
        
        return t, z, a, Mx, dMx

###################################################################################################
