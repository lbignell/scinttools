# -*- coding: utf-8 -*-
"""
Created on Tue Feb  23 15:28:05 2016

@author: lbignell
"""

import numpy as np
import scipy as sp
import scinttools.physics.material
import scinttools.physics.isotope

class TDCR:
    '''
    Calculate efficiencies, etc. for a TDCR detector.
    Limitations (to be fixed):
    - Using the same collection efficiency for each tube.
    - A single, user-entered QE.
    
    Base units: MeV, cm, sec, g.
    '''
    def __init__(self, CollEff, QE, scint=None, branch=None):
        self.CollEff = CollEff
        self.QE = QE
        self.scint = scint
        self.branch = branch
        if scint is not None:
            print('Registering scintillator: {0}'.format(scint.getname()))
        if branch is not None:
            print('Registering beta branch: {0}'.format(branch.getname()))
        return

    def setscint(self, scint):
        self.scint = scint

    def setbranch(self, branch):
        self.branch = branch

    def _effmodel(self, KE, Mparticle, Zparticle, factor):
        return (1 - np.e**(-1*self.CollEff*self.QE*self.scint.LY*
            self.scint.quenched_en(KE, Mparticle, Zparticle)[0]*factor))

    def eff_nPMT_beta(self, n, factor=1):
        '''
        Calculate the efficiency of 1 PMT to the registered scintillator and
        beta decay branch.
        
        Arguments:
        - number of PMTs
        - Optional factor to model efficiency extrapolation.
        '''
        if self.scint is None:
            print('ERROR!! No scintillator has been registered.')
            return None
        elif self.branch is None:
            print('ERROR!! No branch has been registered.')
            return None
        eff = lambda KE, Mparticle, Zparticle, n:\
            self.branch.beta_spec(KE)*self._effmodel(KE,Mparticle,Zparticle,factor)**n
        #I should add a verbosity option later to plot the efficiency spectrum.
        unnorm = sp.integrate.quad(eff, 0, self.branch.Q, 
                                   args=(self.branch.Melec,self.branch.betacharge,n))
        return unnorm[0]/sp.integrate.quad(self.branch.beta_spec, 0, self.branch.Q)[0]
        
    def eff_nPMT_monoenergetic(self, KE, Mparticle, Zparticle, n, factor=1):
        '''
        Calculate the efficiency of 1 PMT to the registered scintillator and
        a monoenergetic emission.
        
        Arguments:
        - Particle energy (MeV)
        - Particle rest mass (MeV/c^2)
        - Particle charge (e)
        - number of PMTs
        - Optional factor to model efficiency extrapolation.
        '''
        if self.scint is None:
            print('ERROR!! No scintillator has been registered.')
            return None
        return self._effmodel(KE,Mparticle,Zparticle,factor)**n

    def EfficiencyExtrap(self, factorvals, endpoint, Mparticle, Zparticle):
        pass

