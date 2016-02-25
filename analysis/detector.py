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

    - The efficiency extrapolation can only be done for a single beta transition.
    
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

    def _effmodel(self, KE, Mparticle, Zparticle, factor,verbose=0):
        return (1 - np.e**(-1*self.CollEff*self.QE*self.scint.LY*
            self.scint.quenched_en(KE, Mparticle, Zparticle,verbose-1)[0]*factor))

    def eff_nPMT_beta(self, n, factor=1, verbose=0):
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
            self.branch.beta_spec(KE)*self._effmodel(KE,Mparticle,Zparticle,factor,verbose-1)**n
        #I should add a verbosity option later to plot the efficiency spectrum.
        unnorm = sp.integrate.quad(eff, 0, self.branch.Q, 
                                   args=(self.branch.Melec,self.branch.betacharge,n))
        if verbose>0:
            print('TDCR.eff_nPMT_beta: n = {0}, factor= {1}, integral = {2} +/- {3}'.format(
                n, factor, unnorm[0], unnorm[1]))
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

    def eff_extrap_beta(self, factorvals, kBvals, verbose=0):
        '''
        Calculate the apparent activity for an efficiency extrapolation measurement,
        for various kB values and TDCRs.
        
        Note that the *true* kB value is that given in the registered scintillator.
        Arguments:
        
        - A list of values to multiply the detection efficiency by.

        - A list of kB values (cm/MeV)
        
        The beta spectrum and scintillator will be inferred from the
        currently-registered objects.
        '''
        if self.scint is None:
            print('ERROR!! No scintillator has been registered.')
            return None
        elif self.branch is None:
            print('ERROR!! No branch has been registered.')
            return None
        kBtrue = self.scint.kB
        #Calculate the detection efficiency vs factor for the true kB.
        #Array stores efficiency[#PMTs][factor]
        effn_true = [[self.eff_nPMT_beta(n+1,factor,verbose-1) for factor in factorvals]
                     for n in range(3)]
        TDCR_true = [effn_true[2][idx]/effn_true[1][idx] for idx in range(len(factorvals))]        
        #This bit needs careful thought.
        #The TDCR is observed, so is independent of the estimated kB.
        #So the problem is to find effn_true[TDCR_wrong], as this is what the
        #efficiency actually is, as opposed to what we calculate with a
        #wrong kB.
        #The ratio effn_wrong[TDCR_wrong]/effn_true[TDCR_wrong] is therefore
        #proportional to the apparently measured activity.
        activity_meas = []
        TDCR_meas = []
        for thiskB in kBvals:
            if verbose>0:
                print('TDCR.eff_extrap_beta: kB = {:0.3f}'.format(thiskB))
            self.scint.setkB(thiskB)
            effn_wrong = [[self.eff_nPMT_beta(n+1,factor,verbose-1) for factor in factorvals]
                          for n in range(3)]
            TDCR_wrong = [effn_wrong[2][idx]/effn_wrong[1][idx] for idx in range(len(factorvals))]
            activity_meas += [[effn_wrong[1][idx]/np.interp(TDCR_wrong[idx],
                              TDCR_true,effn_true[1][:], left=np.inf,right=np.inf) 
                              for idx in range(len(factorvals))]]
            TDCR_meas += [TDCR_wrong]
        #reset kB to its original value
        self.scint.setkB(kBtrue)        
        return activity_meas, TDCR_meas, TDCR_true, effn_true
