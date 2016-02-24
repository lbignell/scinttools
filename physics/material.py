# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:36:27 2016

@author: lbignell
"""
import numpy as np
import scipy as sp
#from enum import Enum

class Scintillator:
    '''
    A class to handle the scintillator quenching, etc.
    
    Base units for this class are: MeV, cm, sec, g
    '''
    def __init__(self, name, LY, kB, density=1, ZonA=0.55509, I=75*1e-6):
        '''
        Instanciate the scintillator.
        
        Arguments:
        - A string name for the scintillator.
        - Light yield in photons/MeV
        - Birks quenching factor in cm/MeV
        - Scintillator density (defaults to water) in g/cm^3
        - Mean Z on A (defaults to water, PDG booklet)
        - Mean ionisation potential (defaults to water, ICRU49) in MeV
        '''
        self.name = name
        self.LY = LY
        self.kB = kB
        self.density = density
        self.ZonA = ZonA
        self.I = I
        self.K = 0.307075 #MeV cm^2 Mol^-1, from PDG booklet
        self.Melec = 0.511
        return
        
    def getname(self):
        '''
        Return the name of the scintillator
        '''
        return self.name

    def _BetheBloch(self, KE, Mparticle, Zparticle):
        gamma = KE/Mparticle + 1
        betasq = 1 - 1/(gamma**2)
        Tmax = (2*self.Melec*betasq*gamma**2)/(1 + 
            (2*gamma*self.Melec/Mparticle) +(self.Melec/Mparticle)**2)
        #We should be able to get by without the density correction as we'll be below MIP.
        return self.K*(Zparticle**2)*self.ZonA*(1/betasq)*\
            (0.5*np.log((2*self.Melec*betasq*(gamma**2)*Tmax)/(self.I**2)) - betasq)

    def BetheBloch(self, KE, Mparticle, Zparticle):
        '''
        Returns the stopping power calculated using the Bethe-Bloch function.
        Values below 100 eV are linearly interpolated to 0.
        Units of returned value are MeVcm^2/g
        
        Arguments:
        - Particle Kinetic Energy (MeV)
        - Particle Mass (MeV/c^2)
        - Particle Charge (e)
        The density correction is ignored, so this isn't valid much above MIP.
        '''
        if KE>0.0001:
            return self._BetheBloch(KE, Mparticle, Zparticle)
        elif KE>0:
            dEdX_100eV = self._BetheBloch(0.0001, Mparticle, Zparticle)
            return np.interp(KE, [0, 0.0001], [0, dEdX_100eV])            
        else:
            print('ERROR!! Negative particle energy! Setting dE/dx to 0...')
            return 0

    def quenched_en(self, KE, Mparticle, Zparticle):
        '''
        Calculate the apparent (from the number of photons) energy of a particle
        interaction in the scintillator.
        
        Arguments:
        - Particle Kinetic Energy (MeV)
        - Particle Mass (MeV/c^2)
        - Particle Charge (e)
        '''
        Birks_fn = lambda KE,Mparticle,Zparticle: 1/(1 + 
            self.kB*self.BetheBloch(KE, Mparticle, Zparticle))
        return sp.integrate.quad(Birks_fn, 0, KE, args=(Mparticle, Zparticle))

    def setkB(self, kB):
        self.kB = kB
        return