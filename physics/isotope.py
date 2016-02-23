# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:25:42 2016

@author: lbignell
"""
import numpy as np
import scipy as sp
#from enum import Enum

class BetaDecay:
    '''
    This class handles the generation of a beta spectrum.
    
    For now it will just handle allowed decays, but when I get around to
    implementing shape factors, it will be more general.

    Base units for this class are: MeV, cm, sec
    '''
    #TO DO: Types = Enum('DecayType', 'Unknown Allowed First_Forbidden etc...')
    def __init__(self, name, Q, Z, A, betacharge, decaytype='Allowed', intensity=1):
        '''
        Instanciate the beta spec object.
        
        - name = a string to identify the beta branch.
        - Q = decay energy
        - Z = nuclear charge (in e)
        - A = atomic mass number
        - betacharge = +1 for positrons, -1 for electrons.
        - decaytype = type of beta decay (see enum)
        - intensity = branch intensity
        '''
        self.name = name
        self.Q = Q
        self.Z = Z
        self.A = A
        if abs(betacharge) != 1:
            print("ERROR!! The charge of the beta particle must be +/- 1! Setting to 1...")
            self.betacharge = 1
        else:
            self.betacharge = betacharge
        self.decaytype = decaytype
        self.fsconst = 1/137
        self.c = 3*1e10
        self.Melec = 0.511
        self.Rnucleus = self.nuclear_radius()
        self.intensity = intensity
        return
        
    def getname(self):
        return self.name

    def nuclear_radius(self):
        '''
        Calculate the nuclear radius using the approximate formula.
        '''
        r0 = 1.25*1e-13 #in cm
        return r0*self.A**(1/3)
        
    def Fermi_func(self, KE):
        '''
        Calculate the Fermi correction to the beta spectrum
        '''
        S = np.sqrt(1 - (self.fsconst**2)*(self.Z**2))
        E = (KE + self.Melec)
        pc = np.sqrt((E**2) - (self.Melec**2))
        eta = self.betacharge*self.fsconst*self.Z*E/(pc)
        return ((2*(1+S))/((sp.special.gamma(1+2*S))**2))* \
            ((2*(pc/self.c)*self.Rnucleus)**(2*S - 2))*(np.e**(np.pi*eta))*\
            ((abs(sp.special.gamma(S + (1j)*eta)))**2)

    def beta_spec(self, KE):
        '''
        Calculate the beta spectrum
        '''
        ff = self.Fermi_func(KE)
        return np.sqrt(KE**2 + 2*KE*self.Melec)*\
            ((self.Q-KE)**2)*\
            (KE + self.Melec)*ff
