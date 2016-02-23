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
            self.scint.get_quenched_en(KE, Mparticle, Zparticle)*factor))

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
            self.branch.get_beta_spec(KE)*self._effmodel(KE,Mparticle,Zparticle,factor)**n
        #I should add a verbosity option later to plot the efficiency spectrum.
        return sp.integrate.quad(eff, 0, self.branch.Q, 
                                 args=(self.branch.Melec,self.branch.betacharge))

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

kB = 0.01 #cm/MeV
QE = 0.2
CollEff = 0.7*0.33 #1/3 of the collection efficiency we saw in the PSD measurements
LY = 100 #photons/MeV
Q = 0.156476
Z=6
A=14
KE = np.linspace(0.00001, Q, 1000)
Zparticle = 1
RestMass = 0.511

QuenchedEn = [[BirksIntegral(En, kB, Zparticle, RestMass) for En in KE] for kB in kBvals]
QuenchedEn = [np.transpose(array) for array in QuenchedEn[:]]

MeanNumPE = [np.multiply(QuenchedEn[theidx][0], CollEff*QE*LY) for theidx in range(len(QuenchedEn))]

print("Calculating Efficiencies...")
Eff_1PMT = [np.sum(spec*(1-np.e**(-MeanNumPE[idx]))) for idx in range(len(MeanNumPE))]
Eff_2PMT = [np.sum(spec*((1-np.e**(-MeanNumPE[idx]))**2)) for idx in range(len(MeanNumPE))]
Eff_3PMT = [np.sum(spec*((1-np.e**(-MeanNumPE[idx]))**3)) for idx in range(len(MeanNumPE))]


#This will calculate the actually-measured TDCRs for an efficiency extrapolation, for various REAL kB values.
#To model the INCORRECT kB values I need to model the difference between this real value and some fake value...
#I can generate the fake value from these data by looking at where the TDCR_real = TDCR_fake and selecting 
#the appropriate lambda value (maybe?)
#Note that activity is the detection efficiency multiplied by a measured count rate.
#I can predict the measured count rate with the REAL kB, and predict the modeled detection efficiency as
#that predicted by the FAKE kB, for the value of TDCR arrived at with the REAL kB... Clear as mud?
#
#I can either fit the Eff vs TDCR plots and use the fitted functions to compare REAL and FAKE,
#or use a shit ton of points
factorvals = np.linspace(0.5,2, 1001)
Eff_1PMT_var = [[np.sum(spec*(1-np.e**(-np.multiply(QuenchedEn[theidx][0], factor*CollEff*QE*LY)))) for factor in factorvals] for theidx in range(len(QuenchedEn))]
Eff_2PMT_var = [[np.sum(spec*((1-np.e**(-np.multiply(QuenchedEn[theidx][0], factor*CollEff*QE*LY)))**2)) for factor in factorvals] for theidx in range(len(QuenchedEn))]
Eff_3PMT_var = [[np.sum(spec*((1-np.e**(-np.multiply(QuenchedEn[theidx][0], factor*CollEff*QE*LY)))**3)) for factor in factorvals] for theidx in range(len(QuenchedEn))]

#Eff_1PMT_var = [[np.sum((1-np.e**(-np.multiply(np.add(QuenchedEnAlpha[theidx][0],QuenchedEnCE[theidx][0],QuenchedEnAuger[theidx][0]), factor*CollEff*QE*LY)))) for factor in factorvals] for theidx in range(len(QuenchedEnAlpha))]
#Eff_2PMT_var = [[np.sum((1-np.e**(-np.multiply(np.add(QuenchedEnAlpha[theidx][0],QuenchedEnCE[theidx][0],QuenchedEnAuger[theidx][0]), factor*CollEff*QE*LY)))**2) for factor in factorvals] for theidx in range(len(QuenchedEnAlpha))]
#Eff_3PMT_var = [[np.sum((1-np.e**(-np.multiply(np.add(QuenchedEnAlpha[theidx][0],QuenchedEnCE[theidx][0],QuenchedEnAuger[theidx][0]), factor*CollEff*QE*LY)))**3) for factor in factorvals] for theidx in range(len(QuenchedEnAlpha))]


print("Done")

colormap = plt.cm.gist_ncar
plt.rc('axes', color_cycle=[colormap(i) for i in np.linspace(0, 0.9, np.shape(Eff_3PMT_var)[0])])
plt.figure()
labels = []
for idx in range(np.shape(Eff_2PMT_var)[0]):
    plt.plot(Eff_2PMT_var[idx], np.divide(Eff_3PMT_var[idx], Eff_2PMT_var[idx]))
    labels.append(r'kB = %.3f cm/MeV' % (kBvals[idx]))

plt.legend(labels, ncol=1, loc='lower right', 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True)

print("Mapping Eff to TDCR...")
#I'll do the analysis for a single kB value first, then generalise...
thekB = 0.044
thekBIdx = np.searchsorted(kBvals, thekB, side='left')
#find whether to use thekBIdx or thekBIdx-1
if (thekB - kBvals[thekBIdx-1])<(kBvals[thekBIdx] - thekB): thekBIdx -= 1
#factorIdx = np.min([np.searchsorted(factorvals,1),len(factorvals)-1])
theNcounts = Eff_2PMT_var[thekBIdx]#[:factorIdx]

#Make the TDCRs to make life easier:
#TDCR_var = [np.divide(Eff_3PMT_var[idx][:factorIdx],Eff_2PMT_var[idx][:factorIdx])\
TDCR_var = [np.divide(Eff_3PMT_var[idx],Eff_2PMT_var[idx])\
    for idx in range(np.shape(Eff_2PMT_var)[0])]
#This long and complicated line calculates the detection efficiency that would be
#calculated, for a FAKE kB, given a TDCR observed using a REAL kB.
#X = TDCR, Y = Eff_D
truexvals = TDCR_var[thekBIdx]
trueyvals = Eff_2PMT_var[thekBIdx]
thisEff_var = [np.interp(TDCR_var[idx], truexvals, trueyvals) for idx in range(np.shape(Eff_2PMT_var)[0])]

#Old, crappier code
#maxidx = np.shape(Eff_2PMT_var)[1]-1
#lowidx = [[np.min([np.searchsorted(TDCR_var[idx][:],TDCR_var[thekBIdx][i]), maxidx]) \
#    for i in range(np.shape(TDCR_var)[1])] for idx in range(np.shape(Eff_2PMT_var)[0])]
#thisEff_var = [[Eff_2PMT_var[idx][lowidx[i][idx]] \
#thisEff_var = [[Eff_2PMT_var[idx][lowidx[idx][i]] + \
#    ((TDCR_var[thekBIdx][i] - TDCR_var[idx][lowidx[idx][i]])/
#    (TDCR_var[idx][lowidx[idx][i]+1] - TDCR_var[idx][lowidx[idx][i]]))* \
#    (Eff_2PMT_var[idx][lowidx[idx][i]+1]-Eff_2PMT_var[idx][lowidx[idx][i]]) \
#    for i in range(np.shape(TDCR_var)[1])] for idx in range(np.shape(Eff_2PMT_var)[0])]

#thisEff_var = [[Eff_2PMT_var[idx][np.min([np.searchsorted(TDCR_var[idx][::-1],TDCR_var[thekBIdx][i]), (np.shape(Eff_2PMT_var)[1]-1)])] \
#    for i in range(np.shape(TDCR_var)[1])] for idx in range(np.shape(Eff_2PMT_var)[0])]
    #if np.searchsorted(TDCR_var[idx],TDCR_var[thekBIdx][i])<np.shape(Eff_2PMT_var)[1] else np.shape(Eff_2PMT_var)[1]-1] \
    #for i in range(np.shape(TDCR_var)[1])] for idx in range(np.shape(Eff_2PMT_var)[0])]

plt.figure()
labels = []
for i in range(np.shape(thisEff_var)[0]):
    Activity = np.divide(theNcounts[:],thisEff_var[i])
    #Activity = np.divide(theNcounts[::-1],thisEff_var[i])
    #plt.plot(np.divide(thisEff_var[i],thisEff_var[i][-1]), np.divide(Activity,Activity[-1]))
    plt.plot(thisEff_var[i], np.divide(Activity,Activity[np.searchsorted(thisEff_var[i],0.05)]))
    labels.append(r'kB = %.3f cm/MeV' % (kBvals[i]))

plt.legend(labels, ncol=1, loc='lower left', 
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=True)
plt.xlabel('Double Coincidence Efficiency')
plt.ylabel('Activity (AU)')
plt.title(r'Activity vs Efficiency, for a true kB = %.3f cm/MeV' % kBvals[thekBIdx])
print("Done...")
#At some point I also need to introduce the variance.
#The variance is given by the # PE, and should follow from that.

########################
#Old code used for plots
########################
#QuenchedEn_0p01 = [BirksIntegral(En, kB, Zparticle, RestMass) for En in KE]
#QuenchedEn_0p01 = np.transpose(QuenchedEn_0p01)
#kB=0.02
#QuenchedEn_0p02 = [BirksIntegral(En, kB, Zparticle, RestMass) for En in KE]
#QuenchedEn_0p02 = np.transpose(QuenchedEn_0p02)
#kB=0.03
#QuenchedEn_0p03 = [BirksIntegral(En, kB, Zparticle, RestMass) for En in KE]
#QuenchedEn_0p03 = np.transpose(QuenchedEn_0p03)
#kB=0.04
#QuenchedEn_0p04 = [BirksIntegral(En, kB, Zparticle, RestMass) for En in KE]
#QuenchedEn_0p04 = np.transpose(QuenchedEn_0p04)
#kB=0.05
#QuenchedEn_0p05 = [BirksIntegral(En, kB, Zparticle, RestMass) for En in KE]
#QuenchedEn_0p05 = np.transpose(QuenchedEn_0p05)

#plt.figure()
#plt.plot(KE,spec,'b',linewidth=2, label='unquenched spectrum')
#label1 = 'kB = 0.01 cm/MeV'
#label2 = 'kB = 0.02 cm/MeV'
#label3 = 'kB = 0.03 cm/MeV'
#label4 = 'kB = 0.04 cm/MeV'
#label5 = 'kB = 0.05 cm/MeV'
#plt.plot(QuenchedEn_0p01[0],spec,'r',linewidth=2, label=label1)
#plt.plot(QuenchedEn_0p02[0],spec,'g',linewidth=2, label=label2)
#plt.plot(QuenchedEn_0p03[0],spec,'y',linewidth=2, label=label3)
#plt.plot(QuenchedEn_0p04[0],spec,'c',linewidth=2, label=label4)
#plt.plot(QuenchedEn_0p05[0],spec,'m',linewidth=2, label=label5)

#x = np.linspace(0.0001, 100, 1000000)
#y = [BetheBlochFn(theEn, 1, 0.511) for theEn in x]
#plt.figure()
#plt.plot(x,y)


