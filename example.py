# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:50:29 2015

@author: lbignell
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy as sp

#This program will do the following:
#   - Calculate the beta spectrum of C-14, Ni-63, or H-3.
#   - Apply the Birks quenching to the beta spectrum (Bethe-Bloch dE/dx)
#   - Apply a light yield, collection and QE factor to predict # PE.
#   - Monte Carlo sample # PE and coincidence rates (what are 'Lagrange Multipliers')
#   - Calculate delta kB / kB
#Base units: MeV, cm

#Fermi function for beta spectrum shape
#def FermiFunc(KE,Z,A):
#    alpha = 1/137 #fine structure const
#    S = np.sqrt(1 - (alpha**2)*(Z**2))
#    MeC2 = 0.511
#    E = (KE + MeC2)
#    pc = np.sqrt((E**2) - (MeC2**2))
#    eta = alpha*Z*E/(pc)
#    r0 = 1.25*1e13 #in cm    
#    rho = r0*A**(1/3) #Nuclear radius, in cm
#    c = 3*1e10 #cm/s
#    fermifunc = ((2*(1+S))/((sp.special.gamma(1+2*S))**2))* \
#        ((2*(pc/c)*rho)**(2*S - 2))*(np.e**(np.pi*eta))*\
#        ((abs(sp.special.gamma(S + (1j)*eta)))**2)    
#    return fermifunc

#def CalcBetaSpec(KE,Q,Z,A):
#    ff = FermiFunc(KE,Z,A)
#    MeC2 = 0.511
#    spec = np.sqrt(KE**2 + 2*KE*MeC2)*((Q-KE)**2)*(KE + MeC2)*ff
    #c = 3*1e10
    #E = KE + MeC2
    #p = np.sqrt((E/c)**2 - (MeC2/c)**2)
    #spec = ff*p*E*((Q-KE)**2)
#    return spec
    
#units: MeV, dimensionless, MeV/c^2
#def BetheBlochFn(En, Zparticle, RestMass):
    #Implement Bethe-Bloch function here...
#    Me = 0.511 #MeV/c^2
    #Zeff = 7.42 #Value for water
    #N_A = 6.022*1E23 #molecules/mol
    #density = 1 #g/cm^3 for water
    #MolMass = 18.01528 #g/mol
    #n = N_A*Zeff*density/MolMass #electrons/cm^3
#    I = 75*1e-6 #75 +/- 3 eV, from ICRU 49
#    ZonA = 0.55509 #for water, from PDG booklet
#    K = 0.307075 #MeV cm^2 Mol^-1, from PDG booklet
#    if En>0.0001:
#        gamma = En/RestMass + 1
#        betasq = 1 - 1/(gamma**2)
#        Tmax = (2*Me*betasq*gamma*gamma)/(1 + (2*gamma*Me/RestMass) +(Me/RestMass)**2)
        #We should be able to get by without the density correction as we'll be below MIP.
#        dEdx = K*(Zparticle**2)*ZonA*(1/betasq)*(0.5*np.log((2*Me*betasq*(gamma**2)*Tmax)/(I**2)) - betasq)
#    else:
#        gamma = 0.0001/RestMass + 1
#        betasq = 1 - 1/(gamma**2)
#        Tmax = (2*Me*betasq*gamma*gamma)/(1 + (2*gamma*Me/RestMass) +(Me/RestMass)**2)
#        dEdX_100eV = K*(Zparticle**2)*ZonA*(1/betasq)*(0.5*np.log((2*Me*betasq*(gamma**2)*Tmax)/(I**2)) - betasq)
#        dEdx = ((0.0001-En)/0.0001)*dEdX_100eV
#    return dEdx

#def BirksIntegral(Energy, kB, Zparticle, RestMass):
    #implement Birks integral for a given energy of stopping electron.
#    BirksFn = lambda En,kB,Zparticle,RestMass: 1/(1+kB*BetheBlochFn(En, Zparticle, RestMass))
#    QuenchedEn = sp.integrate.quad(BirksFn, 0, Energy, args=(kB, Zparticle, RestMass))
#    return QuenchedEn

#betaspec = CalcBetaSpec(6, 14)
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

#BetaSpec
spec = [CalcBetaSpec(En,Q,Z,A) for En in KE]
spec = spec/sum(spec)

kBvals = np.linspace(0.005,0.07,14)

#AlphaSpec
QuenchedEnAlpha = [[BirksIntegral(5.5, kB, 2, 3727.3)] for kB in kBvals]
QuenchedEnAlpha = [np.transpose(array) for array in QuenchedEnAlpha[:]]
QuenchedEnCE = [[BirksIntegral(0.05954, kB, 1, 0.511)] for kB in kBvals]
QuenchedEnCE = [np.transpose(array) for array in QuenchedEnCE[:]]
QuenchedEnAuger = [[BirksIntegral(0.020, kB, 1, 0.511)] for kB in kBvals]
QuenchedEnAuger = [np.transpose(array) for array in QuenchedEnAuger[:]]

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


