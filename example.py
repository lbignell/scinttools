# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:50:29 2015

@author: lbignell
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import scipy as sp
import scinttools.physics.isotope
import scinttools.physics.material
import scinttools.analysis.detector
import seaborn as sns

C14spec = scinttools.physics.isotope.BetaDecay('C-14 Decay to Ground', 0.156476, 6, 14, 1)
WbLS = scinttools.physics.material.Scintillator('1% WbLS', 100, 0.044)
analyse = scinttools.analysis.detector.TDCR(0.7*0.33, 0.2, scint=WbLS, branch=C14spec)
factorvals = np.linspace(0.5, 1.5, 101)
kBvals = np.linspace(0.005, 0.07, 14)
sns.set_palette("husl", len(kBvals))
sns.set_style(style='darkgrid')
sns.set_context('poster')
#Set some nicer formatting defaults.
mpl.rcParams['axes.formatter.useoffset']=False
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['axes.titlesize'] = 30

measuredactivity, TDCR_meas, TDCR_true, effn_true = analyse.eff_extrap_beta(factorvals,kBvals,verbose=1)

plt.figure()
normconst=1
#for normalisation, I'll fit to a straight line and use the value at max(TDCR_true)
linearfit = True
gradient = []
for i in range(len(kBvals)):
    if linearfit:
        xvals = [TDCR_meas[i][j] for j in range(len(TDCR_meas[i]))
            if measuredactivity[i][j] > 0]
        yvals = [y for y in measuredactivity[i] if y > 0]
        fit = np.polyfit(xvals, yvals, 1)
        gradient += [fit[0]]
        line = np.poly1d(fit)
        normconst = line(max(TDCR_true))
        print('kB = {:.3f}, normconst = {:.3f}'.format(kBvals[i], normconst))
    else:
        #use the crappier normalisation method I was using previously.
        diff = min(abs(np.subtract(measuredactivity[i],1)))
        if max(measuredactivity[i])<1:
            normconst = 1 - diff
        else:
            #was elif min(measuredactivity[i])>1:
            normconst = 1 + diff

    plt.plot(xvals, np.divide(yvals,normconst),
         label='kB = {:0.3f}'.format(kBvals[i]))
    #plt.plot(TDCR_meas[i], line(TDCR_meas[i]))

plt.legend(ncol=1, loc='upper right', 
           columnspacing=1.0, labelspacing=0.0,
           handlelength=1.5, fancybox=True, fontsize=16)
plt.xlabel('TDCR')
plt.ylabel('Apparent Activity (AU)')
plt.title(r'Efficiency extrapolation, for a true kB = {:.3f} cm/MeV, LY = {:d} ph/MeV'.format(WbLS.kB, WbLS.LY))

#make a plot of the gradient vs kB.
plt.figure()
plt.plot(kBvals, gradient, 'k')
plt.xlabel('kB (cm/MeV)')
plt.ylabel('Efficiency extrapolation gradient (activity/TDCR)')
plt.title(r'True kB = {:.3f} cm/MeV, LY = {:d} ph/MeV'.format(WbLS.kB, WbLS.LY))
