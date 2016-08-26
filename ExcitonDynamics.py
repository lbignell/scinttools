# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:16:22 2015

@author: lbignell

This code will simulate scintillation quenching in a micelle volume.
It will:
- Generate randomly-placed excitations along a particle track of some radius,
  with an average density given by the stopping power and the mean energy
  required to form an exciton energy.
- The lifetimes of each of these excitations will be pre-determined by sampling
  the decay time distribution. There is no difference between non-radiative
  decay and FRET to fluorophore in this context.
- Step through time, allowing the excitons to diffuse randomly in the micelle
  according to the diffusion constant.
- At the end of each step, the excitons may annihilate with another exciton 
  that comes within some quenching radius with some probability.
- Check whether the excitons have exceeded their lifetime after each step,
  and decay those who have.
- Return the number of original excitons, and the number of fluorophores that
  didn't quench.

Units are nm, MeV, nsec

Things to improve:
- Add ability for particle to hit at any angle/location (currently all go
  through the middle).
- Add ability to change temperature. This will change micelle radius and
  the diffusion constant (it varies with T/n, n = viscosity)
"""
import numpy as np
import matplotlib as mpl
import winsound

#Calculate number of excitons to place, and place them in the track.
def InitializeExcitons(stopping_power, micelle_radius,
                       exciton_formation_energy, track_radius):
    MeanNumExcitons = stopping_power*micelle_radius*2/exciton_formation_energy
    #removing the Monte Carlo sampling reduces the variance, but is dodgy.    
    NumExcitons = np.random.poisson(MeanNumExcitons)#round(MeanNumExcitons)
    #print("NumExcitons = ", NumExcitons)
    coords = []
    for i in range(NumExcitons):
        thisx = track_radius
        thisy = track_radius
        while np.sqrt(thisx**2 + thisy**2)>track_radius or \
        np.sqrt(thisx**2 + thisy**2)>micelle_radius:
            thisx = track_radius*(-1 + 2*np.random.uniform())
            thisy = track_radius*(-1 + 2*np.random.uniform())
            
        thisz = 100*micelle_radius
        while np.sqrt(thisx**2 + thisy**2 + thisz**2)>micelle_radius:
            thisz = micelle_radius*(-1 + 2*np.random.uniform())

        coords += [[thisx, thisy, thisz]]#np.append(coords, [thisx, thisy, thisz])
        
    #print("coords = ", coords)
    return coords
        
#Returns the squared distance between two co-ordinates
def GetDistance(first,second):
    return (first[0] - second[0])**2 + \
    (first[1] - second[1])**2 + \
    (first[2] - second[2])**2

#Check whether any of the distance between any two excitons on the list is
#less than quenching radius, and apply quench probability in the cases of
#singlet-singlet or singlet-triplet.
#In the case of triplet-triplet annihilation, quench one and make the other
#singlet with a new lifetime.
def CheckQuenching(coords, time_of_decay, IsTriplet,
                   quenching_radius, TTA_radius, quenching_probability,
                   exciton_lifetime, triplet_lifetime, Time):
    quench_radius_squared = quenching_radius**2
    newcoords = []
    newTOD = []
    newIsTriplet = []
    NumSS = 0
    NumTT = 0
    NumST = 0
    NumStart = len(coords)
    for i in range(len(coords)):
        this_exciton = coords[i]
        this_TOD = time_of_decay[i]
        is_this_triplet = IsTriplet[i]
        quenched = False
        for j in np.linspace(i+1, len(coords)-2, len(coords)-2-i):#  other_exciton in coords[(i+1):]:
#            print("i = ", i, "j = ", j, ", len(coords) = ", len(coords),
#                  "len(IsTriplet) = ", len(IsTriplet))
            thisdistance = GetDistance(this_exciton, coords[int(j)])
            if ((thisdistance<quench_radius_squared)&
            (np.random.uniform()<quenching_probability)):
                #There are 4 cases:
                #1. They're both singlets, so quench the first exciton.
                #2. They're both triplets, so quench the first exciton and make
                #   the second exciton a singlet.
                #3. This exciton is a singlet and the other exciton is a triplet.
                #   The excitation goes to the triplet, which stays in a triplet state.
                #4. This exciton is a triplet and the other is a singlet.
                #   This exciton disappears, and the other excitation becomes this one.
                if not is_this_triplet and not IsTriplet[int(j)]:
                    quenched = True
                    #print("Singlet-Singlet Quench! Time = ", Time, " # Excitons = ", len(coords))
                    NumSS += 1
                    if np.random.uniform()>0.75:                    
                        IsTriplet[int(j)] = False
                        time_of_decay[int(j)] = Time + np.random.exponential(exciton_lifetime)
                    else:
                        IsTriplet[int(j)] = True
                        time_of_decay[int(j)] = Time + np.random.exponential(triplet_lifetime)

                elif is_this_triplet and IsTriplet[int(j)]:
                    TTA_rad_sq = TTA_radius**2
                    if (thisdistance<TTA_rad_sq):
                        #The TTA *must* be less probable than singlet quenching, as otherwise the yield would go UP with dE/dx.
                        #The physical reason for this is that TTA is mediated by Dexter transfer, not FRET.
                        quenched = True
                        NumTT += 1
                        #Sample whether to go into triplet or singlet state.
                        #print("Triplet-Triplet Quench! Time = ", Time, " # Excitons = ", len(coords))
                        if np.random.uniform()>0.75:                    
                            IsTriplet[int(j)] = False
                            time_of_decay[int(j)] = Time + np.random.exponential(exciton_lifetime)
                        else:
                            IsTriplet[int(j)] = True
                            time_of_decay[int(j)] = Time + np.random.exponential(triplet_lifetime)
                                
                elif not is_this_triplet and IsTriplet[int(j)]:
                    quenched = True
                    #print("Singlet-Triplet Quench! Time = ", Time, " # Excitons = ", len(coords))
                    NumST += 1
                    if np.random.uniform()>0.75:                    
                        IsTriplet[int(j)] = False
                        time_of_decay[int(j)] = Time + np.random.exponential(exciton_lifetime)
                    else:
                        IsTriplet[int(j)] = True
                        time_of_decay[int(j)] = Time + np.random.exponential(triplet_lifetime)

                elif is_this_triplet and not IsTriplet[int(j)]:
                    quenched = True
                    #IsTriplet[int(j)] = True
                    #time_of_decay[int(j)] = this_TOD
                    coords[int(j)] = this_exciton
                    #print("Triplet-Singlet Quench! Time = ", Time, " # Excitons = ", len(coords))
                    NumST += 1
                    if np.random.uniform()>0.75:                    
                        IsTriplet[int(j)] = False
                        time_of_decay[int(j)] = Time + np.random.exponential(exciton_lifetime)
                    else:
                        IsTriplet[int(j)] = True
                        time_of_decay[int(j)] = Time + np.random.exponential(triplet_lifetime)

        if not quenched:
            newcoords += [this_exciton]
            newTOD += [this_TOD]
            newIsTriplet += [is_this_triplet]

    NumEnd = len(newcoords)
    ThisQuenchTime = []
    for i in range(NumStart-NumEnd):
        ThisQuenchTime += [Time]
    #print("NumExcitons = ", len(newcoords), "Time = ", Time)            
    return newcoords, newTOD, newIsTriplet, NumSS, NumTT, NumST, ThisQuenchTime

def InitializeLifetime(coords, exciton_lifetime, triplet_lifetime):        
    time_of_decay = []
    is_triplet = []
    for i in range(len(coords)):
        #Monte carlo sample singlet/triplet
        if np.random.uniform()>0.75:
        #if i < (len(coords)*0.25):
            time_of_decay += [np.random.exponential(exciton_lifetime)]
            is_triplet += [False]
        else:
            time_of_decay += [np.random.exponential(triplet_lifetime)]
            is_triplet += [True]
    #print("TODs = ", time_of_decay)    
    return time_of_decay, is_triplet

#Step each exciton by sqrt(Dt)
def Diffuse(coords, diffusion_const, dt, micelle_radius):
    newcoords = []
    for exciton in coords:
        x = 2*micelle_radius
        y = 2*micelle_radius
        z = 2*micelle_radius
        while ((exciton[0]+x)**2 + (exciton[1]+y)**2 + (exciton[2]+z)**2)>\
        micelle_radius**2:
            r = np.random.normal(0,np.sqrt(diffusion_const*dt))
            #print("Step Length (nm) = ", r)
            theta = np.random.uniform()*2*np.pi
            phi = np.random.uniform()*2*np.pi
            x = r*np.cos(theta)*np.sin(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(phi)

        newcoords += [[exciton[0]+x,exciton[1]+y,exciton[2]+z]] #np.append(newcoords, [x,y,z])
        
    return newcoords

def RunEvent(micelle_radius, track_radius, stopping_power, exciton_formation_energy,
             diffusion_const, exciton_lifetime, triplet_lifetime,
             quenching_radius, TTA_radius, quenching_probability, dt,
             FRET_efficiency, fluor_QY):

    #exciton_formation_energy = exciton_energy*exciton_yield
    ExcitonCoords = InitializeExcitons(stopping_power, micelle_radius,
                       exciton_formation_energy, track_radius)
                       #print("Initial Exciton Coords = ", ExcitonCoords)
    init_exciton_coords = ExcitonCoords
    TOD, IsTriplet = InitializeLifetime(ExcitonCoords, exciton_lifetime, triplet_lifetime)        
    #init_time_of_decay = TOD
    QuenchTime = []
    ScintillationTime = []
    InitNumExcitons = len(ExcitonCoords)
    InitNumSinglet = InitNumExcitons - sum(IsTriplet)
    InitNumTriplet = sum(IsTriplet)
    Time = 0
    DecayedSinglets = 0
    NumSS = 0
    NumTT = 0
    NumST = 0
    while len(ExcitonCoords)>1:
        ExcitonCoords, TOD, IsTriplet, thisSS, thisTT, thisST, ThisQuenchTime = \
        CheckQuenching(ExcitonCoords, TOD, IsTriplet,
                       quenching_radius, TTA_radius, quenching_probability,
                       exciton_lifetime, triplet_lifetime, Time)
        ExcitonCoords = Diffuse(ExcitonCoords, diffusion_const, dt, micelle_radius)
        Time = Time + dt
        NumSS += thisSS
        NumTT += thisTT
        NumST += thisST
        QuenchTime += ThisQuenchTime
        newExcitonCoords = []
        newTOD = []
        newIsTriplet = []
        for i in range(len(ExcitonCoords)):
            if TOD[i]>Time:
                #Add to next time step
                newExcitonCoords += [ExcitonCoords[i]]
                newTOD += [TOD[i]]
                newIsTriplet += [IsTriplet[i]]
            else:
                if not IsTriplet[i]:
                    DecayedSinglets += 1
                    ScintillationTime += [Time]
    
        ExcitonCoords = newExcitonCoords
        TOD = newTOD
        IsTriplet = newIsTriplet

    print("Finished Event! Time (ns) = ", Time)
    #print("Number of excitons = ", len(ExcitonCoords))        
    if len(IsTriplet)==1:    
        if not IsTriplet[0]:
            DecayedSinglets += 1
    else:
        print("Warning, len(IsTriplet)!=1 !! IsTriplet = ", IsTriplet)
        
    Edep = InitNumExcitons*exciton_formation_energy
    if InitNumExcitons!=0:
        LY = (DecayedSinglets*FRET_efficiency*fluor_QY)/(InitNumExcitons*exciton_formation_energy)
        print("Initial # Excitons = ", InitNumExcitons, 
              "Initial # Singlets = ", InitNumSinglet,
              "Initial # Triplets = ", InitNumTriplet,
              ", # Unquenched Singlets= ", DecayedSinglets,
              ", ratio = ", DecayedSinglets/InitNumSinglet,
              ", # Singlet-Singlet = ", NumSS, ", # Singlet-Triplet = ", NumST,
              ", # Triplet-Triplet = ", NumTT, "Quenched Fraction = ", (NumSS + NumST)/InitNumSinglet,
              ", Edep (MeV) = ", Edep, "Light Yield (ph/MeV) = ", LY)
        return InitNumExcitons, DecayedSinglets, Edep, LY, QuenchTime, ScintillationTime

    else:
        print("No Energy Deposit!")
        return 0, 0, 0, 0, 0, 0

micelle_radius = [2]#[200, 20, 10, 5, 2, 1]
track_radius = 3 #Best fit for inorganic crystals: Williams et al., Hard X-ray, Gamma Ray, and Neutron Detector Physics XIII, SPIE 8142 (2011)
stopping_power = 40*1e-7#, 20*1e-7, 40*1e-7, 100*1e-7, 200*1e-7, 500*1e-7]
diffusion_const = 4e-1 #nm^2/nsec = 4e-6 cm^2/sec. A typical value (needs reference).
exciton_lifetime = 7.46 #LAB + 1g/L PPO, Marrodán Undagoitia et al, Rev Sci Instr, 80, 043301 (2009)
triplet_lifetime = 150 #an educated guess, based on http://arxiv.org/pdf/1102.0797.pdf.
TTA_radius = 0.5 #from Nicholas P. Cheremisinoff (Ed.), Handbook of Polymer Science and Technology, Volume 4, Chapter 1: Influence of Molecular Structure on Polymer Photophysical and Photochemical Properties, by H. Ushiki and K. Horie, pg 19 (1989)
#Above param is shorter as TTA is mediated by Dexter transfer, not FRET (see ref).
quenching_radius = 2 #This parameter is, perhaps, the most difficult to pin down. I could set it to the rotationally-averaged LAB diameter?
#Actually, I can tune it in a bulk case (1 um micelle)
quenching_probability = 1
FRET_efficiency = 0.7 #I'm sure I've seen this measured before, but I can't remember where.
fluor_QY = 0.9 #The scint re-emission prob should be something like the FRET_efficiency*fluor_QY.
unquenchedLY = 1e4 #For pure LS.
exciton_energy = 3.5*1e-6 #Corresponds to 354 nm singlet excitation. It isn't too imporant as exciton probability is tuned anyway.
exciton_formation_energy = (0.25*fluor_QY*FRET_efficiency)/(unquenchedLY) #factor to account for all other possible excitations.
#I have tuned exciton probability by considering that I should get 10000 photons/MeV in pure scintillator from quench probability = 0.
#timesteps = [10, 1, 0.1, 0.01, 0.001]
dt = 0.01
InitExcitons = []
EmittingExcitons = []
Ratio = []
MeanRatio = []
UMeanRatio = []
Edep = []
LY = []
QuenchTime = []
ScintillationTime = []
LYvsParam = []
EdepvsParam = []
InitExcitonsvsParam = []
EmitExcitonsvsParam = []
QuenchTimevsParam = []
ScintTimevsParam = []
for theRadius in micelle_radius:
    print("theRadius = ", theRadius)
    for i in range(2000):
        thisInit, thisEmit, thisEdep, thisLY, thisQuenchTime, thisScintTime = \
        RunEvent(theRadius, track_radius, stopping_power, exciton_formation_energy,
                 diffusion_const, exciton_lifetime, triplet_lifetime,
                 quenching_radius, TTA_radius, quenching_probability, dt,
                 FRET_efficiency, fluor_QY)
        print("i = ", i)
        InitExcitons += [thisInit]
        EmittingExcitons += [thisEmit]
        Edep += [thisEdep]
        LY += [thisLY]
        if thisInit != 0:
            QuenchTime += thisQuenchTime
            ScintillationTime += thisScintTime
            Ratio += [thisEmit/thisInit]

    print("Time step (ns) = ", dt)
    print("Mean Init = ", np.mean(InitExcitons), " +/- ", np.std(InitExcitons)/np.sqrt(len(InitExcitons)),
          ", Mean Unquenched Singlets = ", np.mean(EmittingExcitons), "+/-",
          np.std(EmittingExcitons)/np.sqrt(len(EmittingExcitons)), ", Mean Ratio = ",
          np.mean(Ratio), "+/-", np.std(Ratio)/np.sqrt(len(Ratio)), 
          "Edep (MeV) = ", np.mean(Edep), "+/-", np.std(Edep)/np.sqrt(len(Edep)), 
          "LY = ", np.mean(LY), "+/-", np.std(LY)/np.sqrt(len(LY)),
          "Quenching Factor = ", np.mean(LY)/unquenchedLY)

    MeanRatio += [np.mean(Ratio)]
    UMeanRatio += [np.std(Ratio)/np.sqrt(len(Ratio))]
    LYvsParam += [LY]
    EdepvsParam += [Edep]
    InitExcitonsvsParam += [InitExcitons]
    EmitExcitonsvsParam += [EmittingExcitons]
    QuenchTimevsParam += [QuenchTime]
    ScintTimevsParam += [ScintillationTime]

#mpl.pyplot.figure()
#mpl.pyplot.hist(ScintillationTime, 1000, (0, 200), histtype='step', color='b', label='Scintillation Decay Time')
#mpl.pyplot.hist(QuenchTime, 1000, (0, 200), histtype='step', color='r', label='Quench Time')
#mpl.pyplot.xlabel('time (ns)')
#mpl.pyplot.legend()
winsound.Beep(2000,3000)
#mpl.pyplot.errorbar(timesteps, MeanRatio, yerr = UMeanRatio, fmt='b', linewidth=2)