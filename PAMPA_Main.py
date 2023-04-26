#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.linalg as la
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
import math
import random


def main(name,MW,V,logKoa,logKow,ionization,pKa_a,pKa_b,pH,t,deltaW): # input: chemical identitfier, molecular weight, molecular volume, logKoa, logKow, ionization (0 = neutral, 1 = acid or base, 2 = zwitterion  ), pKa_a,pKa_b,pH,incubation time, UWL thickness

    
    if ionization == 0:
        f = 1
    elif ionization == 1 and pKa_a == 0:
        f = 1/(1+10**(pKa_b-pH)) 
    elif ionization  == 1 and pKa_b == 0:
        f = 1/(1+10**(pH-pKa_a))
    elif ionization == 2:
        f = 1/(1+10**abs(0.5*pKa_a+0.5*pKa_b-pH))
    else:
        print("error")
         
    K = f * (10**logKow)
    K = (0.05*K)**0.6 # adjust from Kow to KmembraneW

    if pKa_a != 0:
        negative_charge = 1
    else: negative_charge = 0

    if pKa_b != 0:
        positive_charge = 1
    else: positive_charge = 0

    ita = 7.59
    T = 298
    Dw_org = (10 ** (- 4.13 - 0.453 * np.log10(MW))) # order of -5
    Dw = Dw_org 

    Dm_org = (13.3e-8 * T **1.47 * ita**(((10.2/(V*100))-0.791))) / ((100*V)**0.71) # order of -4
    Dm = Dm_org #* (1/1.5)  #* (f + alpha*(1-f)*positive_charge + beta*(1-f)*negative_charge)

    n = 1
    Ld = 0.2/0.3 # in cm, VR = VD = 200uL, Area = 0.3 cm2
    La = 0.2/0.3
    Lm = 0.0125
    Vd = 0.2
    Va = 0.2
    Lair = 0.002


    deltaM = ((Lm*0.99)/2)

    rv = 1 
    Area = 0.3

    W = Dw * deltaM
    M = Dm * deltaW

    C0 = 10000
    deltaP = 0.01
    deltaA = 0.0002

    Kpw = 0.06*(10**logKow)
    Kaw = 10**(logKow-logKoa)
    Daw = f * Kaw

    Dp = 10**((-2391-3486)/T + 6.39 -2.49*np.log10(MW) + 4.79)
    Da = 0.001*T**1.75 * ((MW+28.97)/(28.97*MW))**0.5 / ((V*100 + 20.1)**2)

    flux_dp_cd = 1.6/0.2 *(Dw/deltaW)*((Dw/deltaW)/(Dw/deltaW + (Kpw*Dp/deltaP)) -1)
    flux_dp_cp = 1.6/0.2 *(Dw/deltaW)*((Dp/deltaP)/(Dw/deltaW + (Kpw*Dp/deltaP)))
    flux_ap_ca = 1.3/0.2 * (Dw/deltaW)*((Dw/deltaW)/(Dw/deltaW + (Kpw*Dp/deltaP)) -1)
    flux_ap_cp = 1.3/0.2 *(Dw/deltaW)*((Dp/deltaP)/(Dw/deltaW + (Kpw*Dp/deltaP)))
    flux_aair_ca = 0.003/0.2 * (Dw/deltaW)*((Dw/deltaW)/(Dw/deltaW + (Daw*Da/deltaA))-1)
    flux_aair_cair = 0.003/0.2 * (Dw/deltaW)*((Da/deltaA)/(Dw/deltaW + (Daw*Da/deltaA)))

    B = np.array([[(Dw/(Ld*deltaW))*((W/(W+M*K))-1) + flux_dp_cd,
                   (Dw/(Ld*deltaW))*(M/(W+K*M)),
                   0,
                   0,
                   flux_dp_cp],

                  [(Dw/(Lm*deltaW))*(1-(W/(W+M*K))),
                   (-Dw/(Lm*deltaW))*(2*M/(W+K*M)),
                   (Dw/(Lm*deltaW))*(1-(W/(W+K*M))),
                  0,
                  0],

                  [0,
                   (Dw/(La*deltaW))*(M/(W+K*M)),
                   (Dw/(La*deltaW))*((W/(W+K*M))-1) + flux_ap_ca +flux_aair_ca,
                  flux_aair_cair,
                  flux_ap_cp],

                  [0,
                  0,
                  -flux_aair_ca,
                  -flux_aair_cair,
                  0],

                  [-flux_dp_cd,
                  0,
                  -flux_ap_ca,
                  0,
                  -flux_ap_cp - flux_dp_cp]])


    w,v = np.linalg.eig(B) # general solution
    eig_values = w
    eig_vectors = v.transpose()
    ivp = eig_vectors.transpose()
    a = np.array([C0,0,0,0,0]) # initial conditions
    c = np.linalg.solve(ivp,a) # solving constants for initial conditions

    Cd = c[0] * eig_vectors[0,0] * np.exp(eig_values[0] * t) + c[1] * eig_vectors[1,0] * np.exp(eig_values[1]*t) + c[2] * eig_vectors[2,0] * np.exp(eig_values[2]*t) + c[3] * eig_vectors[3,0] * np.exp(eig_values[3]*t) + c[4] * eig_vectors[4,0] * np.exp(eig_values[4]*t)
    Cm = c[0] * eig_vectors[0,1] * np.exp(eig_values[0] * t) + c[1] * eig_vectors[1,1] * np.exp(eig_values[1]*t) + c[2] * eig_vectors[2,1] * np.exp(eig_values[2]*t) + c[3] * eig_vectors[3,1] * np.exp(eig_values[3]*t) + c[4] * eig_vectors[4,1] * np.exp(eig_values[4]*t)
    Ca = c[0] * eig_vectors[0,2] * np.exp(eig_values[0] * t) + c[1] * eig_vectors[1,2] * np.exp(eig_values[1]*t) + c[2] * eig_vectors[2,2] * np.exp(eig_values[2]*t) + c[3] * eig_vectors[3,2] * np.exp(eig_values[3]*t) + c[4] * eig_vectors[4,2] * np.exp(eig_values[4]*t)
    Cair = c[0] * eig_vectors[0,3] * np.exp(eig_values[0] * t) + c[1] * eig_vectors[1,3] * np.exp(eig_values[1]*t) + c[2] * eig_vectors[2,3] * np.exp(eig_values[2]*t) + c[3] * eig_vectors[3,3] * np.exp(eig_values[3]*t) + c[4] * eig_vectors[4,3] * np.exp(eig_values[4]*t)  
    Cp = c[0] * eig_vectors[0,4] * np.exp(eig_values[0] * t) + c[1] * eig_vectors[1,4] * np.exp(eig_values[1]*t) + c[2] * eig_vectors[2,4] * np.exp(eig_values[2]*t) + c[3] * eig_vectors[3,4] * np.exp(eig_values[3]*t) + c[4] * eig_vectors[4,4] * np.exp(eig_values[4]*t)

    S = (Va/Vd) * (Ca/C0) + (Cd/C0)

    tao = 1140


    Pe = (-2.303 * Va*Vd / ((Va+Vd) * (Area) * (t-tao))) * (np.log10(1-(((Va+Vd)/(Vd*S))*(Ca/C0)))) # effective permeability
    Re = (100 * (Cm*Lm)/(C0*Ld)) # membrane retention
    
    
    return Pe

