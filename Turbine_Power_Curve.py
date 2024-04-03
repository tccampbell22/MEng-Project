#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:17:02 2024

@author: thomascampbell
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

import pandas as pd


def get_turbine_power_curve(turb_type, def_cut_in=1, def_cut_out= 4.5, def_diameter=20, density=1025,plotting='on', cut_in_ramp='on',cut_out='ramp'):
    """
    

    Parameters
    ----------
    turb_type : choose the name of the turbine
    def_cut_in : cut_in value if none found for turb_type. The default is 1.
    def_cut_out : cut_out value if none found for turb_type. The default is 4.5.
    def_diameter : diameter if none found for turb_type. The default is 20.
    density : Of seawater. The default is 1025.
    plotting : Choose whether to plot a power curve. The default is 'on'.
    cut_in_ramp : Choose whether to have a curve between 0 and cut in power. 
                  Can be off (sharp increase) or on (curve)
    cut_out : Choose whether to curve fit after cut-out. Can be 'off' (no cut-out),
              'on': (cut-out but no curve fit) or 'ramp': (curve fit after cut-out)

    Returns
    -------
    Tinfo : a dictionary containing the power curve as well as turbine information

    """
    Tdata=set_parameters(turb_type,def_cut_in,def_cut_out,def_diameter) #get turbine characteristics from excel file
    
    Tdata['Rho']=density
    Tdata['A_t']=np.pi*((0.5*Tdata['D'])**2) 
    
    
    #Set up a velocities array
    umag_in = np.arange(0., Tdata['u_in'] + 0.05, 0.05)
    umag = np.append(umag_in, Tdata['u_in'] + 0.001)
    umag = np.append(umag, np.arange(Tdata['u_in'] + 0.05, Tdata['u_out'] + 0.05, 0.05))
    umag = np.append(umag, Tdata['u_out'] + 0.001)
    umag = np.append(umag, np.arange(Tdata['u_out'] + 0.05, Tdata['u_out'] + 1.05, 0.05))
    powers = []  # set up empty power list




    for u in umag:
        powers.append(P(u, Tdata, cut_in_ramp, cut_out)/1E6) #P in MW to match Pnom
    #calculate thrust coefficient based on Martin Short 2015, for the turbine params given
           
    
    if plotting == 'on':
        plot(umag,powers,Tdata, cut_in_ramp, cut_out)
    #plot turbine coefficient and power curve
        
    Tinfo = {"speeds": umag, "powers": powers,"P": Tdata['P'], "D": Tdata['D']}

    
    return Tinfo






def set_parameters(turb_type,def_cut_in,def_cut_out,def_diameter):
    
    defaults=[def_cut_out,def_cut_in,def_diameter]
    
    # Turbine parameters & options: user inputs turb name and these are collated from csv
    turbine_data=pd.read_excel('../Data/Turbine data.xlsx',sheet_name='Turbine_Data',index_col=0)
    
    Tdata=turbine_data.loc[turb_type]
    Tcols=turbine_data.columns
    
    
    for i in range(2,5):
        if Tdata[Tcols[i]]=='-':
            Tdata[Tcols[i]]=defaults[i-2]   
    #if all parameters are saved for turbine then code will take them from the file, 
    #if not it will use the user-input defaults
            
    print(Tdata)
    
    return Tdata







def ramp_up(x, a, b, c, d):
    #curve trend for ramp in
        return a * np.exp(-b * x) + c * x**d 

def ramp_down(x, a, b, c):
    #curve trend for ramp out
        return a * (x-b)**c

    

def P(u, Tdata,cut_in_ramp,cut_out):
    """
    

    Parameters
    ----------
    u : velocity
    Tdata : dictionary of turbine parameters
    cut_in_ramp : whether to curve fit at cut in
    cut_out : whether to curve fit at cut-out

    Returns
    -------
    the corresponding power for the input velocity u

    """
    
    c_p = 2*Tdata['P']/(Tdata['Rho']*Tdata['A_t']*(Tdata['u_rated']**3))
    c_start = 0.1

    
    #calculates the constant C_t from martin short (2015)
    f = lambda c: c**3 - 4*c_p*c + 4*(c_p**2)
    f_prime = lambda c: 3*c**2 - 4*c_p
    c_t = so.newton(f, c_start, fprime=f_prime)
    
    #curve fitting a ramp at cut-in
    if cut_in_ramp == 'on':
        popt_in= ramp_in(Tdata)

    #curve fitting a ramp at cut-out
    if cut_out == 'ramp':
        popt_out= ramp_out(Tdata)
    
    #if less than u_ramp_in or more than u_ramp_out P is 0
    #if in the ramp sections c_t is curve fitted and from this P calculated
    #if between cut-in and rated ct is a constant, Ct
    #if between u rated and cut-out P=Prated
    
    if u <= Tdata['u_in']:
        if cut_in_ramp == 'on':
            ct = ramp_up(u, *popt_in)
            cp=1/2 * (1 + np.sqrt(1-ct)) * ct
            return 1/2 * Tdata['Rho'] * cp * Tdata['A_t'] * u**3
        else:
            return 0. #ramps to P at u_in, otherwise 0
        
    elif Tdata['u_in'] < u <= Tdata['u_rated']:
        cp=1/2 * (1 + np.sqrt(1-c_t)) * c_t
        return 1/2 * Tdata['Rho'] * cp * Tdata['A_t'] * u**3 
    #returns P as calculated from power equation, with CP for C_t
    
    elif Tdata['u_rated'] < u <= Tdata['u_out']:
        return Tdata['P'] 
    #just the rated power
    
    else:
        if cut_out == 'ramp':
            ct=ramp_down(u, *popt_out)
            cp=1/2 * (1 + np.sqrt(1-ct)) * ct
            return 1/2 * Tdata['Rho'] * cp * Tdata['A_t'] * u**3
        elif cut_out == 'off':
            return Tdata['P']
        else:
            return 0.
        #ramps from rate power to 0, otherwise just 0



def ramp_out(Tdata): 
    #function for ramping, returns the ramped ct array between ct(u_out) and ct=0
    
    c_start = 0.1
    
    #works out ct at u_out, per martin short (2015)
    c_p = (2 * Tdata['P']) / (Tdata['Rho'] * Tdata['A_t'] * (Tdata['u_out'] ** 3))
    c_start = 0.1
        
    f = lambda c: c ** 3 - 4 * c_p * c + 4 * (c_p ** 2)
    f_prime = lambda c: 3 * c ** 2 - 4 * c_p
    
    c_t_out = so.newton(f, c_start, fprime=f_prime)
    
    u_2 = [Tdata['u_out'], Tdata['u_out']+0.05, Tdata['u_out']+0.5, 1.5*Tdata['u_out']]
    c_2 = [c_t_out, 0.05*c_t_out, 0.01*c_t_out, 0]
    
    popt_out, pcov_out = so.curve_fit(ramp_down, u_2, c_2)
    return popt_out

def ramp_in(Tdata):
        #function for ramping in works out ct at u-in, then curve fits between this and zero
    
        #works out ct at u_rated, per martin short (2015), this is constant between u_in and u_rated
        c_p = (2 * Tdata['P']) / (Tdata['Rho'] * Tdata['A_t'] * (Tdata['u_rated'] ** 3))
        c_start = 0.1
        
        f = lambda c: c ** 3 - 4 * c_p * c + 4 * (c_p ** 2)
        f_prime = lambda c: 3 * c ** 2 - 4 * c_p
        
        
        c_t_in = so.newton(f, c_start, fprime=f_prime)
        
        
        u_1 = [0, 0.3*Tdata['u_in'], 0.5*Tdata['u_in'], 0.7*Tdata['u_in'], 0.9*Tdata['u_in'], Tdata['u_in']]
        c_1 = [0, 0.05*c_t_in ,0.05*c_t_in, 0.05*c_t_in, 0.1*c_t_in, c_t_in]
        
        popt_in, pcov_in = so.curve_fit(ramp_up, u_1, c_1)
        
        return popt_in


def plot(speeds,powers,Tdata, cut_in_ramp, cut_out):
     #plots a power curve
     
    plt.plot(speeds, np.array(powers)/1E6, linestyle='solid', color=(0.45, 0.45, 0.45))
    plt.xlim(0,5.5)
    plt.xlabel('Velocity, u (m/s)', labelpad=3.5, fontweight='ultralight')
    plt.ylabel('Power, $P$ (MW)', labelpad=2, fontweight='ultralight')
    plt.show()
    

    
