# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:44:12 2021 
Centralized HoV dispatch code based on INTEGER PROGRAMMING.
Generates set-points for the diesel generators and Battery inverters and load ON-OFF suggestions. 
PV inveters are running at MPPT.

Implements a charging region for the battery. One the battery hits the charging threshold it should be charged until 
reaching normal range SOC.

"""
__author__ = " Vivek Khatana, UMN"

import pandas as pd
import logging
import time
import numpy as np
import cvxpy as cvx
np.set_printoptions(suppress=True,edgeitems=10)
from threading import Thread
import sys
import argparse
import socket
import struct
import os


### LOG FILE LOCATIONS ###
HOV_LOG_FILENAME = "hovLog.csv" #log file where

#setup logging
fmt = "%(asctime)s.%(msecs)03d - %(message)s"
datefmt='%y%m%d %H:%M:%S'
logLevel = logging.DEBUG

if logging.getLogger().hasHandlers():
    log = logging.getLogger()
    log.setLevel(logLevel)
    log.handlers[0].setFormatter(logging.Formatter(fmt))
else:
    logging.basicConfig(level=logLevel,format=fmt,datefmt=datefmt)
    log = logging.getLogger()
    
    
    
 
def HoVengine(Vec,Horizon):
    """ 
    Cost-optimal dispatch of all the controllable generation sources within single net-load group over a horizon.
    
    -- Notes --
        - All power capacity values are in Watts
        - The load convention is used so all loads are positive power and generators supplying power are negative power
        - For loads Pmin = 0 always and Pmax is the expected load value when turning the load back on (if turn-on power is unknown, the
          the max power in the profile is generally used to be conservative)
        - For solar-based generation Pmin < 0 and will be time-varying based on solar irradiance
        - Exactly one slack bus/generator must be defined in the scenario configuration and should incorporate all generation not represented
          as other net-load devices

    -- Parameters --
    Vec : numpy array
        Measurement vector 
        
    -- Returns --
    X, PVsetpt, Pgsetpt, Pbsetpt
    X : cvxpy array
        Binary-encoded (1=on, 0=off) Defereable load status over the horizon
    Pbsetpt :  
        Active power set points for the battery inverters over the horizon 
    Pgsetpt :  
        Active power set points for the diesel generators over the horizon
        
    """

    global Ng 
    global Nb 
    global Npv
    global Nc
    global Nd
    global IsChargingReqd

    HoV_ON = 1 # flag to ensure required generation for the HoV dispatch;
               # HoV_ON = 1: HoV engine active, 0: HoV engine not used
    
    # initialization 
    
    # time discretization to get the energy remaining
    time_correction_hrs_to_sec = 3600    
    SOCmin = 0; SOEmin = 0
    discharg_thres = 0.75; charg_thres = 0.30; 
    
    # flag to check if the optimal solution is acheieved
    converged = 0
    
    # Horizon receeding horizon controller
    T = Horizon  
    
    w_kw = 1e-3 # Watt to kW change in the measurement quantities
    
    #initial energy rating of the generation sources  
    # Battery energy
    nontype4 = Nd+Nc+Npv
    Eb_initial = w_kw*ConfigData[nontype4:nontype4 + Nb,7]  
    
    # Diesel Genset energy
    Eg_initial = w_kw*ConfigData[nontype4 + Nb:nontype4 + Nb + Ng,7] 
    
    # Deferrable loads cost vector    
    CL = ConfigData[0:Nd,3]
    Plmax = w_kw*ConfigData[0:Nd,6]
    Plmaxinv = np.reciprocal(Plmax)
    Plmaxinv = np.reciprocal(Plmax).reshape(Nd,1)
    Plmaxinv = np.multiply(Plmaxinv, np.ones((1,Horizon)))
    
    TotalPowerMeasurements = Nd + Nc + Npv*2 + Nb*3 + Ng*3 + 3 + 1 # last 3 is for the Slack bus and 1 is for Grid measurement
    
    # Current value of Battery SOCs
    SOC = np.ones(Nb)
    SOC = Vec[TotalPowerMeasurements + 1:TotalPowerMeasurements + 1 + Nb] # +1 as Vec[0] is made to be the time stamp 
    
    for i in range(Nb):
        if (SOC[i] < charg_thres):
            IsChargingReqd[i] = 1
        elif (SOC[i] > charg_thres) and (SOC[i] < discharg_thres) and (IsChargingReqd[i] == 0):
            IsChargingReqd[i] = 0
        elif (SOC[i] > charg_thres) and (SOC[i] < discharg_thres) and (IsChargingReqd[i] == 1):
            IsChargingReqd[i] = 1 
        elif (SOC[i] > discharg_thres):  
            IsChargingReqd[i] = 0
                
    
    # Current value of Diesel Genset SOEs
    SOE = np.ones(Ng)
    SOE = Vec[TotalPowerMeasurements + 1 + Nb: TotalPowerMeasurements + 1 + Nb + Ng] # +1 as Vec[0] is made to be the time stamp 
    
     
    # Energy remaining in the generation sources     
    # Remaining energy in Battery inverters
    Ebd = np.multiply(SOC,Eb_initial)*time_correction_hrs_to_sec
     
    SOCPrime = 1 - SOC
    Ebc = np.multiply(SOCPrime,Eb_initial)*time_correction_hrs_to_sec

    # Remaining energy in Diesel Gensets
    Eg = np.multiply(SOE,Eg_initial)*time_correction_hrs_to_sec
    # Eg0 = SOE0*Eg_initial*dt;  
    

    # Initial time at the start of MPC (time when the measurement of the power system was taken)
    InitTime = np.ceil(Vec[0]); InitTime = np.intc(InitTime);
    
       
    # Total critical load power demand
    Pc = w_kw*loadData[InitTime:InitTime + T, Nd + Nc]
        
    # Controllable/deferable loads instantaneous power demand
    PLi = w_kw*loadData[InitTime:InitTime + T, 1:Nd + Nc].T
    
    # available PV power
    PV = np.zeros(Npv)
    for i in range(Npv):
        PV[i] = -Vec[Nd + Nc + 1 + 2*i]*w_kw
    
    # Slope of the PV power profile to caluclate the predicted PV power values in the future  
    slope = np.ones((T-1,Npv))    
    for i in range(Npv):
        slope[:,i] = np.cumprod(PVslope[InitTime:InitTime+T-1,i])
    # slope0 =  np.cumprod(PVslope[InitTime:InitTime+T-1,0]); slope1 =  np.cumprod(PVslope[InitTime:InitTime+T-1,1]);
    # slope2 =  np.cumprod(PVslope[InitTime:InitTime+T-1,2]); slope3 =  np.cumprod(PVslope[InitTime:InitTime+T-1,3]);
    # slope4 =  np.cumprod(PVslope[InitTime:InitTime+T-1,4]); slope5 =  np.cumprod(PVslope[InitTime:InitTime+T-1,5]);
    # slope6 =  np.cumprod(PVslope[InitTime:InitTime+T-1,6]);  
    
    # Predicted PV power output 
    PredPVOutput = np.zeros((T,Npv))
    for i in range(Npv):
        PredPVOutput[:,i] = np.concatenate((PV[i],PV[i]*slope[:,i]), axis=None)
    # PV0 = np.concatenate((-Vec[140],-Vec[140]*slope0), axis=None); PV1 = np.concatenate((-Vec[142],-Vec[142]*slope1), axis=None);
    # PV2 = np.concatenate((-Vec[144],-Vec[144]*slope2), axis=None); PV3 = np.concatenate((-Vec[146],-Vec[146]*slope3), axis=None);
    # PV4 = np.concatenate((-Vec[148],-Vec[148]*slope4), axis=None); PV5 = np.concatenate((-Vec[150],-Vec[150]*slope5), axis=None);
    # PV6 = np.concatenate((-Vec[152],-Vec[152]*slope6), axis=None); 
    
    
    
    # Upper bound in the available power from the generation sources   
    # Battery power max values
    Pbmax = np.zeros(Nb)
    nontype4PowerMeas = Nd + Nc + Npv*2
    for i in range(Nb):
        Pbmax[i] = -Vec[nontype4PowerMeas + 1 + (3*i+1)]*w_kw
    
    # Pbmax0  = -Vec[155]; Pbmax1 = -Vec[158]; Pbmax2  = -Vec[161]; Pbmax3 = -Vec[164]; Pbmax4  = -Vec[167]; Pbmax5 = -Vec[170]; 
    # Pbmax6  = -Vec[173]; 
    
    # Diesel Gensets power max values
    Pgmax = np.zeros(Ng)
    for i in range(Ng):
        Pgmax[i] = -Vec[nontype4PowerMeas + Nb*3 + 1 + (3*i+1)]*w_kw
   
    # log.info("Pbmax" + str(Pbmax))
    # log.info("Pgmax" + str(Pgmax))
   
    
    # Pgmax0 = -Vec[176]; 
    
    # Slack bus power max and min limits
    Pslkmax = -Vec[nontype4PowerMeas + Nb*3 + Ng*3 + 1 + 1]*w_kw
    Pslkmin = -Vec[nontype4PowerMeas + Nb*3 + Ng*3 + 1 + 2]*w_kw
        
    curr_CL_ON = np.zeros(Nd)
    for i in range(Nd):
        if np.abs(Vec[i+1]) > 0:
            curr_CL_ON[i] = 1
        else:
            curr_CL_ON[i] = 0

    prevSoln = np.multiply(curr_CL_ON.reshape(Nd,1),np.ones(T))

    if np.max(Pgmax) == 0 and np.max(Pbmax) == 0:
        HoV_ON = 0
        return np.zeros(Nd), np.zeros(Nb), np.zeros(Ng), 1, HoV_ON, 0 
        # LoadStatus, Pbsetpt, Pgsetpt, Pslksetpt, HoV_ON, opt_sol_found 

    if HoV_ON == 1:   
        if np.max(SOC) > SOCmin and np.max(SOE) > SOEmin:
            # Optimization problem formulation
 
            # Deciding the number of variables based on the active/available generation
            Activebatt= np.argwhere(SOC > SOCmin).ravel()
            Nbactive  = Activebatt.size
            
            Activediesel= np.argwhere(SOE > SOEmin).ravel()
            Ngactive  = Activediesel.size
            
            Pg = cvx.Variable((Ngactive,T))
            Pb = cvx.Variable((Nbactive,T))
            Pslk = cvx.Variable(T)
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function of the Diesel Gensets
            # The cost assumes the same Pgmax and efficiency curve for the diesel generators
            Cg = cvx.minimum( (18.8/20)*(100*Pg/Pgmax[Activebatt[0]] - 20) + 20,\
                                    (15.1/80)*(100*Pg/Pgmax[Activebatt[0]] - 100) + 33.9 )
            # Cg1 = 10*cvx.minimum( (18.8/20)*(100*Pg[1,:]/Pgmax1 - 20) + 20,\
            #                         (15.1/80)*(100*Pg[1,:]/Pgmax1 - 100) + 33.9 )   
                
                      
            
            # Down cost of the controllable loads
            dummy = np.ones((Nd,T))-X
            dummy0 = cvx.multiply(dummy,PLi) 
            dummy1 = cvx.multiply(dummy0,Plmaxinv)
            Cload = CL@dummy1
                               
            # Cost of battery dispacth
            # cost of discharging the battery based on SOC
            Cd =  np.zeros(Nbactive)
            Cc =  np.zeros(Nbactive)   

            for i in range(Nbactive):
                Cd[i] = np.max([25*SOC[Activebatt[i]], 35*SOC[Activebatt[i]] - 2, 65*SOC[Activebatt[i]] - 14, \
                         125*SOC[Activebatt[i]] - 50, 1000*SOC[Activebatt[i]] - 750])
                Cc[i] = np.max([-1000*SOC[Activebatt[i]] + 250, -125*SOC[Activebatt[i]] + 75, -65*SOC[Activebatt[i]] + 51, \
                         -35*SOC[Activebatt[i]] + 33, -25*SOC[Activebatt[i]] + 25]) 
            
            # Cd = np.max([25*SOC[Activebatt[:]], 35*SOC[Activebatt[:]] - 2, 65*SOC[Activebatt[:]] - 14, \
                         # 125*SOC[Activebatt[:]] - 50, 1000*SOC[Activebatt[:]] - 750])
            # Cd0 = np.max([25*SOC0, 35*SOC0 - 2, 65*SOC0 - 14, 125*SOC0 - 50, 1000*SOC0 - 750]) 
            # Cd1 = np.max([25*SOC1, 35*SOC1 - 2, 65*SOC1 - 14, 125*SOC1 - 50, 1000*SOC1 - 750])
            # Cd2 = np.max([25*SOC2, 35*SOC2 - 2, 65*SOC2 - 14, 125*SOC2 - 50, 1000*SOC2 - 750])
            # Cd3 = np.max([25*SOC3, 35*SOC3 - 2, 65*SOC3 - 14, 125*SOC3 - 50, 1000*SOC3 - 750])
            # Cd4 = np.max([25*SOC4, 35*SOC4 - 2, 65*SOC4 - 14, 125*SOC4 - 50, 1000*SOC4 - 750])
            # Cd5 = np.max([25*SOC5, 35*SOC5 - 2, 65*SOC5 - 14, 125*SOC5 - 50, 1000*SOC5 - 750])
            # Cd6 = np.max([25*SOC6, 35*SOC6 - 2, 65*SOC6 - 14, 125*SOC6 - 50, 1000*SOC6 - 750])
            
            # cost of charging the battery based on SOC
            # Cc = np.max([-1000*SOC[Activebatt[:]] + 250, -125*SOC[Activebatt[:]] + 75, -65*SOC[Activebatt[:]] + 51, \
            #              -35*SOC[Activebatt[:]] + 33, -25*SOC[Activebatt[:]] + 25]) 
            # Cc0 = np.max([-1000*SOC0 + 250, -125*SOC0 + 75, -65*SOC0 + 51, -35*SOC0 + 33, -25*SOC0 + 25]) 
            # Cc1 = np.max([-1000*SOC1 + 250, -125*SOC1 + 75, -65*SOC1 + 51, -35*SOC1 + 33, -25*SOC1 + 25])
            # Cc2 = np.max([-1000*SOC2 + 250, -125*SOC2 + 75, -65*SOC2 + 51, -35*SOC2 + 33, -25*SOC2 + 25])
            # Cc3 = np.max([-1000*SOC3 + 250, -125*SOC3 + 75, -65*SOC3 + 51, -35*SOC3 + 33, -25*SOC3 + 25])
            # Cc4 = np.max([-1000*SOC4 + 250, -125*SOC4 + 75, -65*SOC4 + 51, -35*SOC4 + 33, -25*SOC4 + 25])
            # Cc5 = np.max([-1000*SOC5 + 250, -125*SOC5 + 75, -65*SOC5 + 51, -35*SOC5 + 33, -25*SOC5 + 25])
            # Cc6 = np.max([-1000*SOC6 + 250, -125*SOC6 + 75, -65*SOC6 + 51, -35*SOC6 + 33, -25*SOC6 + 25])
            
            # log.info(Cd)
            # log.info(Cc)
            
            
            # cost of usage includes both charging and discharging with charging and discharging thresholds            
            term01 = 1 - IsChargingReqd
            term02 = IsChargingReqd
            
            term1 = np.multiply(term01, Cd)

            dummyPgmax = np.multiply(Pgmax[Activediesel[:]].reshape(Ngactive,1),np.ones(T))
            
            dummyPbmax = np.multiply(Pbmax[Activebatt[:]].reshape(Nbactive,1),np.ones(T))
            
            term2 = dummyPbmax - Pb   
            
            term3 = np.multiply(term02, Cc)
            
            term4 = Pb
            
            CBatt = cvx.multiply(term1.reshape(Nbactive,1), term2) + cvx.multiply(term3.reshape(Nbactive,1), term4) 
            
            # log.info(term1.shape)
            # log.info(term2.shape)
            # log.info(term3.shape)
            # log.info(term4.shape)
            # log.info(CBatt.shape)
            
            # Cb0 = np.max([0,SOC0 - discharg_thres])*Cd0*(Pbmax0*np.ones(T) - Pb[0,:]) + np.max([0,charg_thres - SOC0])*Cc0*Pb[0,:] 
            # Cb1 = np.max([0,SOC1 - discharg_thres])*Cd1*(Pbmax1*np.ones(T) - Pb[1,:]) + np.max([0,charg_thres - SOC1])*Cc1*Pb[1,:]
            # Cb2 = np.max([0,SOC2 - discharg_thres])*Cd2*(Pbmax2*np.ones(T) - Pb[2,:]) + np.max([0,charg_thres - SOC2])*Cc2*Pb[2,:]
            # Cb3 = np.max([0,SOC3 - discharg_thres])*Cd3*(Pbmax3*np.ones(T) - Pb[3,:]) + np.max([0,charg_thres - SOC3])*Cc3*Pb[3,:]
            # Cb4 = np.max([0,SOC4 - discharg_thres])*Cd4*(Pbmax4*np.ones(T) - Pb[4,:]) + np.max([0,charg_thres - SOC4])*Cc4*Pb[4,:]
            # Cb5 = np.max([0,SOC5 - discharg_thres])*Cd5*(Pbmax5*np.ones(T) - Pb[5,:]) + np.max([0,charg_thres - SOC5])*Cc5*Pb[5,:]
            # Cb6 = np.max([0,SOC6 - discharg_thres])*Cd6*(Pbmax6*np.ones(T) - Pb[6,:]) + np.max([0,charg_thres - SOC6])*Cc6*Pb[6,:]
            
            
            # CBatt  = Cb0 + Cb1 + Cb2 + Cb3 + Cb4 + Cb5 + Cb6 
            
            
            # Total cost function
            C1 = np.ones(Ngactive)@Cg
            C2 = np.ones(Nbactive)@CBatt
            
            Cost = -np.ones(T)@C1 + np.ones(T)@Cload + np.ones(T)@C2
        
            # log.info(Cost.shape)
        
            #Constraints
            Constraints = []
            
            # total deferrable load power
            dummy2 = cvx.multiply(X,PLi)
            TotalDef = cvx.sum(dummy2, axis=0) 
            
            # total generated power
            TotalGen = np.ones(Ngactive)@Pg + np.ones(Nbactive)@Pb + PredPVOutput@np.ones(Npv)
                            
            consTerm1 = np.multiply(IsChargingReqd.reshape(Nbactive,1),dummyPbmax)
            consTerm2 = np.multiply((1-IsChargingReqd).reshape(Nbactive,1),dummyPbmax)
            
            Constraints.extend([ 
                                 -Ebc[Activebatt[:]] <= cvx.sum(Pb,axis=1), cvx.sum(Pb,axis=1) <= Ebd[Activebatt[:]],\
                                 # -Ebc0 <= cvx.sum(Pb[0,:]),  cvx.sum(Pb[0,:]) <= Ebd0,\
                                 # -Ebc1 <= cvx.sum(Pb[1,:]),  cvx.sum(Pb[1,:]) <= Ebd1,\
                                 # -Ebc2 <= cvx.sum(Pb[2,:]),  cvx.sum(Pb[2,:]) <= Ebd2,\
                                 # -Ebc3 <= cvx.sum(Pb[3,:]),  cvx.sum(Pb[3,:]) <= Ebd3,\
                                 # -Ebc4 <= cvx.sum(Pb[4,:]),  cvx.sum(Pb[4,:]) <= Ebd4,\
                                 # -Ebc5 <= cvx.sum(Pb[5,:]),  cvx.sum(Pb[5,:]) <= Ebd5,\
                                 # -Ebc6 <= cvx.sum(Pb[6,:]),  cvx.sum(Pb[6,:]) <= Ebd6,\
                                 0 <= cvx.sum(Pg,axis=1),  cvx.sum(Pg,axis=1) <= Eg[Activediesel[:]],\
                                 # 0 <= cvx.sum(Pg),  cvx.sum(Pg) <= Eg0,\
                                 -consTerm1.T.flatten() <= cvx.vec(Pb), cvx.vec(Pb) <= consTerm2.T.flatten(),\
                                 # -Pbmax0*np.ones(T)*np.max([np.max([0,charg_thres - SOC0])/(charg_thres - SOC0), 0]) <= Pb[0,:], Pb[0,:] <= Pbmax0*np.ones(T),\
                                 # -Pbmax1*np.ones(T)*np.max([np.max([0,charg_thres - SOC1])/(charg_thres - SOC1), 0]) <= Pb[1,:], Pb[1,:] <= Pbmax1*np.ones(T),\
                                 # -Pbmax2*np.ones(T)*np.max([np.max([0,charg_thres - SOC2])/(charg_thres - SOC2), 0]) <= Pb[2,:], Pb[2,:] <= Pbmax2*np.ones(T),\
                                 # -Pbmax3*np.ones(T)*np.max([np.max([0,charg_thres - SOC3])/(charg_thres - SOC3), 0]) <= Pb[3,:], Pb[3,:] <= Pbmax3*np.ones(T),\
                                 # -Pbmax4*np.ones(T)*np.max([np.max([0,charg_thres - SOC4])/(charg_thres - SOC4), 0]) <= Pb[4,:], Pb[4,:] <= Pbmax4*np.ones(T),\
                                 # -Pbmax5*np.ones(T)*np.max([np.max([0,charg_thres - SOC5])/(charg_thres - SOC5), 0]) <= Pb[5,:], Pb[5,:] <= Pbmax5*np.ones(T),\
                                 # -Pbmax6*np.ones(T)*np.max([np.max([0,charg_thres - SOC6])/(charg_thres - SOC6), 0]) <= Pb[6,:], Pb[6,:] <= Pbmax6*np.ones(T),\
                                 0 <= cvx.vec(Pg), cvx.vec(Pg) <= dummyPgmax.T.flatten(),\
                                 # 0 <= Pg, Pg <= Pgmax0*np.ones(T),\
                                 Pslkmin*np.ones(T) <= Pslk, Pslk <= Pslkmax*np.ones(T),\
                                 TotalDef.T + Pc.T <= TotalGen + Pslk, TotalDef.T + Pc.T >= 0.95*(TotalGen + Pslk)
                                 
                                 ])
        
            
            mosek_params = {
                "MSK_DPAR_MIO_TOL_ABS_RELAX_INT": 1e-3,
                "MSK_DPAR_MIO_TOL_ABS_GAP": 1e-2,
                "MSK_DPAR_MIO_TOL_REL_GAP": 1e-2,
                "MSK_DPAR_MIO_REL_GAP_CONST": 1e-5,
                
                # "MSK_IPAR_MIO_MAX_NUM_BRANCHES": 10,
                # "MSK_IPAR_MIO_MAX_NUM_RELAXS": 100,
                # "MSK_IPAR_MIO_MAX_NUM_SOLUTIONS": 5,
                
                "MSK_DPAR_MIO_MAX_TIME": 1200,
                "MSK_DPAR_OPTIMIZER_MAX_TIME": 1200,
                "MSK_IPAR_MIO_SEED": 10
            }
                
            prob = cvx.Problem(cvx.Minimize(Cost), Constraints)
            
            X.value = prevSoln
        
            prob.solve(solver = cvx.MOSEK, mosek_params=mosek_params, verbose=False, warm_start=True)
                        
            if prob.status == 'optimal':
                # If the optimization worked
                converged = 1
                
                endTime = time.time() - tStart - InitTime
                endTime = np.ceil(endTime)
                endTime = np.intc(endTime)
                
                if endTime < T:
                    Xr = X[:,endTime].value   # picking the values corresponding to the last/endTime, 
                   
                    Pbsetpt = np.zeros(Nb)
                    Pbsetpt[Activebatt[:]] = Pb[:,endTime].value # unlike MPC which picks values based on the starting time instant
                    
                    Pgsetpt = np.zeros(Ng)
                    Pgsetpt[Activediesel[:]] = Pg[:,endTime].value
                    
                    Pslksetpt = Pslk[endTime].value 
                    
                else:
                    Xr = X[:,T-1].value # picking the values corresponding to the last/endTime, 
                    
                    Pbsetpt = np.zeros(Nb)
                    Pbsetpt[Activebatt[:]] = Pb[:,T-1].value  # unlike MPC which picks values based on the starting time instant
                    
                    Pgsetpt = np.zeros(Ng)
                    Pgsetpt[Activediesel[:]] = Pg[:,T-1].value
                    
                    Pslksetpt = Pslk[T-1].value
                              
                return Xr, -Pbsetpt, -Pgsetpt, -Pslksetpt, HoV_ON, converged
            else:
                converged = 0
                endTime = time.time() - tStart - InitTime
                endTime = np.ceil(endTime)
                endTime = np.intc(endTime)
                
                
                if endTime < T:
                    
                    Pgsetpt = np.zeros(Ng)
                    Pgsetpt[Activediesel[:]] = Pgmax[Activediesel[:]]

                    Pbsetpt = np.zeros(Nb)
                    
                    for i in range(Nbactive):
                        if IsChargingReqd[i] == 0:
                            Pbsetpt[Activebatt[i]] = Pbmax[Activebatt[i]]
                        else:
                            Pbsetpt[Activebatt[i]] = -Pbmax[Activebatt[i]]

                    Pslksetpt = 0.999*Pslkmax # 0.99 is artificial; added to not stress the Slack bus and can be customized to the actual scenario
                    
                    PGenTotal = np.sum(PredPVOutput[endTime,:]) + np.sum(Pbsetpt) + np.sum(Pgsetpt) + Pslksetpt
                                         
                    LoadCosts = np.unique(CL) # find cost vector with cost entries so that their multiplicity is equal to one
                    sortedCost = np.sort(LoadCosts) # ascending order of cost
                    sortedCost = sortedCost[::-1] # descending order of cost
                                       
                    LoadIndex = 0    
                    for i in range(len(LoadCosts)):
                        LoadIndex1 = np.argwhere(CL==sortedCost[i])
                        LoadIndex = np.concatenate((LoadIndex1,LoadIndex),axis = None)                    
  
                    LdPower = Pc[endTime]

                    loadfinal = 0 # minimum number of highest priority loads that led to power balance
                    for l in range(len(LoadIndex)-1): # run until len(LoadIndex)-1 because last entry of LoadIndex is equal to 0 which is a junk entry
                        LdPower = LdPower + PLi[LoadIndex[l],endTime]
                        if LdPower == PGenTotal:
                            loadfinal = l
                            break
                        elif LdPower > PGenTotal:
                            loadfinal = l - 1
                            break
 
                    if loadfinal == 0:
                        Xr = np.ones(Nd)  # All loads ON. Can change this to charge some of the batteries
                    else:
                        Xr = np.zeros(Nd)
                        for j in range(loadfinal):
                            Xr[LoadIndex[j]] = 1 # turn ON loadfinal number of loads with highest cost/priority

                    ONindex = np.flatnonzero(Xr) # indices of loads that are decided to be turned ON
                    
                    PloadON = Pc[endTime] + np.sum(PLi[ONindex,endTime])  # total load power ON
                    
                    PGenWithoutSlack = np.sum(PredPVOutput[endTime,:]) + np.sum(Pbsetpt) + np.sum(Pgsetpt)
                    Pslksetpt = PloadON -  PGenWithoutSlack  

                else:

                    Pgsetpt = np.zeros(Ng)
                    Pgsetpt[Activediesel[:]] = Pgmax[Activediesel[:]]

                    Pbsetpt = np.zeros(Nb)
                    for i in range(Nbactive):
                        if IsChargingReqd[i] == 0:
                            Pbsetpt[Activebatt[i]] = Pbmax[Activebatt[i]]
                        else:
                            Pbsetpt[Activebatt[i]] = -Pbmax[Activebatt[i]]
                    
                    Pslksetpt = 0.999*Pslkmax # 0.99 is artificial; added to not stress the Slack bus and can be customized to the actual scenario 
                    
                    PGenTotal = np.sum(PredPVOutput[T-1,:]) + np.sum(Pbsetpt) + np.sum(Pgsetpt) + Pslksetpt
        
                    LoadCosts = np.unique(CL) # find cost vector with cost entries so that their multiplicity is equal to one
                    sortedCost = np.sort(LoadCosts) # ascending order of cost
                    sortedCost = sortedCost[::-1] # descending order of cost

                    LoadIndex = 0    
                    for i in range(len(LoadCosts)):
                        LoadIndex1 = np.argwhere(CL==sortedCost[i])
                        LoadIndex = np.concatenate((LoadIndex1,LoadIndex),axis = None)

                    LdPower = Pc[T-1]
                    loadfinal = 0 # minimum number of highest priority loads that led to power balance
                    for l in range(len(LoadIndex)-1): # run until len(LoadIndex)-1 because last entry of LoadIndex is equal to 0 which is a junk entry
                        LdPower = LdPower + PLi[LoadIndex[l],T-1]                        
                        if LdPower == PGenTotal:
                            loadfinal = l
                            break
                        elif LdPower > PGenTotal:
                            loadfinal = l - 1
                            break
                    
                    if loadfinal == 0:
                        Xr = np.ones(Nd)  # All loads ON. Can change this to charge some of the batteries
                    else:
                        Xr = np.zeros(Nd)
                        for j in range(loadfinal):
                            Xr[LoadIndex[j]] = 1 # turn ON loadfinal number of loads with highest cost/priority
                                                
                    ONindex = np.flatnonzero(Xr) # indices of loads that are decided to be turned ON
                    
                    PloadON = Pc[T-1] + np.sum(PLi[ONindex,T-1])  # total load power ON
                    
                    PGenWithoutSlack = np.sum(PredPVOutput[T-1,:]) + np.sum(Pbsetpt) + Pgsetpt
                    Pslksetpt = PloadON -  PGenWithoutSlack 
                    
                return Xr, -Pbsetpt, -Pgsetpt, -Pslksetpt, HoV_ON, converged
                log.info('Did not work')
         

def execHoVDispatch():
    """
    Execute the HoV algorithm and send out dispatch command vector
    """
    global Horizon 
    global opt_sol_found
    global Pbsetpt
    global Pgsetpt
    global Pslksetpt
    global measVec
    global measurement_reception_started
    
    while threadGo:
        
        if measurement_reception_started == 1:
            
            LoadStatus, Pbsetpt, Pgsetpt, Pslksetpt, HoV_ON, opt_sol_found = HoVengine(measVec,Horizon)  
        
            if HoV_ON == 1:
                kw_w = 1e3 # kW to Watt change in the dispatch quantities
                
                if opt_sol_found == 1:                
                    t2 = time.time() - tStart
                    log.info("Optimal HoV dispatch:") 
                    log.info("Dispatch time=" + str(t2))
                    log.info("LoadStatus=" + str(LoadStatus))
                    log.info("PbSetpt=" + str(Pbsetpt*kw_w))
                    log.info("PgSetpt=" + str(Pgsetpt*kw_w))
                    log.info("PSlack=" + str(Pslksetpt*kw_w))
                    HoVcmd = np.concatenate((LoadStatus, Pbsetpt*kw_w, Pgsetpt*kw_w), axis=None)  
                    sendOutHov(HoVcmd)
                    # time.sleep(60)
                else:
                    t2 = time.time() - tStart
                    log.info("Did not find optimal solution but a feasible solution is implemented")
                    log.info("Dispatch time=" + str(t2))
                    log.info("LoadStatus=" + str(LoadStatus))
                    log.info("PbSetpt=" + str(Pbsetpt*kw_w))
                    log.info("PgSetpt=" + str(Pgsetpt*kw_w))
                    log.info("PSlack=" + str(Pslksetpt*kw_w))
                    HoVcmd = np.concatenate((LoadStatus, Pbsetpt*kw_w, Pgsetpt*kw_w), axis=None)  
                    sendOutHov(HoVcmd)
                    # time.sleep(45)
                    

def initializeViaConfigFile(filename):
    """
    Loads scenario config file and runs some checks to ensure all data is correct. Initializes stateTable using data in the config file

    -- Returns --
    True if initialization was successful, False if an error occurred
    """
    
    global ConfigData 
    global Ng 
    global Nb 
    global Nc
    global Npv
    global Nd
    global measVec 
    global measurement_reception_started
    global IsChargingReqd
    
    #verify that file is of XLSX format (or at least filename is...)
    if filename[-5:] != ".xlsx":
        log.error('Config file is not a XLSX file')
        return False
    
    #read in XLSX file via pandas
    df = pd.read_excel(filename, sheet_name='Sheet1')
    cols = df.columns

    # verify that file contains all columns required
    if ("type" not in cols) or ("status" not in cols) or ("cost_up" not in cols) or ("cost_down" not in cols) or ("Pmeas" not in cols) or ("Pmin" not in cols) or ("Pmax" not in cols):
        log.error("Not all required columns are present in config file")
        return False

    # verify that file contains exactly one type=10 device (slack generator)
    if len(df[df['type'] == 10]) != 1:
        log.error("Config file must contain exactly one slack generator/grid-forming source definition")
        return False
    
    # go through all rows and ensure: min <= meas <= max, if a load then min = 0
    for ridx, row in df.iterrows():
        #check min <= meas <= max
        if (row['Pmin'] > row['Pmeas']) or (row['Pmax'] < row['Pmeas']):
            log.error("Config file error on 0-based row #" + str(ridx) + ": must satisfy Pmin<=Pmeas<=Pmax")
            return False
        #for loads, check min = 0
        if row['type'] == 0:
            if not row['Pmin'] == 0:
                log.error("Config file error on 0-based row #" + str(ridx) + ": loads (type 0) must have Pmin=0")
                return False

    ConfigData = pd.read_excel(filename, sheet_name='Sheet1')
    ConfigData = np.array(ConfigData)
    
    # number of controllable loads
    Nd = np.argwhere(ConfigData[:,0] == 0).size
    # number of critical loads
    Nc = np.argwhere(ConfigData[:,0] == 1).size
    # number of PV inverters
    Npv = np.argwhere(ConfigData[:,0] == 3).size  
    # number of GFL Battery inverters
    Nb = np.array(np.where( (ConfigData[:,0] == 4) & (ConfigData[:,6] > 0))).size 
    # number of diesel generators
    Ng = np.array(np.where( (ConfigData[:,0] == 4) & (ConfigData[:,6] == 0))).size

    
    MeasVeclen = 1 + Nd + 1 + Npv*2 + Nb*4 + Ng*4 + 1*3 + 1 # First 1 is for time stamp and last 1 is the grid power
    measVec = np.zeros(MeasVeclen)
   
    measurement_reception_started = 0
    
    IsChargingReqd = np.zeros(Nb)

    log.info("Successfully initialized for a system with " + str(Nd) + " Controllable loads, " + str(Npv) + " PV invs, and " + str(Ng+Nb) + " Batt invs/ Diesel Gens.")

    return True
          
def receiveMeasHoV():
    """
    Listen for measurements provided by the real-time simulated power system model
    Upon receipt of a measurement, update the local measurement vector and relays the latest measurement vector to load management engine 
    """
 
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
    s.bind(LOCAL_UDP_ADDR) #IP:Port here is for the computer running this script
    log.info('Bound to ' + str(LOCAL_UDP_ADDR) + ' and waiting for UDP packet-based measurements')
    
    
    global measVec
    global measurement_reception_started
    global Ng 
    global Nb 
    global Npv
    global Nd
    
    
    while threadGo:
        #use UDP comms - receive packet with array
        data,addr = s.recvfrom(10000) #buffer size can be smaller if needed, but this works fine
        l = len(data)
        p = int(l/4) #get number of floats sent
        
        vector = np.array(struct.unpack('<{}'.format('f'*p),data)) 
        
        inputmeasveclen = Nd + 1 + Npv*2 + Nb*4 + Ng*4 + 1*3 + 1
             
        if len(vector) == inputmeasveclen:   
            measurement_reception_started = 1                
            # ldPV = vector[0:Nd+1] 
            
            # if opt_sol_found:
            #     # Send the updated 
            #     # fix the slack bus pmax pmin for the GFLs to the actual power setpoint for the use in NLM engine
            #     ContGenPmeas = np.array([Pbsetpt[0],Pbsetpt[1],Pbsetpt[2],Pbsetpt[3],Pbsetpt[4],Pbsetpt[5],Pbsetpt[6],Pgsetpt,Pslksetpt]) # 7 Batt, 1 Diesel, 1 Slack
            #     ContGenPmin = np.array([Pbsetpt[0],Pbsetpt[1],Pbsetpt[2],Pbsetpt[3],Pbsetpt[4],Pbsetpt[5],Pbsetpt[6],Pgsetpt,vector[178]]) # 7 Batt, 1 Diesel, 1 Slack
            #     ContGenPmax = -ContGenPmin
                
            #     # ContGenPmin = np.array([vector[154],vector[157],vector[160],vector[163],vector[166],vector[169],vector[172],vector[175],vector[178]]) # 7 Batt, 1 Diesel, 1 Slack
            #     # ContGenPmax = np.array([vector[155],vector[158],vector[161],vector[164],vector[167],vector[170],vector[173],vector[176],vector[179]]) # 7 Batt, 1 Diesel, 1 Slack      
            # else:
            #     # fix the slack bus pmax pmin for the GFLs to the actual power setpoint for the use in NLM engine
            #     ContGenPmeas = np.array([vector[153],vector[156],vector[159],vector[162],vector[165],vector[168],vector[171],vector[174],vector[177]]) # 7 Batt, 1 Diesel, 1 Slack
            #     ContGenPmin = np.array([vector[153],vector[156],vector[159],vector[162],vector[165],vector[168],vector[171],vector[174],vector[178]]) # 7 Batt, 1 Diesel, 1 Slack
            #     ContGenPmax = -ContGenPmin
                
            #     # ContGenPmin = np.array([vector[154],vector[157],vector[160],vector[163],vector[166],vector[169],vector[172],vector[175],vector[178]]) # 7 Batt, 1 Diesel, 1 Slack
            #     # ContGenPmax = np.array([vector[155],vector[158],vector[161],vector[164],vector[167],vector[170],vector[173],vector[176],vector[179]]) # 7 Batt, 1 Diesel, 1 Slack   
            
            # nlmVec = np.concatenate((ldPV,np.sum(ContGenPmeas),np.sum(ContGenPmin),np.sum(ContGenPmax),vector[180]),axis=None) # 180 is the grid measurement
            
            dt = time.time() - tStart
            measVec = np.concatenate((dt,vector),axis=None)

        
        log.info("Received udp-based measurement packet")
        # log.debug(str(measVec))


def sendOutHov(outVec):
    #REMOTE_HOST_ADDR = ('127.0.0.1', 7300) #IP:Port of remote host to send to
    #send via UDP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
    p = len(outVec) #determine how many floats we are sending\n",
    msgBytes = struct.pack('<{}'.format('f'*p),*outVec.tolist()) #bytepack numpy row vector - use little Endian format (required by Opal-RT)\n",
    s.sendto(msgBytes, REMOTE_HOST_ADDR)
    
    dt = time.time() - tStart
    to_log = np.concatenate((dt,outVec),axis=None)
    #log latest stateTable Pmeas to scenario log file
    with open(HOV_LOG_FILENAME, "a") as f:
        a = to_log
        a = a.reshape((1,len(a)))
        np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="filename of config file (XLSX format) to define the scenario", default='-1')
    ap.add_argument("--loadprediction", help="filename of predicted load profiles (XLSX format)", default='-1')
    ap.add_argument("--pvpred", help="filename of scenario profile (XLSX format) defining the predicted PV profile for this scenario", default='-1')
    ap.add_argument("--debug", action="store_true", help="turn debug mode on, including additional logging and messages")
    ap.add_argument("--horizon", help="Time horizon for the HoV engine", default='-1')
    ap.add_argument("--localip", help="IPv4 address of this computer. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--localport", help="Port to receive UDP packets on for this computer. Default is 4100", default="4100")
    ap.add_argument("--remoteip", help="IPv4 address of the remote computer to send UDP-based commands to. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--remoteport", help="Port on remote computer to send UDP-based commands to. Default is 7300", default="7300")

    args = ap.parse_args()

    debugModeON = args.debug

    if debugModeON:
        log.setLevel(logging.DEBUG)
        log.info("DEBUG mode activated")
    else:
        log.setLevel(logging.INFO)
    
    if args.config == '-1' or args.config[-5:] != ".xlsx":
        log.error("ERROR: must define load profile file in XLSX format")
        sys.exit()
    else:
        scenarioConfigFile = args.config
        
      
    if args.loadprediction == '-1' or args.loadprediction[-5:] != ".xlsx":
        log.error("ERROR: must define load profile file in XLSX format")
        sys.exit()
    else:
        PredloadProfile = args.loadprediction       

    if args.pvpred == '-1' or args.pvpred[-5:] != ".xlsx":
        log.error("ERROR: must define scenario config file in XLSX format")
        sys.exit()
    else:
        slopePVprofile = args.pvpred    
    
    Horizon = int(args.horizon)

    if Horizon != -1:
        if Horizon < 0:
            print("Error: horizon must be > 0")
            sys.exit()  
    
    udpPortLocal = int(args.localport)
    udpPortRemote = int(args.remoteport)

    if udpPortLocal < 100 or udpPortLocal > 100000 or udpPortRemote < 100 or udpPortRemote > 100000:
        log.error("ERROR: both local and remote ports must be in range 100-100000")
        sys.exit()

    LOCAL_UDP_ADDR = (args.localip, udpPortLocal) #IP address and port to listen for measurements on for computer running this script
    REMOTE_HOST_ADDR = (args.remoteip, udpPortRemote) #IP:Port of remote host to send cmds to
    
    PVslope = pd.read_excel(slopePVprofile, sheet_name='Sheet1')
    PVslope = np.array(PVslope)    
    
    loadData = pd.read_excel(PredloadProfile, sheet_name='Sheet1')
    loadData = np.array(loadData)
            

    #remove any existing log file that we appended to last time
    try:
        os.remove(HOV_LOG_FILENAME)
    except FileNotFoundError as e:
        pass
    

    #initialize stateTable based on the scenario definition file
    initSuccess = initializeViaConfigFile(scenarioConfigFile)    

    # setup threads
    # setup var for threads
    threadGo = True #set to false to quit threads
    
    tStart = time.time()
    
    # startup thread to listen for measurements from DCG
    thread1 = Thread(target=receiveMeasHoV)
    thread1.daemon = True
    thread1.start()
    
    # startup thread for optimization engine
    thread2 = Thread(target=execHoVDispatch)
    thread2.daemon = True
    thread2.start()

    
    # while True:
    #     print("Update data:", time.time())
    #     sleep = 30 - int(time.time()) % 30
    #     if sleep == 30:                   
    #         # startup thread to listen for measurements from DCG
    #         thread1 = Thread(target=receiveMeasHoV)
    #         thread1.daemon = True
    #         thread1.start()
            
    #         # startup thread for optimization engine
    #         thread2 = Thread(target=execHoVDispatch)
    #         thread2.daemon = True
    #         thread2.start()
    
    #         time.sleep(1)
    #     else:
    #         time.sleep(1)


    ## do infinite while to keep main alive so that logging from threads shows in console
    try:
        while True:
            time.sleep(0.001)
    except KeyboardInterrupt:
        log.info("cleaning up threads and exiting...")
        threadGo = False
        log.info("done.")
