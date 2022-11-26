# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 17:19:02 2021

"""

# -*- coding: utf-8 -*-
__version__ = "1.1"
__author__ = " Vivek Khatana, UMN"

import pandas as pd
import logging
import time
import numpy as np
import cvxpy as cvx
np.set_printoptions(suppress=True,edgeitems=10)
from threading import Thread, Lock
import sys
import argparse
import socket
import struct
import os

####
SCENARIO_LOG_FILENAME = "CsCfState.csv"
####


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
    
    
####### This code is for non-charging scenarios #######   
 
def HoVengine(Vec,PVslope,Horizon):
    """           
    -- Returns --
    X, PVsetpt, Pgsetpt, Pbsetpt
    X : cvxpy array
        Binary-encoded (1=on, 0=off) Defereable load status over the horizon
    PVsetpt : 
        Active power set points for the PV inverter 
    Pgsetpt :  
        Active power set points for the diesel generators 
    Pbsetpt :  
        Active power set points for the battery # 1
    """
    HoV_ON = 1 # flag to ensure required generation for the HoV dispatch; HoV_ON = 1: HoV engine active, 0: HoV engine not used
    
    
    # initialization 
    
    # time discretization to get the energy remaining
    dt = 3600; #100
    
    #initial energy rating of the generation sources  
    # Battery energy
    Eb_initial = np.array([70000,70000,18000,56000,140000,35000,70000]) 
    # Diesel Genset energy
    Eg_initial = 140000 
    
    # Deferrable loads cost vector    
    #CL = np.array([10, 11, 12, 13, 12, 12, 13, 10, 50, 55, 30, 32, 34, 33, 31, 10, 30, 35, 38, 41, 42, 25, 28, 24, 21, 1, 1, 1, 1,	1, 0, 0, 0,	100]);
    CL = DefLoadCost[:,0]
    Plmaxinv = DefLoadCost[:,1]
    Plmaxinv = np.array(Plmaxinv).reshape(138,1)
    Plmaxinv = np.multiply(Plmaxinv, np.ones((1,Horizon)))
    
    # Horizon receeding horizon controller
    T = Horizon
    
    # flag to check if the optimal solution is acheieved
    converged = 0
    
    # Current value of Battery SOCs
    SOC0 = Vec[182]; SOC1 = Vec[183]; SOC2 = Vec[184]; SOC3 = Vec[185]; SOC4 = Vec[186]; SOC5 = Vec[187]; 
    SOC6 = Vec[188]; 
    
    # Current value of Diesel Genset SOEs
    SOE0 = Vec[189]
    
     
    # Energy remaining in the generation sources     
    # Remaining energy in Battery inverters
    Eb0 = SOC0*Eb_initial[0]*dt; Eb1 = SOC1*Eb_initial[1]*dt; Eb2 = SOC2*Eb_initial[2]*dt; 
    Eb3 = SOC3*Eb_initial[3]*dt; Eb4 = SOC4*Eb_initial[4]*dt; Eb5 = SOC5*Eb_initial[5]*dt;
    Eb6 = SOC6*Eb_initial[6]*dt;    
 
    # Remaining energy in Diesel Gensets
    Eg0 = SOE0*Eg_initial*dt;  

    
    # Initial time at the start of MPC (time when the measurement of the power system was taken)
    InitTime = np.ceil(Vec[0]); InitTime = np.intc(InitTime);
        
    # Slope of the PV power profile to caluclate the predicted PV power values in the future    
    slope0 =  np.cumprod(PVslope[InitTime:InitTime+T-1,0]); slope1 =  np.cumprod(PVslope[InitTime:InitTime+T-1,1]);
    slope2 =  np.cumprod(PVslope[InitTime:InitTime+T-1,2]); slope3 =  np.cumprod(PVslope[InitTime:InitTime+T-1,3]);
    slope4 =  np.cumprod(PVslope[InitTime:InitTime+T-1,4]); slope5 =  np.cumprod(PVslope[InitTime:InitTime+T-1,5]);
    slope6 =  np.cumprod(PVslope[InitTime:InitTime+T-1,6]); 
    
       
    # Total critical load power demand
    Pc = loadData[InitTime:InitTime+T,139]
    
    # Controllable/deferable loads instantaneous power demand
    PLi = loadData[InitTime:InitTime+T,1:139].T
    Nd,Nimp = PLi.shape
       
    # Upper bound in the available power from the generation sources 
    # PV inverter power max values
    PV0 = np.concatenate((-Vec[140],-Vec[140]*slope0), axis=None); PV1 = np.concatenate((-Vec[142],-Vec[142]*slope1), axis=None);
    PV2 = np.concatenate((-Vec[144],-Vec[144]*slope2), axis=None); PV3 = np.concatenate((-Vec[146],-Vec[146]*slope3), axis=None);
    PV4 = np.concatenate((-Vec[148],-Vec[148]*slope4), axis=None); PV5 = np.concatenate((-Vec[150],-Vec[150]*slope5), axis=None);
    PV6 = np.concatenate((-Vec[152],-Vec[152]*slope6), axis=None); 
    
    PV = np.array([PV0,PV1,PV2,PV3,PV4,PV5,PV6])
    
    # Battery power max values
    Pbmax0  = -Vec[155]; Pbmax1 = -Vec[158]; Pbmax2  = -Vec[161]; Pbmax3 = -Vec[164]; Pbmax4  = -Vec[167]; Pbmax5 = -Vec[170]; 
    Pbmax6  = -Vec[173]; 
    
    # Diesel Gensets power max values
    Pgmax0 = -Vec[176]; 
    
    
    # Slack bus power max value
    Pslkmax = -Vec[179];
    Pslkmin = -Vec[180];
    
    if PV0[0] == 0 and Pgmax0 == 0 and Pbmax0 == 0:
        HoV_ON = 0
        return np.zeros(Nd), np.zeros(7), np.zeros(7), 0, 0, HoV_ON, 0

    if HoV_ON == 1:   
        if Pgmax0 > 0 and Pbmax0 > 0:
            # Optimization problem formulation
            Pg = cvx.Variable(T)
            # PV = cvx.Variable((7,T))
            Pb = cvx.Variable((7,T))
            Pslk = cvx.Variable(T)
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function of the Diesel Gensets
            Cg0 = 10*cvx.minimum( (18.8/20)*(100*Pg/Pgmax0 - 20) + 20,\
                                    (15.1/80)*(100*Pg/Pgmax0 - 100) + 33.9 )
            # Cg1 = 10*cvx.minimum( (18.8/20)*(100*Pg[1,:]/Pgmax1 - 20) + 20,\
            #                         (15.1/80)*(100*Pg[1,:]/Pgmax1 - 100) + 33.9 )   
                
            Cg = Cg0

            # Down cost of the deferrrable loads
            dummy = np.ones((Nd,T))-X
            dummy0 = cvx.multiply(dummy,PLi) 
            dummy1 = cvx.multiply(dummy0,Plmaxinv)
            Cload = CL@dummy1
                               
            # Cost related to PV inverter power utilization
            # CPV0 = 10*(PVmax0 - PV[0,:]); CPV1 = 10*(PVmax1 - PV[1,:]); CPV2 = 10*(PVmax2 - PV[2,:]); CPV3 = 10*(PVmax3 - PV[3,:]);
            # CPV4 = 10*(PVmax4 - PV[4,:]); CPV5 = 10*(PVmax5 - PV[5,:]); CPV6 = 10*(PVmax6 - PV[6,:]);
            
            # CPV = CPV0 + CPV1 + CPV2 + CPV3 + CPV4 + CPV5 + CPV6 
            
            # Cost of battery dispacth
            # cost of discharging the battery based on SOC
            Cd0 = np.max([25*SOC0, 35*SOC0 - 2, 65*SOC0 - 14, 125*SOC0 - 50, 1000*SOC0 - 750]) 
            Cd1 = np.max([25*SOC1, 35*SOC1 - 2, 65*SOC1 - 14, 125*SOC1 - 50, 1000*SOC1 - 750])
            Cd2 = np.max([25*SOC2, 35*SOC2 - 2, 65*SOC2 - 14, 125*SOC2 - 50, 1000*SOC2 - 750])
            Cd3 = np.max([25*SOC3, 35*SOC3 - 2, 65*SOC3 - 14, 125*SOC3 - 50, 1000*SOC3 - 750])
            Cd4 = np.max([25*SOC4, 35*SOC4 - 2, 65*SOC4 - 14, 125*SOC4 - 50, 1000*SOC4 - 750])
            Cd5 = np.max([25*SOC5, 35*SOC5 - 2, 65*SOC5 - 14, 125*SOC5 - 50, 1000*SOC5 - 750])
            Cd6 = np.max([25*SOC6, 35*SOC6 - 2, 65*SOC6 - 14, 125*SOC6 - 50, 1000*SOC6 - 750])
            
            # cost of charging the battery based on SOC
            Cc0 = np.max([-1000*SOC0 + 250, -125*SOC0 + 75, -65*SOC0 + 51, -35*SOC0 + 33, -25*SOC0 + 25]) 
            Cc1 = np.max([-1000*SOC1 + 250, -125*SOC1 + 75, -65*SOC1 + 51, -35*SOC1 + 33, -25*SOC1 + 25])
            Cc2 = np.max([-1000*SOC2 + 250, -125*SOC2 + 75, -65*SOC2 + 51, -35*SOC2 + 33, -25*SOC2 + 25])
            Cc3 = np.max([-1000*SOC3 + 250, -125*SOC3 + 75, -65*SOC3 + 51, -35*SOC3 + 33, -25*SOC3 + 25])
            Cc4 = np.max([-1000*SOC4 + 250, -125*SOC4 + 75, -65*SOC4 + 51, -35*SOC4 + 33, -25*SOC4 + 25])
            Cc5 = np.max([-1000*SOC5 + 250, -125*SOC5 + 75, -65*SOC5 + 51, -35*SOC5 + 33, -25*SOC5 + 25])
            Cc6 = np.max([-1000*SOC6 + 250, -125*SOC6 + 75, -65*SOC6 + 51, -35*SOC6 + 33, -25*SOC6 + 25])
            
            # cost of usage includes both charging and discharging with charging and discharging thresholds
            discharg_thres = 0.10; charg_thres = 0.20;
            Cb0 = np.max([0,SOC0 - discharg_thres])*Cd0*(Pbmax0*np.ones(T) - Pb[0,:]) + np.max([0,charg_thres - SOC0])*Cc0*Pb[0,:] 
            Cb1 = np.max([0,SOC1 - discharg_thres])*Cd1*(Pbmax1*np.ones(T) - Pb[1,:]) + np.max([0,charg_thres - SOC1])*Cc1*Pb[1,:]
            Cb2 = np.max([0,SOC2 - discharg_thres])*Cd2*(Pbmax2*np.ones(T) - Pb[2,:]) + np.max([0,charg_thres - SOC2])*Cc2*Pb[2,:]
            Cb3 = np.max([0,SOC3 - discharg_thres])*Cd3*(Pbmax3*np.ones(T) - Pb[3,:]) + np.max([0,charg_thres - SOC3])*Cc3*Pb[3,:]
            Cb4 = np.max([0,SOC4 - discharg_thres])*Cd4*(Pbmax4*np.ones(T) - Pb[4,:]) + np.max([0,charg_thres - SOC4])*Cc4*Pb[4,:]
            Cb5 = np.max([0,SOC5 - discharg_thres])*Cd5*(Pbmax5*np.ones(T) - Pb[5,:]) + np.max([0,charg_thres - SOC5])*Cc5*Pb[5,:]
            Cb6 = np.max([0,SOC6 - discharg_thres])*Cd6*(Pbmax6*np.ones(T) - Pb[6,:]) + np.max([0,charg_thres - SOC6])*Cc6*Pb[6,:]
            
            
            CBatt  = Cb0 + Cb1 + Cb2 + Cb3 + Cb4 + Cb5 + Cb6 
            
            
            # Total cost function
            # Cost = -np.ones(T)@Cg + np.ones(T)@Cload + np.ones(T)@CPV + np.ones(T)@CBatt
            Cost = -np.ones(T)@Cg + np.ones(T)@Cload + np.ones(T)@CBatt
        
            #Constraints
            Constraints = []
            
            # total deferrable load power
            dummy2 = cvx.multiply(X,PLi)
            TotalDef = cvx.sum(dummy2, axis=0) 
            
            # total generated power
            TotalGen = Pg + np.ones(7)@PV + np.ones(7)@Pb 
             
            Constraints.extend([ 
                                 -Eb0 <= cvx.sum(Pb[0,:]),  cvx.sum(Pb[0,:]) <= Eb0,\
                                 -Eb1 <= cvx.sum(Pb[1,:]),  cvx.sum(Pb[1,:]) <= Eb1,\
                                 -Eb2 <= cvx.sum(Pb[2,:]),  cvx.sum(Pb[2,:]) <= Eb2,\
                                 -Eb3 <= cvx.sum(Pb[3,:]),  cvx.sum(Pb[3,:]) <= Eb3,\
                                 -Eb4 <= cvx.sum(Pb[4,:]),  cvx.sum(Pb[4,:]) <= Eb4,\
                                 -Eb5 <= cvx.sum(Pb[5,:]),  cvx.sum(Pb[5,:]) <= Eb5,\
                                 -Eb6 <= cvx.sum(Pb[6,:]),  cvx.sum(Pb[6,:]) <= Eb6,\
                                 0 <= cvx.sum(Pg),  cvx.sum(Pg) <= Eg0,\
                                 # 0.9*PVmax0 <= PV[0,:], PV[0,:] <= PVmax0,\
                                 # 0.9*PVmax1 <= PV[1,:], PV[1,:] <= PVmax1,\
                                 # 0.9*PVmax2 <= PV[2,:], PV[2,:] <= PVmax2,\
                                 # 0.9*PVmax3 <= PV[3,:], PV[3,:] <= PVmax3,\
                                 # 0.9*PVmax4 <= PV[4,:], PV[4,:] <= PVmax4,\
                                 # 0.9*PVmax5 <= PV[5,:], PV[5,:] <= PVmax5,\
                                 # 0.9*PVmax6 <= PV[6,:], PV[6,:] <= PVmax6,\
                                 -Pbmax0*np.ones(T)*np.max([np.max([0,charg_thres - SOC0])/(charg_thres - SOC0), 0]) <= Pb[0,:], Pb[0,:] <= Pbmax0*np.ones(T),\
                                 -Pbmax1*np.ones(T)*np.max([np.max([0,charg_thres - SOC1])/(charg_thres - SOC1), 0]) <= Pb[1,:], Pb[1,:] <= Pbmax1*np.ones(T),\
                                 -Pbmax2*np.ones(T)*np.max([np.max([0,charg_thres - SOC2])/(charg_thres - SOC2), 0]) <= Pb[2,:], Pb[2,:] <= Pbmax2*np.ones(T),\
                                 -Pbmax3*np.ones(T)*np.max([np.max([0,charg_thres - SOC3])/(charg_thres - SOC3), 0]) <= Pb[3,:], Pb[3,:] <= Pbmax3*np.ones(T),\
                                 -Pbmax4*np.ones(T)*np.max([np.max([0,charg_thres - SOC4])/(charg_thres - SOC4), 0]) <= Pb[4,:], Pb[4,:] <= Pbmax4*np.ones(T),\
                                 -Pbmax5*np.ones(T)*np.max([np.max([0,charg_thres - SOC5])/(charg_thres - SOC5), 0]) <= Pb[5,:], Pb[5,:] <= Pbmax5*np.ones(T),\
                                 -Pbmax6*np.ones(T)*np.max([np.max([0,charg_thres - SOC6])/(charg_thres - SOC6), 0]) <= Pb[6,:], Pb[6,:] <= Pbmax6*np.ones(T),\
                                 -Pgmax0*np.ones(T) <= Pg, Pg <= Pgmax0*np.ones(T),\
                                 Pslkmin*np.ones(T) <= Pslk, Pslk <= Pslkmax*np.ones(T),\
                                 TotalDef.T + Pc.T == TotalGen + Pslk
                                 
                                 ])
        
            prob = cvx.Problem(cvx.Minimize(Cost),Constraints)
        
            prob.solve(solver = cvx.MOSEK)
                        
            if prob.status == 'optimal':
                # If the optimization worked
                converged = 1
                
                endTime = time.time() - tStart - InitTime
                endTime = np.ceil(endTime)
                endTime = np.intc(endTime)
                
                if endTime <= T:
                    Xr = X[:,endTime].value   # picking the values corresponding to the last/endTime, 
                    Pbsetpt = Pb[:,endTime].value  # unlike MPC which picks values based on the starting time instant
                    PVsetpt = PV[:,endTime]
                    Pgsetpt = Pg[endTime].value
                    Pslksetpt = Pslk[endTime].value 
                    
                else:
                    Xr = X[:,T-1].value # picking the values corresponding to the last/endTime, 
                    Pbsetpt = Pb[:,T-1].value  # unlike MPC which picks values based on the starting time instant
                    PVsetpt = PV[:,T-1]
                    Pgsetpt = Pg[T-1].value
                    Pslksetpt = Pslk[T-1].value
                              
                return Xr, -PVsetpt, -Pbsetpt, -Pgsetpt, -Pslksetpt, HoV_ON, converged
            else:
                converged = 0
                endTime = time.time() - tStart - InitTime
                endTime = np.ceil(endTime)
                endTime = np.intc(endTime)
                
                if endTime <= T:
                    PVsetpt = PV[:,endTime]
                    Pgsetpt = Pgmax0
                    Pbsetpt = np.array([Pbmax0, Pbmax1, Pbmax2, Pbmax3, Pbmax4, Pbmax5, Pbmax6])
                    Pslksetpt = -Vec[178]
                    
                    PGenTotal = np.sum(PVsetpt) + np.sum(Pbsetpt) + Pgsetpt + Pslksetpt
                    
                    LoadCosts = np.unique(CL)
                    sortedCost = np.sort(LoadCosts) # ascending order of cost
                    sortedCost = sortedCost[::-1] # descending order of cost
                    
                    LoadIndex = 0    
                    for i in range(len(LoadCosts)):
                        LoadIndex1 = np.argwhere(CL==LoadCosts[i])
                        LoadIndex = np.concatenate((LoadIndex1,LoadIndex),axis = None)                    
                    
                    LdPower = Pc[endTime]
                    loadfinal = 0
                    for l in range(len(LoadIndex)-1):
                        LdPower = LdPower + PLi[endTime,LoadIndex[l]]
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
                            Xr[LoadIndex[j]] = 1

                    ONindex = np.flatnonzero(Xr)
                    
                    PloadON = Pc[endTime] + np.sum(PLi[endTime,ONindex]) 
                    
                    PGenWithoutSlack = np.sum(PVsetpt) + np.sum(Pbsetpt) + Pgsetpt
                    Pslksetpt = PloadON -  PGenWithoutSlack  

                else:
                    PVsetpt = PV[:,T-1]
                    Pgsetpt = Pgmax0
                    Pbsetpt = np.array([Pbmax0, Pbmax1, Pbmax2, Pbmax3, Pbmax4, Pbmax5, Pbmax6])
                    
                    Pslksetpt = -Vec[178]
                    
                    PGenTotal = np.sum(PVsetpt) + np.sum(Pbsetpt) + Pgsetpt + Pslksetpt
        
                    LoadCosts = np.unique(CL)
                    sortedCost = np.sort(LoadCosts) # ascending order of cost
                    sortedCost = sortedCost[::-1] # descending order of cost

                    LoadIndex = 0    
                    for i in range(len(LoadCosts)):
                        LoadIndex1 = np.argwhere(CL==LoadCosts[i])
                        LoadIndex = np.concatenate((LoadIndex1,LoadIndex),axis = None)

                    LdPower = Pc[T-1]
                    loadfinal = 0
                    for l in range(len(LoadIndex)-1):
                        LdPower = LdPower + PLi[T-1,LoadIndex[l]]                        
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
                            Xr[LoadIndex[j]] = 1
                                                
                    ONindex = np.flatnonzero(Xr)
                    
                    PloadON = Pc[T-1] + np.sum(PLi[T-1,ONindex]) 
                    
                    PGenWithoutSlack = np.sum(PVsetpt) + np.sum(Pbsetpt) + Pgsetpt
                    Pslksetpt = PloadON -  PGenWithoutSlack 
                    
                return Xr, -PVsetpt, -Pbsetpt, -Pgsetpt, -Pslksetpt, HoV_ON, converged
                log.info('Did not work')
            

opt_sol_found = 0
Pbsetpt = np.zeros(7)
Pgsetpt = 0
Pslksetpt = 0
                      

def execHoVDispatch():
    """
    Execute the HoV algorithm and send out dispatch command vector
    """
    global Horizon 
    global opt_sol_found
    global Pbsetpt
    global Pgsetpt
    global Pslksetpt
    
    while threadGo:
       
        LoadStatus, PVsetpt, Pbsetpt, Pgsetpt, Pslksetpt, HoV_ON, opt_sol_found = HoVengine(measVec,PVslope,Horizon)  
        
        if HoV_ON == 1:
            
            if opt_sol_found == 1:                
                t2 = time.time() - tStart
                log.info("Optimal HoV dispatch:") 
                log.info("Dispatch time=" + str(t2))
                log.info("LoadStatus=" + str(LoadStatus))
                log.info("PVSetpt=" + str(PVsetpt))
                log.info("PbSetpt=" + str(Pbsetpt))
                log.info("PgSetpt=" + str(Pgsetpt))
                log.info("PSlack=" + str(Pslksetpt))
                # HoVcmd = np.concatenate((t2, LoadStatus, PVsetpt, Pbsetpt, Pgsetpt), axis=None) 
                HoVcmd = np.concatenate((LoadStatus, PVsetpt, Pbsetpt, Pgsetpt), axis=None)  
                sendOutHov(HoVcmd)
                # time.sleep(60)
            else:
                t2 = time.time() - tStart
                log.info("Did not find optimal solution")
                # log.info("results from HoV dispatch:") 
                # log.info("Dispatch time=" + str(t2))
                # log.info("LoadStatus=" + str(LoadStatus))
                # log.info("PVSetpt=" + str(PVsetpt))
                # log.info("PbSetpt=" + str(Pbsetpt))
                # log.info("PgSetpt=" + str(Pgsetpt))
                # log.info("PSlack=" + str(Pslksetpt))
                # time.sleep(45)
                # HoVcmd = np.concatenate((LoadStatus, PVsetpt, Pbsetpt, Pgsetpt), axis=None)   
                # sendOutHov(HoVcmd)


measVec = np.zeros(190)
          
def receiveMeasHoV():
    """
    Listen for measurements provided by the real-time simulated power system model
    Upon receipt of a measurement, update the local system state (stateTable) 
    """
 
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
    s.bind(LOCAL_UDP_ADDR) #IP:Port here is for the computer running this script
    log.info('Bound to ' + str(LOCAL_UDP_ADDR) + ' and waiting for UDP packet-based measurements')
    
    
    global measVec
    global flag
    # global opt_sol_found
    # global Pbsetpt
    # global Pgsetpt
    # global Pslksetpt
    
    
    while threadGo:
        #use UDP comms - receive packet with array
        data,addr = s.recvfrom(10000) #buffer size can be smaller if needed, but this works fine
        l = len(data)
        p = int(l/4) #get number of floats sent
        
        vector = np.array(struct.unpack('<{}'.format('f'*p),data))  
        log.debug(vector)
             
        if len(vector) == 189:                   
            # ldPV = vector[0:153] 
            
            # if opt_sol_found:
            #     # Send the updated 
            #     ContGenPmeas = np.array([Pbsetpt[0],Pbsetpt[1],Pbsetpt[2],Pbsetpt[3],Pbsetpt[4],Pbsetpt[5],Pbsetpt[6],Pgsetpt,Pslksetpt]) # 7 Batt, 1 Diesel, 1 Slack
            #     ContGenPmin = np.array([vector[154],vector[157],vector[160],vector[163],vector[166],vector[169],vector[172],vector[175],vector[178]]) # 7 Batt, 1 Diesel, 1 Slack
            #     ContGenPmax = np.array([vector[155],vector[158],vector[161],vector[164],vector[167],vector[170],vector[173],vector[176],vector[179]]) # 7 Batt, 1 Diesel, 1 Slack      
            # else:
            #     ContGenPmeas = np.array([vector[153],vector[156],vector[159],vector[162],vector[165],vector[168],vector[171],vector[174],vector[177]]) # 7 Batt, 1 Diesel, 1 Slack
            #     ContGenPmin = np.array([vector[154],vector[157],vector[160],vector[163],vector[166],vector[169],vector[172],vector[175],vector[178]]) # 7 Batt, 1 Diesel, 1 Slack
            #     ContGenPmax = np.array([vector[155],vector[158],vector[161],vector[164],vector[167],vector[170],vector[173],vector[176],vector[179]]) # 7 Batt, 1 Diesel, 1 Slack   
            
            # nlmVec = np.concatenate((ldPV,np.sum(ContGenPmeas),np.sum(ContGenPmin),np.sum(ContGenPmax),vector[180]),axis=None)
            
            dt = time.time() - tStart
            measVec = np.concatenate((dt,vector),axis=None)
            
            fsdVec = np.concatenate((dt,vector),axis=None)
            #if enabled, immediately relay this vector on to the NLM engine
            if fsdMeasRelay:
                p = len(fsdVec) #determine how many floats we are sending\n",
                msgBytes = struct.pack('<{}'.format('f'*p),*fsdVec.tolist()) #bytepack numpy row vector - use little Endian format (required by Opal-RT)\n",
                s.sendto(msgBytes, fsdMeasIP)
                # log.debug("Relayed measurement packet to fast time engine on " + str(fsdMeasIP))
                # log.debug(fsdVec)
        
        #log latest stateTable Pmeas to scenario log file
        with open(SCENARIO_LOG_FILENAME, "a") as f:
            a = measVec
            a = a.reshape((1,len(a)))
            np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')
        
        log.info("Received udp-based measurement packet")
        log.debug(str(measVec))


def sendOutHov(outVec):
    #REMOTE_HOST_ADDR = ('127.0.0.1', 7300) #IP:Port of remote host to send to
    #send via UDP
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
    p = len(outVec) #determine how many floats we are sending\n",
    msgBytes = struct.pack('<{}'.format('f'*p),*outVec.tolist()) #bytepack numpy row vector - use little Endian format (required by Opal-RT)\n",
    s.sendto(msgBytes, REMOTE_HOST_ADDR)

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loadprofile", help="filename of config file (XLSX format) to define this scenario", default='-1')
    ap.add_argument("--loadcost", help="filename of the file (XLSX format) with load cost and max power", default='-1')
    ap.add_argument("--pvpred", help="filename of scenario profile (XLSX format) defining the predicted PV profile for this scenario", default='-1')
    ap.add_argument("--debug", action="store_true", help="turn debug mode on, including additional logging and messages")
    ap.add_argument("--horizon", help="Time horizon for the HoV engine", default='-1')
    ap.add_argument("--localip", help="IPv4 address of this computer. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--localport", help="Port to receive UDP packets on for this computer. Default is 4100", default="4100")
    ap.add_argument("--remoteip", help="IPv4 address of the remote computer to send UDP-based commands to. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--remoteport", help="Port on remote computer to send UDP-based commands to. Default is 7300", default="7300")
    ap.add_argument("--log", help="filename of CSV log file to write latest system state and measurements to at each measurement update", default="-1")
    ap.add_argument("--fsdmeasport", help="port on machine hosting HoV code to receive UDP measurements relayed by this code (from net-load system simulation originally) on", default="5000")
    args = ap.parse_args()

    debugModeON = args.debug

    if debugModeON:
        log.setLevel(logging.DEBUG)
        log.info("DEBUG mode activated")
    else:
        log.setLevel(logging.INFO)
    
    if args.pvpred == '-1' or args.pvpred[-5:] != ".xlsx":
        log.error("ERROR: must define scenario config file in XLSX format")
        sys.exit()
    else:
        slopePVprofile = args.pvpred
   
    if args.loadprofile == '-1' or args.loadprofile[-5:] != ".xlsx":
        log.error("ERROR: must define load profile file in XLSX format")
        sys.exit()
    else:
        loadProfile = args.loadprofile
        
    if args.loadcost == '-1' or args.loadcost[-5:] != ".xlsx":
        log.error("ERROR: must define load profile file in XLSX format")
        sys.exit()
    else:
        LoadCost = args.loadcost
        
   
    Horizon = int(args.horizon)

    if Horizon != -1:
        if Horizon < 0:
            print("Error: horizon must be > 0")
            sys.exit()
   
    fsdMeasPort = int(args.fsdmeasport)
    if fsdMeasPort != -1:
        #check that UDP port specified is within range
        if fsdMeasPort > 100 and fsdMeasPort < 100000:
            fsdMeasRelay = True
            fsdMeasIP = (args.localip, fsdMeasPort)
        else:
            log.error("ERROR: NLM port must be between 100-100000")
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
    
    loadData = pd.read_excel(loadProfile, sheet_name='Sheet1')
    loadData = np.array(loadData)
    
    DefLoadCost = pd.read_excel(LoadCost, sheet_name='Sheet1')
    DefLoadCost = np.array(DefLoadCost)
            

    if args.log != "-1":
        if args.log[-4:] == ".csv":
            doLogScenario = True
            scenarioLogFile = args.log
        else:
            log.error("ERROR: If specifying log file, it must end in .csv")
            sys.exit()
    else:
        doLogScenario = False
    
    lockMeas = Lock() #lock to ensure no concurrent modification of stateTable by parallel threads

    #remove any existing log file that we appended to last time
    try:
        os.remove(SCENARIO_LOG_FILENAME)
    except FileNotFoundError as e:
        pass

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
    
    
