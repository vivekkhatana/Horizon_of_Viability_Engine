# -*- coding: utf-8 -*-
__version__ = "1.3"
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
    HoV_ON = 1 #### IMPORTANT change this for base case/optimized case###
        
    opt_sol_found = 0
    SOC1 = Vec[51]
    SOE1 = Vec[52]
    SOE2 = Vec[53]
    T = Horizon
    dt = 3600
    Eb_initial = 100000
    E_max1 = SOC1*Eb_initial*dt
    Eg_initial = np.array([370000,100000])
    Eg1 = SOE1*Eg_initial[0]*dt
    Eg2 = SOE2*Eg_initial[1]*dt

    
    CriticalLoad = Vec[35]
    PDefLoads = Vec[1:35]
    
    Nd = len(PDefLoads)
    Pc = CriticalLoad*np.ones(T)
    
    PLi = np.zeros((Nd, T))
    for i in range(T):
        PLi[:, i] = PDefLoads
    CL = np.array([13, 18, 18, 19, 16, 15, 15, 18, 19, 19, 13, 15, 18, 14, 16, 14, 16, 13, 17, 18, 13, 15, 16, 15, 13,\
                   17, 18, 19, 19, 19, 11, 12, 12, 20])

    
    ti = np.ceil(Vec[0])
    ti = np.intc(ti)
    
    slope =  np.cumprod(PVslope[ti:ti+T-1]) 

    PVinv_max = np.concatenate((-Vec[37],-Vec[37]*slope), axis=None)
    P_Rated1 = -Vec[42]
    P_Rated2 = -Vec[45]    
    Pb_max1  = -Vec[39]
    Pb_max2 = -Vec[48]
    
    
    if PVinv_max[0] == 0 and P_Rated1 == 0 and P_Rated2 == 0 and Pb_max1 == 0:
        HoV_ON = 0
        return np.zeros(Nd), 0, np.array([0,0]), 0, HoV_ON, 0

    if HoV_ON == 1:   
        if P_Rated1 > 0 and P_Rated2 > 0 and  Pb_max1 > 0:
            # Optimization problem formulation
            Pg = cvx.Variable((2,T))
            PVinv = cvx.Variable(T)
            Pb_d = cvx.Variable((2,T))
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function
            cost11 = cvx.minimum( (18.8/20)*(100*Pg[0,:]/P_Rated1 - 20) + 20,\
                                    (15.1/80)*(100*Pg[0,:]/P_Rated1 - 100) + 33.9 )
            cost12 = cvx.minimum( (18.8/20)*(100*Pg[1,:]/P_Rated2 - 20) + 20,\
                                    (15.1/80)*(100*Pg[1,:]/P_Rated2 - 100) + 33.9 )            


            dummy = np.ones((Nd,T))-X
            dummy1 = cvx.multiply(dummy,PLi)
            cost2 = CL@dummy1
                            
            cost3 = -Pb_d[0,:]
            
            cost4 = 10*(PVinv_max - PVinv)
               
            Cost = -np.ones(T)@cost11 - np.ones(T)@cost12 + np.ones(T)@cost2 + np.ones(T)@cost3 + np.ones(T)@cost4
        
            #Constraints
            Constraints = []
            
            dummy4 = cvx.multiply(X,PLi)
            dummy5 = cvx.sum(dummy4, axis=0) 
            dummy6 = PVinv + np.ones(2)@Pg + np.ones(2)@Pb_d
             
            Constraints.extend([ 
                                 0 <= cvx.sum(Pb_d[0,:]),  cvx.sum(Pb_d[0,:]) <= E_max1,\
                                 0 <= cvx.sum(Pg[0,:]),  cvx.sum(Pg[0,:]) <= Eg1,\
                                 0 <= cvx.sum(Pg[1,:]),  cvx.sum(Pg[1,:]) <= Eg2,\
                                 0 <= PVinv, PVinv <= PVinv_max,\
                                 np.zeros(T) <= Pg[0,:], Pg[0,:] <= P_Rated1*np.ones(T),\
                                 np.zeros(T) <= Pg[1,:], Pg[1,:] <= P_Rated2*np.ones(T),\
                                 np.zeros(T) <= Pb_d[0,:], Pb_d[0,:] <= Pb_max1*np.ones(T),\
                                 np.zeros(T) <= Pb_d[1,:], Pb_d[1,:] <= Pb_max2*np.ones(T),\
                                 dummy5.T + Pc.T == dummy6
                                 
                                 ])
        
            prob = cvx.Problem(cvx.Minimize(Cost),Constraints)
        
            prob.solve(solver = cvx.MOSEK)
                        
            if prob.status == 'optimal':
                # If the optimization worked
                # Return the first input
                opt_sol_found = 1
                
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    # Xr = X[:,0].value  
                    Xr = X[:,et].value
                    Pbsetpt = Pb_d[0,et].value
                    PVsetpt = PVinv[et].value
                    Pgsetpt = Pg[:,et].value
                    
                else:
                    Xr = X[:,T].value
                    Pbsetpt = Pb_d[0,T].value
                    PVsetpt = PVinv[T].value
                    Pgsetpt = Pg[:,T].value
                              
                return Xr, -PVsetpt, -Pbsetpt, -Pgsetpt, HoV_ON, opt_sol_found
            else:
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    PVsetpt = PVinv_max[et]
                else:
                    PVsetpt = PVinv_max[T]
                                    
                return np.zeros(Nd), -PVsetpt, -Pb_max1, np.array([-P_Rated1,-P_Rated2]), HoV_ON, opt_sol_found
                log.info('Did not work')
                
        elif P_Rated1 > 0 and P_Rated2 > 0 and Pb_max1 == 0:
            # Optimization problem formulation
            Pg = cvx.Variable((2,T))
            PVinv = cvx.Variable(T)
            Pb_d = cvx.Variable(T)
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function         
            cost11 = cvx.minimum( (18.8/20)*(100*Pg[0,:]/P_Rated1 - 20) + 20,\
                                  (15.1/80)*(100*Pg[0,:]/P_Rated1 - 100) + 33.9 )
            cost12 = cvx.minimum( (18.8/20)*(100*Pg[1,:]/P_Rated2 - 20) + 20,\
                                  (15.1/80)*(100*Pg[1,:]/P_Rated2 - 100) + 33.9 )
            
            dummy = np.ones((Nd,T))-X
            dummy1 = cvx.multiply(dummy,PLi)
            cost2 = CL@dummy1   
                         
            cost4 = 10*(PVinv_max - PVinv)
               
            Cost = -np.ones(T)@cost11 - np.ones(T)@cost12 + np.ones(T)@cost2 + np.ones(T)@cost4
        
        
            #Constraints
            Constraints = []
            
            dummy4 = cvx.multiply(X,PLi)
            dummy5 = cvx.sum(dummy4, axis=0) 
            dummy6 = PVinv + np.ones(2)@Pg + Pb_d 
             
            Constraints.extend([ 
                                 0 <= cvx.sum(Pg[0,:]),  cvx.sum(Pg[0,:]) <= Eg1,\
                                 0 <= cvx.sum(Pg[1,:]),  cvx.sum(Pg[1,:]) <= Eg2,\
                                 0 <= PVinv, PVinv <= PVinv_max,\
                                 np.zeros(T) <= Pg[0,:], Pg[0,:] <= P_Rated1*np.ones(T),\
                                 np.zeros(T) <= Pg[1,:], Pg[1,:] <= P_Rated2*np.ones(T),\
                                 np.zeros(T) <= Pb_d, Pb_d <= Pb_max2*np.ones(T),\
                                 dummy5.T + Pc.T == dummy6
                                 
                                 ])
        
            prob = cvx.Problem(cvx.Minimize(Cost),Constraints)
        
            prob.solve(solver = cvx.MOSEK)
            
                      
            if prob.status == 'optimal':
                # If the optimization worked
                # Return the first input
                opt_sol_found = 1
                
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    # Xr = X[:,0].value  
                    Xr = X[:,et].value
                    PVsetpt = PVinv[et].value
                    Pgsetpt = Pg[:,et].value
                    
                else:
                    Xr = X[:,T].value
                    PVsetpt = PVinv[T].value
                    Pgsetpt = Pg[:,T].value
                            
                return Xr, -PVsetpt, -Pb_max1, -Pgsetpt, HoV_ON, opt_sol_found
            else:
                # It didn't work so don't return anything
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    PVsetpt = PVinv_max[et]
                else:
                    PVsetpt = PVinv_max[T]
                
                return np.zeros(Nd), -PVsetpt, -Pb_max1, np.array([-P_Rated1,-P_Rated2]), HoV_ON, opt_sol_found
                log.info('Did not work')
                
        elif P_Rated1 == 0 and P_Rated2 > 0 and Pb_max1 == 0:
            # Optimization problem formulation
            Pg = cvx.Variable(T)
            PVinv = cvx.Variable(T)
            Pb_d = cvx.Variable(T)
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function
            cost12 = cvx.minimum( (18.8/20)*(100*Pg/P_Rated2 - 20) + 20,\
                                  (15.1/80)*(100*Pg/P_Rated2 - 100) + 33.9 )
            
            dummy = np.ones((Nd,T))-X
            dummy1 = cvx.multiply(dummy,PLi)
            cost2 = CL@dummy1
              
            
            cost4 = 10*(PVinv_max - PVinv)
               
            Cost = - np.ones(T)@cost12 + np.ones(T)@cost2 + np.ones(T)@cost4
        
        
            #Constraints
            Constraints = []
            
            dummy4 = cvx.multiply(X,PLi)
            dummy5 = cvx.sum(dummy4, axis=0) 
            dummy6 = PVinv + Pg + Pb_d 
             
            Constraints.extend([ 
                                 0 <= cvx.sum(Pg),  cvx.sum(Pg) <= Eg2,\
                                 0 <= PVinv, PVinv <= PVinv_max,\
                                 np.zeros(T) <= Pg, Pg <= P_Rated2*np.ones(T),\
                                 np.zeros(T) <= Pb_d, Pb_d <= Pb_max2*np.ones(T),\
                                 dummy5.T + Pc.T == dummy6
                                 
                                 ])
        
            prob = cvx.Problem(cvx.Minimize(Cost),Constraints)
        
            prob.solve(solver = cvx.MOSEK)
            
                    
           
            if prob.status == 'optimal':
                # If the optimization worked
                # Return the first input
                opt_sol_found = 1
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    # Xr = X[:,0].value  
                    Xr = X[:,et].value
                    Pbsetpt = Pb_d[0,et].value
                    PVsetpt = PVinv[et].value
                    Pgsetpt = Pg[et].value                    
                else:
                    Xr = X[:,T].value
                    Pbsetpt = Pb_d[0,T].value
                    PVsetpt = PVinv[T].value
                    Pgsetpt = Pg[T].value
                            
                
                return Xr, -PVsetpt, -Pb_max1, np.array([-P_Rated1, -Pgsetpt]), HoV_ON, opt_sol_found
            else:
                # It didn't work so don't return anything
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    PVsetpt = PVinv_max[et]                 
                else:
                    PVsetpt = PVinv_max[T]                           

                return np.zeros(Nd), -PVsetpt, -Pb_max1, np.array([-P_Rated1,-P_Rated2]), HoV_ON, opt_sol_found
                log.info('Did not work')
                
        elif P_Rated1 > 0 and P_Rated2 == 0 and  Pb_max1 > 0:
            # Optimization problem formulation
            Pg = cvx.Variable(T)
            PVinv = cvx.Variable(T)
            Pb_d = cvx.Variable((2,T))
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function
            cost11 = cvx.minimum( (18.8/20)*(100*Pg/P_Rated1 - 20) + 20,\
                                    (15.1/80)*(100*Pg/P_Rated1 - 100) + 33.9 )


            dummy = np.ones((Nd,T))-X
            dummy1 = cvx.multiply(dummy,PLi)
            cost2 = CL@dummy1
                            
            cost3 = -Pb_d[0,:]
            
            cost4 = 10*(PVinv_max - PVinv)
               
            Cost = -np.ones(T)@cost11 + np.ones(T)@cost2 + np.ones(T)@cost3 + np.ones(T)@cost4
        
            #Constraints
            Constraints = []
            
            dummy4 = cvx.multiply(X,PLi)
            dummy5 = cvx.sum(dummy4, axis=0) 
            dummy6 = PVinv + Pg + np.ones(2)@Pb_d
             
            Constraints.extend([ 
                                 0 <= cvx.sum(Pb_d[0,:]),  cvx.sum(Pb_d[0,:]) <= E_max1,\
                                 0 <= cvx.sum(Pg),  cvx.sum(Pg) <= Eg1,\
                                 0 <= PVinv, PVinv <= PVinv_max,\
                                 np.zeros(T) <= Pg, Pg <= P_Rated1*np.ones(T),\
                                 np.zeros(T) <= Pb_d[0,:], Pb_d[0,:] <= Pb_max1*np.ones(T),\
                                 np.zeros(T) <= Pb_d[1,:], Pb_d[1,:] <= Pb_max2*np.ones(T),\
                                 dummy5.T + Pc.T == dummy6
                                 
                                 ])
        
            prob = cvx.Problem(cvx.Minimize(Cost),Constraints)
        
            prob.solve(solver = cvx.MOSEK)
                        
            if prob.status == 'optimal':
                # If the optimization worked
                # Return the first input
                opt_sol_found = 1
                
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    # Xr = X[:,0].value  
                    Xr = X[:,et].value
                    Pbsetpt = Pb_d[0,et].value
                    PVsetpt = PVinv[et].value
                    Pgsetpt = Pg[et].value
                    
                else:
                    Xr = X[:,T].value
                    Pbsetpt = Pb_d[0,T].value
                    PVsetpt = PVinv[T].value
                    Pgsetpt = Pg[T].value
                
                return Xr, -PVsetpt, -Pbsetpt, np.array([-Pgsetpt, -P_Rated2]), HoV_ON, opt_sol_found
            else:
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                # It didn't work so don't return anything
                
                if et <= T:
                    PVsetpt = PVinv_max[et]                  
                else:
                    PVsetpt = PVinv_max[T]
                
                return np.zeros(Nd), -PVsetpt, -Pb_max1, np.array([-P_Rated1, -P_Rated2]), HoV_ON, opt_sol_found
                log.info('Did not work')
                
        elif P_Rated1 == 0 and P_Rated2 > 0 and  Pb_max1 > 0:
            # Optimization problem formulation
            Pg = cvx.Variable(T)
            PVinv = cvx.Variable(T)
            Pb_d = cvx.Variable((2,T))
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function
            cost12 = cvx.minimum( (18.8/20)*(100*Pg/P_Rated2 - 20) + 20,\
                                    (15.1/80)*(100*Pg/P_Rated2 - 100) + 33.9 )


            dummy = np.ones((Nd,T))-X
            dummy1 = cvx.multiply(dummy,PLi)
            cost2 = CL@dummy1
                            
            cost3 = -Pb_d[0,:]
            
            cost4 = 10*(PVinv_max - PVinv)
               
            Cost = -np.ones(T)@cost12 + np.ones(T)@cost2 + np.ones(T)@cost3 + np.ones(T)@cost4
        
            #Constraints
            Constraints = []
            
            dummy4 = cvx.multiply(X,PLi)
            dummy5 = cvx.sum(dummy4, axis=0) 
            dummy6 = PVinv + Pg + np.ones(2)@Pb_d
             
            Constraints.extend([ 
                                 0 <= cvx.sum(Pb_d[0,:]),  cvx.sum(Pb_d[0,:]) <= E_max1,\
                                 0 <= cvx.sum(Pg),  cvx.sum(Pg) <= Eg2,\
                                 0 <= PVinv, PVinv <= PVinv_max,\
                                 np.zeros(T) <= Pg, Pg <= P_Rated2*np.ones(T),\
                                 np.zeros(T) <= Pb_d[0,:], Pb_d[0,:] <= Pb_max1*np.ones(T),\
                                 np.zeros(T) <= Pb_d[1,:], Pb_d[1,:] <= Pb_max2*np.ones(T),\
                                 dummy5.T + Pc.T == dummy6
                                 
                                 ])
        
            prob = cvx.Problem(cvx.Minimize(Cost),Constraints)
        
            prob.solve(solver = cvx.MOSEK)
                        
            if prob.status == 'optimal':
                # If the optimization worked
                # Return the first input
                opt_sol_found = 1
                
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    # Xr = X[:,0].value  
                    Xr = X[:,et].value
                    Pbsetpt = Pb_d[0,et].value
                    PVsetpt = PVinv[et].value
                    Pgsetpt = Pg[et].value
                    
                else:
                    Xr = X[:,T].value
                    Pbsetpt = Pb_d[0,T].value
                    PVsetpt = PVinv[T].value
                    Pgsetpt = Pg[T].value
                               
                return Xr, -PVsetpt, -Pbsetpt, np.array([-P_Rated1, -Pgsetpt]), HoV_ON, opt_sol_found
            else:
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                # It didn't work so don't return anything
                
                if et <= T:
                    PVsetpt = PVinv_max[et]
                else:
                    PVsetpt = PVinv_max[T]
                
                return np.zeros(Nd), -PVsetpt, -Pb_max1, np.array([-P_Rated1,-P_Rated2]), HoV_ON, opt_sol_found
                log.info('Did not work')
                
        elif P_Rated1 == 0 and P_Rated2 == 0 and  Pb_max1 > 0:
            # Optimization problem formulation
            PVinv = cvx.Variable(T)
            Pb_d = cvx.Variable((2,T))
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function
        
            dummy = np.ones((Nd,T))-X
            dummy1 = cvx.multiply(dummy,PLi)
            cost2 = CL@dummy1
                            
            cost3 = -Pb_d[0,:]
            
            cost4 = 10*(PVinv_max - PVinv)
               
            Cost = np.ones(T)@cost2 + np.ones(T)@cost3 + np.ones(T)@cost4
        
            #Constraints
            Constraints = []
            
            dummy4 = cvx.multiply(X,PLi)
            dummy5 = cvx.sum(dummy4, axis=0) 
            dummy6 = PVinv + np.ones(2)@Pb_d
             
            Constraints.extend([ 
                                 0 <= cvx.sum(Pb_d[0,:]),  cvx.sum(Pb_d[0,:]) <= E_max1,\
                                 0 <= PVinv, PVinv <= PVinv_max,\
                                 np.zeros(T) <= Pb_d[0,:], Pb_d[0,:] <= Pb_max1*np.ones(T),\
                                 np.zeros(T) <= Pb_d[1,:], Pb_d[1,:] <= Pb_max2*np.ones(T),\
                                 dummy5.T + Pc.T == dummy6
                                 
                                 ])
        
            prob = cvx.Problem(cvx.Minimize(Cost),Constraints)
        
            prob.solve(solver = cvx.MOSEK)
                        
            if prob.status == 'optimal':
                # If the optimization worked
                # Return the first input
                opt_sol_found = 1
                
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    # Xr = X[:,0].value  
                    Xr = X[:,et].value
                    Pbsetpt = Pb_d[0,et].value
                    PVsetpt = PVinv[et].value
                    
                else:
                    Xr = X[:,T].value
                    Pbsetpt = Pb_d[0,T].value
                    PVsetpt = PVinv[T].value

                                
                return Xr, -PVsetpt, -Pbsetpt, np.array([-P_Rated1,-P_Rated2]), HoV_ON, opt_sol_found
            else:
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                # It didn't work so don't return anything
                
                if et <= T:
                    PVsetpt = PVinv_max[et]    
                else:
                    PVsetpt = PVinv_max[T]
                
                return np.zeros(Nd), -PVinv_max[et], -Pb_max1, np.array([-P_Rated1,-P_Rated2]), HoV_ON, opt_sol_found
                log.info('Did not work')
                
        elif P_Rated1 > 0 and P_Rated2 == 0 and Pb_max1 == 0:
            # Optimization problem formulation
            Pg = cvx.Variable(T)
            PVinv = cvx.Variable(T)
            Pb_d = cvx.Variable(T)
            X = cvx.Variable((Nd,T), boolean = True)
        
            # Cost function
            cost11 = cvx.minimum( (18.8/20)*(100*Pg/P_Rated1 - 20) + 20,\
                                  (15.1/80)*(100*Pg/P_Rated1 - 100) + 33.9 )
            
            dummy = np.ones((Nd,T))-X
            dummy1 = cvx.multiply(dummy,PLi)
            cost2 = CL@dummy1
              
            
            cost4 = 10*(PVinv_max - PVinv)
               
            Cost = - np.ones(T)@cost11 + np.ones(T)@cost2 + np.ones(T)@cost4
        
        
            #Constraints
            Constraints = []
            
            dummy4 = cvx.multiply(X,PLi)
            dummy5 = cvx.sum(dummy4, axis=0) 
            dummy6 = PVinv + Pg + Pb_d 
             
            Constraints.extend([ 
                                 0 <= cvx.sum(Pg),  cvx.sum(Pg) <= Eg1,\
                                 0 <= PVinv, PVinv <= PVinv_max,\
                                 np.zeros(T) <= Pg, Pg <= P_Rated1*np.ones(T),\
                                 np.zeros(T) <= Pb_d, Pb_d <= Pb_max2*np.ones(T),\
                                 dummy5.T + Pc.T == dummy6
                                 
                                 ])
        
            prob = cvx.Problem(cvx.Minimize(Cost),Constraints)
        
            prob.solve(solver = cvx.MOSEK)
            
                    
           
            if prob.status == 'optimal':
                # If the optimization worked
                # Return the first input
                opt_sol_found = 1
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    # Xr = X[:,0].value  
                    Xr = X[:,et].value
                    PVsetpt = PVinv[et].value
                    Pgsetpt = Pg[et].value
                    
                else:
                    Xr = X[:,T].value
                    PVsetpt = PVinv[T].value
                    Pgsetpt = Pg[T].value
                
                return Xr, -PVsetpt, -Pb_max1, np.array([-Pgsetpt, -P_Rated2]), HoV_ON, opt_sol_found
            else:
                # It didn't work so don't return anything
                et = time.time() - tStart - ti
                et = np.ceil(et)
                et = np.intc(et)
                
                if et <= T:
                    PVsetpt = PVinv_max[et]                   
                else:
                    PVsetpt = PVinv_max[T]
                
                return np.zeros(Nd), -PVsetpt, -Pb_max1, np.array([-P_Rated1, -P_Rated2]), HoV_ON, opt_sol_found
                log.info('Did not work')
                
    else:
        et = time.time() - tStart - ti
        et = np.ceil(et)
        et = np.intc(et)
        
        if et <= T:
            PVsetpt = PVinv_max[et]                   
        else:
            PVsetpt = PVinv_max[T]
        
        return np.zeros(Nd), -PVsetpt, -Pb_max1, np.array([-P_Rated1, -P_Rated2]), HoV_ON, opt_sol_found
        log.info('Did not work')
                      
           

def execHoVDispatch():
    """
    Execute the HoV algorithm and send out dispatch command vector
    """
    global Horizon 
    while threadGo:
       
        LoadStatus, PVsetpt, Pbsetpt, Pgsetpt, HoV_ON, converged = HoVengine(measVec,PVslope,Horizon)  
        
        if HoV_ON == 1:
            
            if converged == 1:                
                t2 = time.time() - tStart
                log.info("Optimal HoV dispatch:") 
                log.info("Dispatch time=" + str(t2))
                log.info("LoadStatus=" + str(LoadStatus))
                log.info("PVSetpt=" + str(PVsetpt))
                log.info("PbSetpt=" + str(Pbsetpt))
                log.info("PgSetpt=" + str(Pgsetpt))
                # HoVcmd = np.concatenate((t2, LoadStatus, PVsetpt, Pbsetpt, Pgsetpt), axis=None) 
                HoVcmd = np.concatenate((LoadStatus, PVsetpt, Pbsetpt, Pgsetpt), axis=None)  
                sendOutHov(HoVcmd)
            else:
                t2 = time.time() - tStart
                log.info("Did not find optimal solution")
                log.info("results from HoV dispatch:") 
                log.info("Dispatch time=" + str(t2))
                log.info("LoadStatus=" + str(LoadStatus))
                log.info("PVSetpt=" + str(PVsetpt))
                log.info("PbSetpt=" + str(Pbsetpt))
                log.info("PgSetpt=" + str(Pgsetpt))
                #HoVcmd = np.concatenate((LoadStatus, PVsetpt, Pbsetpt, Pgsetpt), axis=None)   
                #sendOutHov(HoVcmd)

          
measVec = np.zeros(54)

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
    
    while threadGo:
        #use UDP comms - receive packet with array
        data,addr = s.recvfrom(10000) #buffer size can be smaller if needed, but this works fine
        l = len(data)
        p = int(l/4) #get number of floats sent

        vector = np.array(struct.unpack('<{}'.format('f'*p),data))
        if len(vector) == 53:         
            dt = time.time() - tStart
            measVec = np.concatenate((dt,vector),axis=None)
        
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
    # ap.add_argument("--config", help="filename of config file (XLSX format) to define this scenario", default='-1')
    ap.add_argument("--pvpred", help="filename of scenario profile (XLSX format) defining the predicted PV profile for this scenario", default='-1')
    ap.add_argument("--debug", action="store_true", help="turn debug mode on, including additional logging and messages")
    ap.add_argument("--horizon", help="Time horizon for the HoV engine", default='-1')
    ap.add_argument("--localip", help="IPv4 address of this computer. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--localport", help="Port to receive UDP packets on for this computer. Default is 4100", default="4100")
    ap.add_argument("--remoteip", help="IPv4 address of the remote computer to send UDP-based commands to. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--remoteport", help="Port on remote computer to send UDP-based commands to. Default is 7300", default="7300")
    ap.add_argument("--log", help="filename of CSV log file to write latest system state and measurements to at each measurement update", default="-1")
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
    
    ## do infinite while to keep main alive so that logging from threads shows in console
    try:
        while True:
            time.sleep(0.001)
    except KeyboardInterrupt:
        log.info("cleaning up threads and exiting...")
        threadGo = False
        log.info("done.")
    
    
