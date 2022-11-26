# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:11:05 2021 
Centralized Load Management engine, generates load ON-OFF decisions for controllable loads. 
Set-points for the diesel generators and Battery inverters and PV inveters are provided by a generation scheduler like HoV engine.

""" 
__version__ = "1.1"
__author__ = " Vivek Khatana, UMN"

import logging
import time
import pandas as pd
import numpy as np
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


def listenDCGMeas():
    """
    Listen for measurements provided by the device control gateway (DCG) or via real-time simulated power system model
    Upon receipt of a measurement, update the local measurement vector and run the auto dispatch function if enabled
    """
    
    global measVec
    global scenarioLogFile


    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
    s.bind(LOCAL_UDP_ADDR) #IP:Port here is for the computer running this script
    log.info('Load Management engine listen bound to ' + str(LOCAL_UDP_ADDR) + ' and waiting for UDP packet-based measurements')
    
    while threadGo:
                #use UDP comms - receive packet with array
        data,addr = s.recvfrom(10000) #buffer size can be smaller if needed, but this works fine
        l = len(data)
        p = int(l/4) #get number of floats sent

        measVec = np.array(struct.unpack('<{}'.format('f'*p),data))
        log.info("Received udp-based measurement packet")
        log.debug(str(measVec))
        
        autoDispatch()

        
       
def listenHovUDP(hovIP):
    """
    Listen for commands provided by the horizon of viability (HoV) or other equivalent generation dispatch engine. Upon receipt simply update the 
    global hovCmdVec variable that is checked periodically by the autodispatch function
    """

    global threadGo
    global Ng 
    global Nb 
    global Nd
    global hovCmdVec
    global hovCmdReady
    global hov_command_started

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
    s.bind(hovIP) #IP:Port here is for the computer running this script
    log.info('HoV listen bound to ' + str(hovIP) + ' and waiting for UDP packet-based commands')
    
    while threadGo:
        data,addr = s.recvfrom(10000) #buffer size can be smaller if needed, but this works fine
        l = len(data)
        p = int(l/4) #get number of floats sent

        hovCmdVec = np.array(struct.unpack('<{}'.format('f'*p),data)).astype('int')
        log.info("FROM HoV: Received udp-based command packet")
        log.debug(str(hovCmdVec))
    
        ##check if received command vector is of expected size based on scenario configuration, if not show error
        if len(hovCmdVec) != (Nd + Nb + Ng):
            log.error("ERROR: Received HoV command vector of length " + str(len(hovCmdVec)) + ", but expected size " + str(Nd + Nb + Ng) + " based on scenario configuration.")
            hovCmdReady = False
        else:
            hovCmdReady = True
            
        # # if hov logging is enabled, record what we received
        # if hovLogON:
        #     with open(hovLogFilename, "a") as f:
        #         dt = time.time() - tSt
        #         a = np.hstack((np.array([dt,0,int(cmdVecIsFeasible(hovCmdVec)),-1,-1]), hovCmdVec.T))
        #         a = a.reshape((1,len(a))) #format a as a row vector for writing to file
        #         np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')
        

def execHoVDispatch(dispCmd):
    """
    Send out hovCmdVec directly as our command vector
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets    
    dispatch_vec = dispCmd    
    p = len(dispatch_vec) #determine how many floats we are sending\n",
    msgBytes = struct.pack('<{}'.format('f'*p),*dispatch_vec.tolist()) #bytepack numpy row vector - use little Endian format (required by Opal-RT)\n",
    s.sendto(msgBytes, REMOTE_HOST_ADDR)
    log.debug("sent Dispatch command")
    log.debug(dispatch_vec)

    #if enabled, log raw udp packet sent to file
    if doLogUDP:
        with open(udpLogFile, "a") as f:
            dt = time.time() #add timestamp of current time
            a = np.hstack((np.array(dt), 2, dispatch_vec.T)) #0=UDP packet received, 1=sent, 2=sentHov
            a = a.reshape((1,len(a))) #format a as a row vector for writing to file
            np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')


def cmdVecIsFeasible(cmdVec):
    
    global measVec
    global Ng 
    global Nb 
    global Npv
    global Nd
    global INCLUDE_TYPE4
    
    #first check to see if cmdVec has been updated (started receiving HoV inputs). If it is len=0, then we know an hovCmdVec hasn't been received yet
    #so just return    
    if len(cmdVec) == 0:
        return False, 0
    
    dT = np.ceil(measVec[0])
    dT = np.intc(dT)
    
    # Controllable/deferable loads instantaneous power demand
    PLi = loadData[dT,1:Nd + 1].T
    
    defLoadHoVON = cmdVec[0:Nd]
    defloadHoVProfile = PLi # Load profiles of all the potential loads that can be turned ON
    defloadPower = np.multiply(defLoadHoVON,defloadHoVProfile) # load power turned ON by the HoV engine
    defLoad = np.sum(defloadPower) # Total load power turned ON by the HoV engine
    critLoad= measVec[Nd+1]
   
   
    ##check if hov command vector is feasible 
    # for loads verify that command vector only has 0/1 values
    if np.sum(defLoadHoVON) < 0 or np.sum(defLoadHoVON) > Nd:
        log.error("HoV status: command given for loads is not consistent with 0/1 structure. HoV cmd infeasible.")
        return False, 0
    
    ## for generation, first verify that individual gen set points are feasible (within their respective generator's capacity limits)
    # check Battery inverter and Diesel generator set-points. PV inverters running in MPPT mode 
    numType4 = Nb + Ng
    if INCLUDE_TYPE4:
        for ii in range(numType4):
            cmd = int(cmdVec[Nd+ii])
            if cmd < measVec[Nd + 1 + Npv*2 + 1 + (3*ii+1)] or cmd > measVec[Nd + 1 + Npv*2 + 1 + (3*ii+2)]:
                log.error("HoV status: command (" + str(cmd) + ") given for Battery or Diesel gen id=" + str(ii+1) + " is out of gen capacity limits. HoV cmd infeasible.")
                return False, 0
                
    ## gen cmds are all individually feasible, so get aggregate command now
    totalGen = np.sum(cmdVec[Nd:Nd+Nb+Ng])
    
    # available PV power
    availablePVpower = np.zeros(Npv)
    for ii in range(Npv):
        availablePVpower[ii] = measVec[Nd + 1 + 1 + 2*ii]
    
    totalPV = np.sum(availablePVpower)
    
    ## calculate resulting slack bus power
    slackPower = -(defLoad + critLoad + totalGen + totalPV) #defLoad and critLoad are positive, totalGen and totalPV are negative, slackPower is generation so should use negative sign convention for power that must be supplied
    
    # log.info(slackPower)
     
    log.debug("HoV: defLoad=" + str(defLoad) + ", critLoad=" + str(critLoad) + ", totGen=" + str(totalGen + totalPV) + ", slackPower=" + str(slackPower))
    
    # Slack bus power max and min limits    
    slackPmin = measVec[Nd + 1 + Npv*2 + Nb*3 + Ng*3 + 1 + 1]
    slackPmax = measVec[Nd + 1 + Npv*2 + Nb*3 + Ng*3 + 1 + 2]

    ## finally, check if slackPower is within the power capacity limits for the slack bus to determine if overall cmdVec is feasible
    if slackPower < slackPmin or slackPower > slackPmax:
        log.info("HoV status: command vector given is infeasible: Slack bus power out of limits")
        return False, slackPower
    else:
        log.info("HoV status: command vector is feasible")
        return True, slackPower
    


def autoDispatch():
    """
    Look at current power system operating condition for power balance and determine if a dispatch is needed to ensure: 
      TotalPgen + Pslack = TotalPload 
    is maintained. 
    This function runs one such "auto dispatch" action and is called periodically each time a measurement update is received (from listenDCGMeas)

    """
    
    global measVec
    global Ng 
    global Nb 
    global Npv
    global Nd
    global INCLUDE_TYPE4
    global hovCmdVec
    global hovInputON
    global hovCmdReady  
    global hov_command_started
    global lastDispPoint
    global lastDispPointhov

 
    # check if hov command vector is feasible
    hovCmdFeasible, hovSlackPower = cmdVecIsFeasible(hovCmdVec) 
    
    # check the state of the system to determine if autoDispatch is needed when HoV is not ready/infeasible    
    defLoad = np.sum(measVec[1:Nd+1])
    critLoad = measVec[Nd+1]
    
    # available PV power
    PVpower = np.zeros(Npv)
    for ii in range(Npv):
        PVpower[ii] = measVec[Nd + 1 + 1 + 2*ii]
    
       
    BattPower = np.zeros(Nb)
    for ii in range(Nb):
        BattPower[ii] = measVec[Nd + 1 + Npv*2 + 1 + 3*ii]
    
    dieselGenPower = np.zeros(Ng)
    for ii in range(Ng):
        dieselGenPower[ii] = measVec[Nd + 1 + Npv*2 + Nb*3 + 1 + 3*ii]
    
    totavailableGen = np.sum(PVpower) + np.sum(BattPower) + np.sum(dieselGenPower) # note that totavailableGen is negative

    # calculate slack bus power
    slackPower = -(defLoad + critLoad + totavailableGen) #defLoad and critLoad are positive, totGen is negative, slackPower is generation so should use negative sign convention for power that must be supplied
    # log.info(slackPower)
    
    # Slack bus power max and min limits    
    slackPmin = measVec[Nd + 1 + Npv*2 + Nb*3 + Ng*3 + 1 + 1]
    slackPmax = measVec[Nd + 1 + Npv*2 + Nb*3 + Ng*3 + 1 + 2]
           
    if hovInputON and hovCmdFeasible and hovCmdReady:
        #do dispatch and turn off hovCmdReady
        log.info("Dispatch: HoV command feasible and ready. Executing.")
        execHoVDispatch(hovCmdVec)
        lastDispPointhov = hovSlackPower
        hovCmdReady = False
    elif hovInputON and hovCmdFeasible and not hovCmdReady:
        dispTHhov = 2000 #change in dispatch set point threshold that will trigger another dispatch command being sent (i.e., if first command didn't cause a change of more than this then no further dispatch)
        if np.abs(lastDispPointhov - hovSlackPower) > dispTHhov:
            log.info("Earlier HoV dispatch command is feasible and produces significant net-load change. Executing.")
            execHoVDispatch(hovCmdVec)
            lastDispPointhov = hovSlackPower
        else:
            log.info(">>Earlier HoV dispatch command is feasible but not repeating the same HoV dispatch")    
        #if hov logging is enabled, record what we received
        # if hovLogON:
        #     with open(hovLogFilename, "a") as f:
        #         dt = time.time() - tSt
        #         a = np.hstack((np.array([dt,1,totPmeasNoSlack,slackPmin,-1]), hovCmdVec.T))
        #         a = a.reshape((1,len(a))) #format a as a row vector for writing to file
        #         np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')
    elif hovInputON and not hovCmdFeasible and (slackPower > slackPmin) and (slackPower < slackPmax):
        #in hov mode, but command is not ready and/or not feasible. However, system is viable as is, so do nothing
        log.info("No dispatch: HoV command not ready, but system is viable.")
        #if hov logging is enabled, record what we received
        # if hovLogON:
        #     with open(hovLogFilename, "a") as f:
        #         dt = time.time() - tSt
        #         a = np.array([dt,9,totPmeasNoSlack,slackPmin,-1])
        #         a = a.reshape((1,len(a))) #format a as a row vector for writing to file
        #         np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')
        
    if hovInputON and (slackPower < slackPmin): 
        #in hov mode, hov vec not ready/feasible and system is not viable so run local dispatch
        log.debug("autoDispatch: defLoad=" + str(defLoad) + ", critLoad=" + str(critLoad) + ", totGen=" + str(totavailableGen) + ", slackPower=" + str(slackPower))
        log.info("Dispatch: starting no-HoV auto dispatch: Curtailing loads")
        
        dispTH = 2000 #change in dispatch set point threshold that will trigger another dispatch command being sent (i.e., if first command didn't cause a change of more than this then no further dispatch)
        
        PVsetpt = -PVpower
        Pgsetpt = -dieselGenPower
        Pbsetpt = -BattPower
        Pslksetpt = -0.999*slackPmin # 0.99 is artificial; added to not stress the Slack bus and can be customized to the actual scenario
        
        PGenTotal = np.sum(PVsetpt) + np.sum(Pbsetpt) + np.sum(Pgsetpt) + Pslksetpt # made positive for ease of comaprison with total load power
        
        CL = ConfigData[0:Nd,3]
        
        LoadCosts = np.unique(CL) # find cost vector with cost entries so that their multiplicity is equal to one
        sortedCost = np.sort(LoadCosts) # ascending order of cost
        sortedCost = sortedCost[::-1] # descending order of cost
        
        LoadIndex = 0    
        for i in range(len(LoadCosts)):
            LoadIndex1 = np.argwhere(CL==sortedCost[i]) 
            LoadIndex = np.concatenate((LoadIndex1,LoadIndex),axis = None)  # Array with load indices of maximum cost at the beginning
        
        LdPower = critLoad        
        dT = np.ceil(measVec[0])
        dT = np.intc(dT)
        
        # Controllable/deferable loads instantaneous power demand
        PLi = loadData[dT,1:Nd + 1].T
     
        loadfinal = 0 # minimum number of highest priority loads that led to power balance
        for l in range(len(LoadIndex)-1): # run until len(LoadIndex)-1 because last entry of LoadIndex is equal to 0 which is a junk entry
            LdPower = LdPower + PLi[LoadIndex[l]]
            if LdPower == PGenTotal:
                loadfinal = l 
                break
            elif LdPower > PGenTotal:
                loadfinal = l - 1
                break

        Xr = np.zeros(Nd)
        for j in range(loadfinal):
            Xr[LoadIndex[j]] = 1 # turn ON loadfinal number of loads with highest cost/priority

        ONindex = np.flatnonzero(Xr) # indices of loads that are decided to be turned ON

        PloadON = critLoad + np.sum(PLi[ONindex]) # total load power ON
        
        PGenWithoutSlack = np.sum(PVsetpt) + np.sum(Pbsetpt) + np.sum(Pgsetpt)
        Pslksetpt = PloadON -  PGenWithoutSlack
        
        if np.abs(lastDispPoint - Pslksetpt) > dispTH:
            cmdVec = np.concatenate((Xr, -Pbsetpt, -Pgsetpt),axis = None)
            log.debug("autoDispatch: defLoad=" + str(np.sum(PLi[ONindex])) + ", critLoad=" + str(critLoad) + ", totGen=" + str(-PGenWithoutSlack) + ", slackPower=" + str(Pslksetpt))
            execHoVDispatch(cmdVec)
            lastDispPoint = Pslksetpt
        else:
            log.info(">>Outside of desired operation window, but not repeating the same dispatch")
    elif hovInputON and (slackPower > slackPmax):
        # in hov mode, hov vec not ready/feasible and system is not viable so run local dispatch
        log.debug("autoDispatch: defLoad=" + str(defLoad) + ", critLoad=" + str(critLoad) + ", totGen=" + str(totavailableGen) + ", slackPower=" + str(slackPower))
        log.info("Dispatch: starting no-HoV auto dispatch: Turning additional loads ON")
        
        dispTH = 2000 #change in dispatch set point threshold that will trigger another dispatch command being sent (i.e., if first command didn't cause a change of more than this then no further dispatch)
        
        PVsetpt = -PVpower
        Pgsetpt = -dieselGenPower
        Pbsetpt = -BattPower
        Pslksetpt = 0.999*slackPmax # 0.99 is artificial; added to not stress the Slack bus and can be customized to the actual scenario
        
        PGenTotal = np.sum(PVsetpt) + np.sum(Pbsetpt) + np.sum(Pgsetpt) + Pslksetpt # made positive for ease of comaprison with total load power
        
        CL = ConfigData[0:Nd,3]
        
        LoadCosts = np.unique(CL) # find cost vector with cost entries so that their multiplicity is equal to one
        sortedCost = np.sort(LoadCosts) # ascending order of cost
        sortedCost = sortedCost[::-1] # descending order of cost
        
        LoadIndex = 0    
        for i in range(len(LoadCosts)):
            LoadIndex1 = np.argwhere(CL==sortedCost[i])
            LoadIndex = np.concatenate((LoadIndex1,LoadIndex),axis = None) # Array with load indices of maximum cost at the beginning                    
        
        LdPower = critLoad
        dT = np.ceil(measVec[0])
        dT = np.intc(dT)
        
        # Controllable/deferable loads instantaneous power demand
        PLi = loadData[dT,1:Nd + 1].T
        
        loadfinal = 0 # minimum number of highest priority loads that led to power balance
        for l in range(len(LoadIndex)-1): # run until len(LoadIndex)-1 because last entry of LoadIndex is equal to 0 which is a junk entry
            LdPower = LdPower + PLi[LoadIndex[l]]
            if LdPower == PGenTotal:
                loadfinal = l
                break
            elif LdPower > PGenTotal:
                loadfinal = l - 1
                break

        Xr = np.zeros(Nd)
        for j in range(loadfinal):
            Xr[LoadIndex[j]] = 1 # turn ON loadfinal number of loads with highest cost/priority

        ONindex = np.flatnonzero(Xr) # indices of loads that are decided to be turned ON
        
        PloadON = critLoad + np.sum(PLi[ONindex])  # total load power ON
        
        PGenWithoutSlack = np.sum(PVsetpt) + np.sum(Pbsetpt) + np.sum(Pgsetpt)
        Pslksetpt = PloadON -  PGenWithoutSlack
                
        if np.abs(lastDispPoint - Pslksetpt) > dispTH:
            cmdVec = np.concatenate((Xr, -PVsetpt, -Pbsetpt, -Pgsetpt),axis = None)
            log.debug("autoDispatch: defLoad=" + str(np.sum(PLi[ONindex])) + ", critLoad=" + str(critLoad) + ", totGen=" + str(-PGenWithoutSlack) + ", slackPower=" + str(Pslksetpt))
            execHoVDispatch(cmdVec)
            lastDispPoint = Pslksetpt
        else:
            log.info(">>Outside of desired operation window, but not repeating the same dispatch")


def initializeViaConfigFile(filename):
    """
    Loads scenario config file and runs some checks to ensure all data is correct. Initializes stateTable using data in the config file

    -- Returns --
    True if initialization was successful, False if an error occurred
    """
    
    global ConfigData 
    global Ng 
    global Nb 
    global Npv
    global Nd
    global hovCmdVec
    global INCLUDE_TYPE4
    
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

    hovCmdVec = -np.ones(Nd + Nb + Ng)
    
    #determine if any type 4 generators are in this configuration and adjust type4 flag accordingly
    if (Ng + Nb) == 0:
        #no type 4 units defined
        INCLUDE_TYPE4 = False
    else:
        INCLUDE_TYPE4 = True

    log.info("Successfully initialized for a system with " + str(Nd) + " Controllable loads, " + str(Npv) + " PV invs, and " + str(Ng+Nb) + " Batt invs/ Diesel Gens.")

    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="filename of config file (XLSX format) to define this scenario", default='-1')
    ap.add_argument("--loadprofile", help="filename of load profiles (XLSX format)", default='-1')
    ap.add_argument("--debug", action="store_true", help="turn debug mode on, including additional logging and messages")
    ap.add_argument("--localip", help="IPv4 address of this computer. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--localport", help="Port to receive UDP packets on for this computer. Default is 7100", default="7100")
    ap.add_argument("--remoteip", help="IPv4 address of the remote computer to send UDP-based commands to. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--remoteport", help="Port on remote computer to send UDP-based commands to. Default is 4000", default="4000")
    ap.add_argument("--log", help="filename of CSV log file to write latest system state and measurements to at each measurement update", default="-1")
    ap.add_argument("--logudp", help="filename of CSV log file to write raw udp meassages (in and out) to at each UDP message send or receive event", default="-1")
    ap.add_argument("--hovcmdport", help="turn HoV input mode on and receive UDP dispatch from HoV here on the port specified", default="-1")
    ap.add_argument("--hovlog", help="turn on logging of HoV-related dispatch, including HoV cmd's received, dispatched, etc.", default="-1")
    args = ap.parse_args()

    debugModeON = args.debug

    if debugModeON:
        log.setLevel(logging.DEBUG)
        log.info("DEBUG mode activated")
    else:
        log.setLevel(logging.INFO)

    if args.config == '-1' or args.config[-5:] != ".xlsx":
        log.error("ERROR: must define scenario config file in XLSX format")
        sys.exit()
    else:
        scenarioConfigFile = args.config
        
       
    if args.loadprofile == '-1' or args.loadprofile[-5:] != ".xlsx":
        log.error("ERROR: must define load profile file in XLSX format")
        sys.exit()
    else:
        loadProfile = args.loadprofile 
       
    udpPortLocal = int(args.localport)
    udpPortRemote = int(args.remoteport)

    if udpPortLocal < 100 or udpPortLocal > 100000 or udpPortRemote < 100 or udpPortRemote > 100000:
        log.error("ERROR: both local and remote ports must be in range 100-100000")
        sys.exit()

    LOCAL_UDP_ADDR = (args.localip, udpPortLocal) #IP address and port to listen for measurements on for computer running this script
    REMOTE_HOST_ADDR = (args.remoteip, udpPortRemote) #IP:Port of remote host to send cmds to

    if args.log != "-1":
        if args.log[-4:] == ".csv":
            doLogScenario = True
            scenarioLogFile = args.log
        else:
            log.error("ERROR: If specifying log file, it must end in .csv")
            sys.exit()
    else:
        doLogScenario = False

    if args.logudp != "-1":
        if args.logudp[-4:] == ".csv":
            doLogUDP = True
            udpLogFile = args.logudp
        else:
            log.error("ERROR: If specifying udplog file, it must end in .csv")
            sys.exit()
    else:
        doLogUDP = False
    
    
    loadData = pd.read_excel(loadProfile, sheet_name='Sheet1')
    loadData = np.array(loadData)

    hovInputON = False
    hovCmdVec = np.array([])
    hovCmdReady = False
    hovPort = int(args.hovcmdport)
    if hovPort != -1:
        #check that UDP port specified is within range
        if hovPort > 100 and hovPort < 100000:
            hovInputON = True
            hovIP = (args.localip, hovPort)
        else:
            log.error("ERROR: HoV port must be between 100-100000")
            sys.exit()


    hovLogON = False
    if args.hovlog != "-1":
        if args.hovlog[-4:] == ".csv":
            hovLogON = True
            hovLogFilename = args.hovlog
        else:
            log.error("ERROR: Hov log filename must be a csv file")
            sys.exit()

    lastDispPoint = -1 #last dispatch point - used for auto dispatch
    lastDispPointhov = -1

    #initialize stateTable based on the scenario definition file
    INCLUDE_TYPE4 = False #include type 4 generation in dispatch and control
    initSuccess = initializeViaConfigFile(scenarioConfigFile)

   
    ## setup threads
    # setup var for threads
    threadGo = True #set to false to quit threads
    tSt = time.time()
    
    # startup thread to listen for measurements from DCG
    thread1 = Thread(target=listenDCGMeas)
    thread1.daemon = True
    thread1.start()
    
    # # startup thread to listen for dispatch init / config
    # thread2 = Thread(target=listenDispatchInitConfig)
    # thread2.daemon = True
    # thread2.start()

    # startup thread to listen for HoV input commands if appropriate
    if hovInputON:
        thread2 = Thread(target=listenHovUDP, args=(hovIP,))
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