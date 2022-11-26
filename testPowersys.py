# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 22:51:43 2021

@author: khata010
"""

"""
Example code to test the HoV and Load Management engine. This code send out via UDP each row of
PROFILE_FILENAME as individual UDP packets at the time index of each row listed.
The first column of each row in PROFILE_FILENAME must be a time index in seconds. 
This script then keeps track of time locally and sends out the row in PROFILE_FILENAME
as close as possible to the time index listed. We load the profile in and then begin
the local clock at t=0 after loaded and ready to go.

"""
__author__ = " Vivek Khatana, UMN"


import socket
import logging
import numpy as np
np.set_printoptions(suppress=True,edgeitems=10)
import pandas as pd
from threading import Thread
import struct
import time
import argparse
import sys
import os

### LOG FILE LOCATIONS ###
SCENARIO_LOG_FILENAME = "PowerSystemState.csv" #log file where

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
    

def sendMeasVec():
     
    global SOC 
    global SOE
    global Ng 
    global Nb 
    global Npv
    global Nc
    global Nd
    global hovCmdVec
    
    #periodically send out row of scenario profile as a UDP packet until end of time
    #vector in scenario profile is reached
    
    while threadGo:
        
        for i in range(0, len(scenProfile)):            
            tProfile = scenProfile['t'].iloc[i]
            dt = time.time() - tStart #calculate current relative time in sec
            tSleep = tProfile - dt
        
            if tSleep > 0:
                #tProfile hasn't been reached yet, sleep until it will arrive
                time.sleep(tSleep)
            
            dt = time.time() - tStart #recalculate current relative time in sec
            
            #get row of scenario profile to send based on current index
            profVec = np.array(scenProfile.iloc[i])     
            
            profVec = profVec[1:]
            
            profVec[0:Nd] = np.multiply(hovCmdVec[0:Nd],profVec[0:Nd])

            for i in range(Nb):
                profVec[Nd + 2*Npv + 3*i + 1] = hovCmdVec[Nd+i]
            
            for i in range(Ng):
                profVec[Nd + 2*Npv + 3*Nb + 3*i + 1] = hovCmdVec[Nd + Nb +i]
    
            measVec = np.concatenate((profVec, SOC, SOE), axis=None)
                    
            #send out the cmdVec
            # Open UDP socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            p = len(measVec) #determine how many floats we are sending
            msgBytes = struct.pack('<{}'.format('f'*p),*measVec.tolist()) #bytepack numpy row vector - use little Endian format (required by Opal-RT)
            s.sendto(msgBytes, REMOTE_HOST_ADDR) #send UDP packet
            
            to_log = np.concatenate((dt,measVec),axis=None)
            #log latest stateTable Pmeas to scenario log file
            with open(SCENARIO_LOG_FILENAME, "a") as f:
                a = to_log
                a = a.reshape((1,len(a)))
                np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')
            
            # print("(t=" + str(dt) + ") sent ", measVec)
            log.info("Sent new measurement with SOC" + str( np.concatenate((SOC,SOE),axis=None)) )
            
            

def listenHovCmd():
    """
    Listen for commands provided by the HoV engine (or other net-load dispatch engine).
    Upon receipt of a command, update the local measurement vector
    """
    
    global threadGo
    global Ng 
    global Nb 
    global Nd
    global SOC
    global hovCmdVec
    global scenarioLogFile


    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
    s.bind(LOCAL_UDP_ADDR) #IP:Port here is for the computer running this script
    log.info('Power system model listen bound to ' + str(LOCAL_UDP_ADDR) + ' and waiting for UDP packet-based commands')
    
    while threadGo:
                #use UDP comms - receive packet with array
        data,addr = s.recvfrom(10000) #buffer size can be smaller if needed, but this works fine
        l = len(data)
        p = int(l/4) #get number of floats sent

        hovCmdVec = np.array(struct.unpack('<{}'.format('f'*p),data))
        
        ##check if received command vector is of expected size based on scenario configuration, if not show error
        if len(hovCmdVec) != (Nd + Nb + Ng):
            log.error("ERROR: Received HoV command vector of length " + str(len(hovCmdVec)) + ", but expected size " + str(Nd + Nb + Ng) + " based on scenario configuration.")
        else:
            log.info("Received udp-based measurement packet")
            # log.debug(str(hovCmdVec))
          

def getSOC():
    
    global hovCmdVec
    global Ng 
    global Nb 
    global Npv
    global Nc
    global Nd
    global SOC 
    global SOE
    
    while threadGo:  
        
        cmd = hovCmdVec
        nontype4 = Nd+Nc+Npv
        
        Eb_initial = ConfigData[nontype4:nontype4 + Nb,7] 
        
        # Diesel Genset energy
        Eg_initial = ConfigData[nontype4 + Nb:nontype4 + Nb + Ng,7] 
        
        for i in range(Nb):
            SOC[i] = SOC[i] + cmd[Nd+i]/(3600*Eb_initial[i])

        Pgmax = -ConfigData[nontype4 + Nb:nontype4 + Nb + Ng,5]/pf

        Pg = np.zeros(Ng)
        
        for i in range(Ng):
            scaledPg = np.abs(cmd[Nd+Nb+i])/Pgmax  
            
            Pg[i] = np.abs(cmd[Nd+Nb+i])/( np.min([ (0.2/0.2)*(scaledPg - 0.20) + 0.2, (0.07/0.2)*(scaledPg - 0.40) + 0.27, \
                                                    (0.05/0.2)*(scaledPg - 0.60) + 0.32, (0.03/0.2)*(scaledPg - 0.80) + 0.35, \
                                                    (0.01/0.2)*(scaledPg - 1) + 0.36 ]) ) 
            
            SOE[i] = SOE[i] - Pg[i]/(3600*Eg_initial[i])
            
        time.sleep(1)

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
    global SOC
    global SOE
    global hovCmdVec 
    global measurement_reception_started
    global INCLUDE_TYPE4
    global pf 
    
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

    
    nontype4 = Nd+Nc+Npv
    
    Pgmax = -ConfigData[nontype4 + Nb:nontype4 + Nb + Ng,5]
    Pbmax = -ConfigData[nontype4:nontype4 + Nb,5]
    
    pf = 0.9
    
    hovCmdVec = np.concatenate(( np.ones(Nd), -Pbmax*np.ones(Nb)/pf, -Pgmax*np.ones(Ng)/pf), axis=None) 
        
    #determine if any type 4 generators are in this configuration and adjust type4 flag accordingly
    if (Ng + Nb) == 0:
        #no type 4 units defined
        INCLUDE_TYPE4 = False
    else:
        INCLUDE_TYPE4 = True

    SOC = np.ones(Nb)
    SOE = np.ones(Ng)
    
    return True
        

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="filename of config file (XLSX format) to define the scenario", default='-1')
    ap.add_argument("--scenarioProfileFile", help="filename of config file (XLSX format) to define the scenario", default='-1')
    ap.add_argument("--debug", action="store_true", help="turn debug mode on, including additional logging and messages")
    ap.add_argument("--localip", help="IPv4 address of this computer. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--localport", help="Port to receive UDP packets on for this computer. Default is 4000", default="4000")
    ap.add_argument("--remoteip", help="IPv4 address of the remote computer to send UDP-based commands to. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--remoteport", help="Port on remote computer to send UDP-based commands to. Default is 7100", default="7100")
    ap.add_argument("--log", help="filename of CSV log file to write latest system state and measurements to at each measurement update", default="-1")
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
    
    
    if args.scenarioProfileFile == '-1' or args.config[-5:] != ".xlsx":
        log.error("ERROR: must define Scenario profile file in XLSX format")
        sys.exit()
    else:
        scenarioProfile = args.scenarioProfileFile
    
    # load in scenario profile
    scenProfile = pd.read_excel(scenarioProfile, sheet_name='Sheet1')
    
    
    udpPortLocal = int(args.localport)
    udpPortRemote = int(args.remoteport)

    if udpPortLocal < 100 or udpPortLocal > 100000 or udpPortRemote < 100 or udpPortRemote > 100000:
        log.error("ERROR: both local and remote ports must be in range 100-100000")
        sys.exit()

    LOCAL_UDP_ADDR = (args.localip, udpPortLocal) #IP address and port to listen for measurements on for computer running this script
    REMOTE_HOST_ADDR = (args.remoteip, udpPortRemote) #IP:Port of remote host to send cmds to

    #initialize stateTable based on the scenario definition file
    INCLUDE_TYPE4 = False #include type 4 generation in dispatch and control
    
     #remove any existing log file that we appended to last time
    try:
        os.remove(SCENARIO_LOG_FILENAME)
    except FileNotFoundError as e:
        pass
    
    
    initSuccess = initializeViaConfigFile(scenarioConfigFile) 

    # setup threads
    # setup var for threads
    threadGo = True #set to false to quit threads
    
    tStart = time.time()
    
    dt = 0
    
    # startup thread to listen for measurements from DCG
    thread1 = Thread(target=sendMeasVec)
    thread1.daemon = True
    thread1.start()

    # startup thread to listen for HoV commands if appropriate
    thread2 = Thread(target=listenHovCmd)
    thread2.daemon = True
    thread2.start()

    # startup thread to simulate the continuous change in the SOC of the Battery and Diesel generators
    thread3 = Thread(target=getSOC)
    thread3.daemon = True
    thread3.start()
    

    ## do infinite while to keep main alive so that logging from threads shows in console
    try:
        while True:
            time.sleep(0.001)
    except KeyboardInterrupt:
        log.info("cleaning up threads and exiting...")
        threadGo = False
        log.info("done.")

