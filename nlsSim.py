#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOTICE

THIS COMPUTER SOFTWARE IS PROTECTED DATA UNDER THE TERMS OF NREL NON-DISCLOSURE AGREEMENT (NDA) NO. 20-14729. THIS SOFTWARE SHOULD ONLY BE USED
FOR PURPOSES OF THE DE-FOA-0001858 "OPEN"/"RVSG" PROJECT AND MAY NOT BE COPIED OR DISCLOSED OUTSIDE OF THE NDA CO-SIGNERS FOR ANY REASON.

NOTICE
 
THIS COMPUTER SOFTWARE IS COPYRIGHT (C) 2020, ALLIANCE FOR SUSTAINABLE ENERGY, LLC, ALL RIGHTS RESERVED

This computer software was produced by Alliance for Sustainable Energy, LLC under Contract No. DE-AC36-08GO28308 with the U.S. Department of Energy.
For 5 years from the date permission to assert copyright was obtained, the Government is granted for itself and others acting on its behalf a 
non-exclusive, paid-up, irrevocable worldwide license in this software to reproduce, prepare derivative works, and perform publicly and display 
publicly, by or on behalf of the Government. There is provision for the possible extension of the term of this license. Subsequent to that period 
or any extension granted, the Government is granted for itself and others acting on its behalf a non-exclusive, paid-up, irrevocable worldwide license 
in this software to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others 
to do so. The specific term of the license can be identified by inquiry made to Alliance for Sustainable Energy, LLC or DOE. NEITHER ALLIANCE FOR 
SUSTAINABLE ENERGY, LLC, THE UNITED STATES NOR THE UNITED STATES DEPARTMENT OF ENERGY, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS OR 
IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY DATA, APPARATUS, PRODUCT, OR PROCESS 
DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.

------------------------------------
| Net-load System Simulator (NLSS) |
------------------------------------

The NLSS:
-- Init --
1. Initializes a stateTable containing net-load statuses, measurements, set points, capacity limits, costs, etc. from the specified scenario configuration file.
   This includes energy capacity, state of charge, and efficiencies for type 4 generators.
2. Loads in a scenario profile containing a time series of Pmeas, Pmin/Pmax when appropriate, etc. for net-load resources in the simulation

-- Continuously --
3. Listens for dispatch command vectors (via UDP by default or ZMQ by selection) and uses these dispatch commands to update the net-load system
   state as follows:
    i. Deferrable Loads (type = 0): stores on/off state and updates Pmeas,i = (on/off),i * Pprofile,i
    ii. PV inverters (type = 3): stores power setpoint given (Psp) and updates Pmeas,i = max(Psp, Pmin) where Pmin is the time-varying (based on solar
        irradiance profile) power output capability of the inverter. Psp, Pmeas, and Pmin here are all negative (negative values indicate power generated).
    iii. Batt inverters/generators (type = 4): stores power set point given (Psp) and updates Pmeas,i = Psp while ensuring Pmin <= Pmeas,i <= Pmax.

-- Periodically thereafter --
4. Updates the stateTable based on the latest scenario profile values - Pprofile,i for loads and Pmin,Pmax for inverters and generators are updated
    i. By default a log file "logScenario.csv"
5. As part of #4 update, the state of charge of each type 4 generator is updated by calculating how much energy has been transferred to/from the generator
   based on the current power rate (Pmeas), while accounting for efficiency (i.e., input-side power is higher than output-side power, Pmeas to account for 
   the efficiency of the generator), and the elapsed time since the last SOC update.
6. Sends out via UDP (or publishes via ZMQ if that option is selected) the lastest set of net-load system measurements stored in the stateTable in format
   prescribed by the scenario configuration file. The SOC of each generator is checked while building the measurement vector. If a generator energy state
   of charge is full then update the Pmax = 0 (can't charge anymore) or if SOC is empty then update the Pmin = 0 (can't discharge anymore).

-- Simulation Logging --
By default, the following is logged to file throughout a simulation:
- Each time the stateTable is updated based on a scenario profile update, the current scenario time, totPmeas, slack generation margin, total non-critical
  load unserved, and the Pmeas of all individual net-loads is logged as a row to SCENARIO_LOG_FILENAME
- Each time a dispatch command is executed the scenario time, totPmeas (before dispatch cmd is executed), dispatch command, -1, and the Pmeas of all individual
  net-loads is logged as a row to SCENARIO_LOG_FILENAME
- Each time a dispatch command vector is received, it is logged, along with the current simulation time, to CMDRECV_LOG_FILENAME

If "state logging" (use command line flag --stlog) is on:
- The initial net-load system state, before any scenario profile updates, is logged to stateInit.csv
- The net-load system state after the first scenario profile update is logged to state0.csv
- Each time a dispatch command is executed the net-load system's starting and ending states are logged to files stateXst.csv and stateXed.csv where X is
  a counter of how many dispatch events have occurred

NOTES:
- A scenario configuration file defining the net-load resources and their attributes for the power system configuration in consideration must be specified
  via the command-line argument --config
- Use the --debug flag command-line argument to enable additional logging
- Can operate using UDP or ZMQ-based communication. UDP is default. Specify ZMQ via command line.
- For UDP-based communication, local ip and port and remote ip and port should be specified using the corresponding command-line arguments
- Use --stlog for logging of detailed system state before and after each dispatch command execution. May create a lot of files for long scenario profiles
  with many dispatch actions.

EXAMPLE USAGE FROM COMMAND LINE:
python nlsSim.py --config scenarioConfig_TC3.xlsx --profile scenarioProfile_TC3.xlsx --localip 127.0.0.1 --localport 4000 --remoteip 127.0.0.1 --remoteport 7100 --debug

-- VERSION HISTORY --
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
| Version | Who            | Comments                                                                                                                               |    
--------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
| 1.0     | B. Lundstrom   | Initial release                                                                                                                        |
| 1.0.1   | B. Lundstrom   | Updates to ensure initialization and periodic update from scenario profile support cases with multiple type 4 generators (e.g., TC#4)  |
|         |                | and trigger detailed system state logging only on command line flag versus by default                                                  |
| 1.1     | B. Lundstrom   | Add energy capacity tracking for type 4 generators, including initialization from config file, updating energy/SOC at each profile     |
|         |                | update, sending out measurement vector including SOC's and Pmin/Pmax's dependent on SOC state, and double-checking dispatch commands   |
|         |                | received to ensure that command for type 4 generator is feasible given generator's SOC (e.g., don't charge if full)                    |
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

__version__ = "1.1"
__author__ = "Blake Lundstrom, NREL"
__copyright__ = "Copyright 2020, Alliance for Sustainable Energy, LLC"

import random
import logging
import time
import numpy as np
np.set_printoptions(suppress=True,edgeitems=10)
import pandas as pd
import pickle
from threading import Thread, Lock
from scipy import interpolate
import zmq
import pyarrow as pa
import json
import sys
import os
from tabulate import tabulate
import socket
import struct
import argparse

### LOG FILE LOCATIONS ###
SCENARIO_LOG_FILENAME = "scenarioLog.csv" #log file where
CMDRECV_LOG_FILENAME = "scenarioLogUDPCmd.csv" #log file where all UDP commands received will be logged to
##########################

### PARAMETERS FOR ZMQ-BASED (UDP IS DEFAULT) COMMUNICATION (HARD-CODED FOR NOW) ###
#server info for communicating with dispatcher (topics are defined in main below)
PORT_PUB = 5021
PORT_SUB = 5020
RELAY_SERVER_ADDR = '127.0.0.1'

#info for communicating with dispatcher via relay
TOPIC_CMDS = 'rvsgDispatch:cmd'
TOPIC_REPLIES = 'rvsgDispatch:reply'
#####################################################################################

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

def recalcStateTableMeas():
    """
    Recalculate the current system state's calculated variables, including calculated variables within stateTable as well as summary global 
    variables totPmeas, totPmeasNoSlack, totNCLoadUnserved
    """

    global marginGenIncAvail #amount generation (type 3, type 10 slack) can increase (capacity minus load that is being served) - does NOT include grid power (this calc is used for off-grid op.)
    global totPmeas #sum of online net-load (type 0, type 1, type 3, type 10)
    global totPmeasNoSlack #sum of online net-load except type 10 (slack bus)
    global totNCLoadUnserved #non-critical load not being served
    global stateTable
    global lockST
    
    lockST.acquire()
    try:
        #calculate power available in each direction and associated cost
        stateTable.loc[:,'Pav_down'] = stateTable['Pmeas']-stateTable['Pmin']
        stateTable.loc[:,'Pav_down_cost'] = stateTable['Pav_down']*stateTable['cost_down']
        
        #have to calc Pav_up differently for loads because we use Pmax to store the expected power if that load was turned back on
        
        #for loads 
        stateTable.loc[(stateTable['type'] == 0) & (stateTable['status'] == 1), 'Pav_up'] = 0 #any loads that are status=on have no Pav_up
        stateTable.loc[(stateTable['type'] == 0) & (stateTable['status'] == 0), 'Pav_up'] = stateTable.loc[(stateTable['type'] == 0) & (stateTable['status'] == 0), 'Pmax'] #any loads that are status=off have Pav_up = Pmax
        stateTable.loc[stateTable['type']==0,'Pav_up_cost'] = 0
        
        #for inverters
        stateTable.loc[stateTable['type']==3,'Pav_up'] = stateTable.loc[stateTable['type']==3,'Pmax']-stateTable.loc[stateTable['type']==3,'Pmeas']
        stateTable.loc[stateTable['type']==3,'Pav_up_cost'] = stateTable.loc[stateTable['type']==3,'Pav_up']*stateTable.loc[stateTable['type']==3,'cost_up']

        if INCLUDE_TYPE4:
            #type 4 devices
            stateTable.loc[stateTable['type']==4,'Pav_up'] = stateTable.loc[stateTable['type']==4,'Pmax']-stateTable.loc[stateTable['type']==4,'Pmeas']
            stateTable.loc[stateTable['type']==4,'Pav_up_cost'] = stateTable.loc[stateTable['type']==4,'Pav_up']*stateTable.loc[stateTable['type']==4,'cost_up']
        
        #update margin var's
        if INCLUDE_TYPE4:
            totPmeas = stateTable[(stateTable['type'] == 0) | (stateTable['type'] == 1) | (stateTable['type'] == 3) | (stateTable['type'] == 4)| (stateTable['type'] == 10)]['Pmeas'].sum()
            marginGenIncAvail = stateTable[(stateTable['type'] == 3) | (stateTable['type'] == 4) | (stateTable['type'] == 10)]['Pav_down'].sum() - totPmeas
        else:
            totPmeas = stateTable[(stateTable['type'] == 0) | (stateTable['type'] == 1) | (stateTable['type'] == 3) | (stateTable['type'] == 10)]['Pmeas'].sum()
            marginGenIncAvail = stateTable[(stateTable['type'] == 3) | (stateTable['type'] == 10)]['Pav_down'].sum() - totPmeas

        totPmeasNoSlack = totPmeas - stateTable[(stateTable['type'] == 10)]['Pmeas'].sum()        
        totNCLoadUnserved = stateTable[stateTable['type'] == 0]['Pav_up'].sum()
    finally:
        lockST.release()
    
def sendStateTableToDispatcher():
    """
    Sends the latest net-load system state (stored in stateTable) to the dispatcher using UDP (default) or ZMQ. If using ZMQ a 
    simplified measurement vector based on the config file is sent. If ZMQ, the full stateTable DataFrame is sent.
    """
    
    global useUDPComm

    if useUDPComm:
        #send via UDP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
        vec = buildOutputMeasVecFromST()
        log.debug("sending meas to dispatcher via UDP packet: " + str(vec))
        p = len(vec) #determine how many floats we are sending\n",
        msgBytes = struct.pack('<{}'.format('f'*p),*vec.tolist()) #bytepack numpy row vector - use little Endian format (required by Opal-RT)\n",
        s.sendto(msgBytes, REMOTE_HOST_ADDR)
    else:
        #send via ZMQ
        # st = time.time()
        # stateTableArrow = pa.Table.from_pandas(stateTable) #other method to do the conversion
        stateTableArrow = pa.serialize(stateTable).to_buffer()
        # ed = time.time()
        # log.debug("took " + str((ed-st)*1000) + " ms to convert to pyArrow format")
        
        dispatchReplyBus.send_multipart([TOPIC_REPLIES.encode(), stateTableArrow])
        log.info("sent state table to dispatcher via ZMQ")

def getLatestMeasVec():
    """
    Returns a numpy vector with the latest measurements from the scenarioProfile
    """

    #for now, we obtain this vector via the scenarioProfile
    current_time = time.time()
    # current_time = tProfileStart + 750 #this line can be used to "pause" the scenario profile at a point in time
    dt_sec = current_time - tProfileStart

    # log.debug('dt_sec: ' + str(dt_sec))

    #make sure dt_sec doesn't run over the end of the time vector defined (i.e., hold at last value/row)
    if dt_sec > scenProfile['t'].max():
        dt_sec = scenProfile['t'].max()

    measVec = np.array(scenProfile[scenProfile['t'] >= dt_sec].iloc[0].astype('int32'))
    measVec = measVec[1:] #cut off the time column at the beginning

    #update load values based on their status - if status=0, Pmeas=0, else Pmeas as is in profile
    ldStatusArr = np.array(stateTable[stateTable['type'] == 0]['status'])
    ldStatusIX = np.where(ldStatusArr == 0)[0]
    measVec[ldStatusIX] = 0

    log.debug('measVec:')
    log.debug(measVec)

    return (measVec, dt_sec)

def getSOCArray():
    """
    Returns the state of charge for all type 4 generators in a numpy array (float) in ascending net-load id order
    """

    gids = np.array(stateTable[stateTable['type']==4]['id'])
    SOCs = np.zeros((len(gids),))

    ix = 0
    for gid in gids:
        SOCs[ix] = GenEData[gid]['soc']
        ix += 1
    
    return SOCs

def updateStateTableFromScenarioProf():
    """
    update state table based on latest load status in the stateTable and latest scenario (load and gen) profile values
    (i.e., dispatch cmd following method will update status and Pmeas = Pmax, but now we must update Pmeas of loads
    to be the latest value in the load profile (if load is on) and update Pmeas,Pmax of inverters to be the latest values
    based on time varying profiles)

    Also, the state of charge of each type 4 generator is updated by calculating how much energy has been transferred 
    to/from the generator based on the current power rate (Pmeas), while accounting for efficiency (i.e., input-side 
    power is higher than output-side power, Pmeas to account for the efficiency of the generator), and the elapsed time 
    since the last SOC update.
    """

    global justDispUp
    global tLastEnergyCalc
    global GenEData

    #get latest measurement vector
    (measVec, dt_sec) = getLatestMeasVec()

    lockST.acquire()
    try:
        ##update state table according to measVec
        #update controllable loads based on latest values (vector already accounts for latest 0/1 status)
        stateTable.loc[stateTable['type'] == 0, 'Pmeas'] = measVec[vecMeasIX[0,0]:vecMeasIX[0,1]] #non-crit loads
        stateTable.loc[(stateTable['type'] == 0) & (stateTable['Pmeas'] > 20), 'status'] = 1
        
        #update non-controllable (critical) load
        stateTable.loc[stateTable['type'] == 1, 'Pmeas'] = measVec[vecMeasIX[1,0]:vecMeasIX[1,1]]
        # log.debug('NON-Controllable Load: ' + str(measVec[numScenarioCL:(numScenarioCL+1)]) + " W")

        #update PV inverter Pmin's
        for ii in range(0, len(stateTable[stateTable['type'] == 3])):
            stateTable.loc[(stateTable[stateTable['type'] == 3].index[ii]), 'Pmin'] = measVec[(vecMeasIX[3,0] + ii*2 + 1):(vecMeasIX[3,0] + ii*2 + 2)]

        #TODO: may also want to read in PV Pmeas for the case that the PV inverter is non-responsive to our set point
        #and use this below to override our set point if below the set point and the limit

        #update PV inverters to ensure their Pmeas is clamped by their set point
        minPVPower = np.array(stateTable.loc[stateTable['type']==3,'Pmin'])
        PVInvSP = np.array(stateTable.loc[stateTable['type']==3,'Psp'])
        finalPVInvSP = np.max(np.vstack((minPVPower, PVInvSP)), 0)
        stateTable.loc[stateTable['type']==3,'Pmeas'] = finalPVInvSP

        if INCLUDE_TYPE4:
            # update Pmeas,Pmin,Pmax for all type 4 generators
            for ii in range(0,len(stateTable[stateTable['type'] == 4])):
                stateTable.loc[stateTable[stateTable['type'] == 4].index[ii],['Pmin','Pmax']] = measVec[(vecMeasIX[4,0] + ii*3 + 1):(vecMeasIX[4,0] + ii*3 + 3)] #do not update Pmeas from profile - this will be provided by dispatcher

            # update the energy state of charge for all type 4 generators
            dt = time.time() - tLastEnergyCalc

            for gid in np.array(stateTable[stateTable['type']==4]['id']):
                PgenMeas = stateTable.loc[gid,'Pmeas'] * -1.0 #convert to load convention for SOC purposes
                PgenMax = stateTable.loc[gid,'Pmin'] * -1.0
                
                if PgenMeas == 0:
                    effGen = 1 #avoid /0 case below
                    PgenPU = 0
                else:
                    PgenPU = PgenMeas / PgenMax
                    effGen = np.interp(PgenPU, effX, GenEData[gid]['eff'])

                Eold = GenEData[gid]['soc'] * GenEData[gid]['Emax']
                Enew = Eold - PgenMeas / effGen * (dt/3600.0)
                if Enew > GenEData[gid]['Emax']:
                    Enew = GenEData[gid]['Emax']
                    if stateTable.loc[gid,'Pmeas'] > 0:
                        #currently charging but hit SOC upper limit so stop charging
                        stateTable.loc[gid,'Pmeas'] = 0

                if Enew < 0:
                    Enew = 0
                    if stateTable.loc[gid,'Pmeas'] < 0:
                        #currently discharging but hit SOC lower limit so stop discharging
                        stateTable.loc[gid,'Pmeas'] = 0

                SOCnew = Enew / GenEData[gid]['Emax']

                #store new SOC
                GenEData[gid]['soc'] = SOCnew

                # log.debug("-- Updating nl=" + str(gid) + " energy cap --")
                # log.debug("PgenMeas: " + str(PgenMeas) + " PgenMax: " + str(PgenMax))
                # log.debug("PgenPU: " + str(PgenPU))
                # log.debug("effGen: " + str(effGen))
                # log.debug("dt: " + str(dt))
                # log.debug("Eold: " + str(Eold))
                # log.debug("Enew: " + str(Enew))
                # log.debug("SOCnew: " + str(SOCnew))

            tLastEnergyCalc = time.time()

        #update slackbus Pmin,Pmax
        stateTable.loc[stateTable['type'] == 10, ['Pmin','Pmax']] = measVec[vecMeasIX[10,0]+1:vecMeasIX[10,1]]

        #update grid Pmeas
        stateTable.loc[stateTable['type'] == 11, 'Pmeas'] = measVec[vecMeasIX[11,0]:vecMeasIX[11,1]]

    finally:
        lockST.release()

    #need to recalc before we do final adjustment of slack bus
    #recalc state table Pav's and reserve margin after having updated Pmeas for load and PV
    recalcStateTableMeas() 

    lockST.acquire()
    try:
        #with all others updated, update slack bus to attempt to get to Pmeas=0 or use all slack Pav_down (if above totPmeas=0) or Pav_up (if below totPmeas=0) capacity (whichever comes first)
        if INCLUDE_TYPE4:
            totPmeasLatest = stateTable[(stateTable['type'] == 0) | (stateTable['type'] == 1) | (stateTable['type'] == 3) | (stateTable['type'] == 4) | (stateTable['type'] == 10)]['Pmeas'].sum()
        else:
            totPmeasLatest = stateTable[(stateTable['type'] == 0) | (stateTable['type'] == 1) | (stateTable['type'] == 3) | (stateTable['type'] == 10)]['Pmeas'].sum()

        slackResDown = stateTable[(stateTable['type'] == 10)]['Pav_down'].sum()
        slackResUp = stateTable[(stateTable['type'] == 10)]['Pav_up'].sum()

        if totPmeasLatest > 0:
            #decrease totPmeas by moving slack Pav_down to slack Pmeas (i.e., dispatch slack gen)
            log.debug("reducing slack Pav_down to get closer to totPmeas=0")
            minRes = np.min((slackResDown, totPmeasLatest))
            
            log.debug("starting slack Pmeas=" + str(stateTable.loc[(stateTable['type'] == 10), "Pmeas"].sum()))
            log.debug("starting slack Pav_down=" + str(stateTable.loc[(stateTable['type'] == 10), "Pav_down"].sum()))
            stateTable.loc[(stateTable['type'] == 10), "Pmeas"] = stateTable.loc[(stateTable['type'] == 10), "Pmeas"] - minRes
            log.debug("ending slack Pmeas=" + str(stateTable.loc[(stateTable['type'] == 10), "Pmeas"].sum()))
        
        else:
            #increase totPmeas by moving slack Pav_up to slack Pmeas (i.e., dispatch slack -gen)
            log.debug("reducing slack Pav_up to get closer to totPmeas=0")
            minRes = np.min((np.abs(totPmeasLatest), slackResUp)) #totPmeasLatest is <0 here so use abs for comparison

            log.debug("starting slack Pmeas=" + str(stateTable.loc[(stateTable['type'] == 10), "Pmeas"].sum()))
            log.debug("starting slack Pav_up=" + str(stateTable.loc[(stateTable['type'] == 10), "Pav_up"].sum()))
            stateTable.loc[(stateTable['type'] == 10), "Pmeas"] = stateTable.loc[(stateTable['type'] == 10), "Pmeas"] + minRes
            log.debug("ending slack Pmeas=" + str(stateTable.loc[(stateTable['type'] == 10), "Pmeas"].sum()))

    finally:
        lockST.release()
    
    #final recalc after final adjustment of slack bus
    recalcStateTableMeas()
    
    #send state table to dispatcher
    sendStateTableToDispatcher()

    #log latest stateTable Pmeas to scenario log file
    with open(SCENARIO_LOG_FILENAME, "a") as f:
        dt = time.time() - tProfileStart
        a = np.hstack((np.array(dt), totPmeasNoSlack, marginGenIncAvail, totNCLoadUnserved, np.array(stateTable['Pmeas']).T, getSOCArray()))
        a = a.reshape((1,len(a))) #format a as a row vector for writing to file
        np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')

    if INCLUDE_TYPE4:
        cmdDownLim = stateTable[(stateTable['type'] == 0) & (stateTable['Pmeas'] > 0)].sort_values(by=['cost_down']).iloc[0:20]["Pmeas"].sum() + stateTable[(stateTable['type'] == 3) | (stateTable['type'] == 4) |(stateTable['type'] == 10)]["Pav_down"].sum()
    else:
        cmdDownLim = stateTable[(stateTable['type'] == 0) & (stateTable['Pmeas'] > 0)].sort_values(by=['cost_down']).iloc[0:20]["Pmeas"].sum() + stateTable[(stateTable['type'] == 3) | (stateTable['type'] == 10)]["Pav_down"].sum()
    cmdUpLim = stateTable[(stateTable['type'] == 0) & (stateTable['Pav_up'] > 0)].sort_values(by=['cost_up']).iloc[0:20]["Pav_up"].sum()
    ldUp = stateTable[(stateTable['type'] == 0) & (stateTable['Pav_up'] > 0)]
    if len(ldUp) > 0:
        cmdUpMinLd = ldUp.sort_values(by=['Pav_up']).iloc[0]["Pav_up"]
    else:
        cmdUpMinLd = -1

    slackBusPmeas = stateTable.loc[stateTable['type'] == 10, 'Pmeas'].sum()
    slackBusPmin = stateTable.loc[stateTable['type'] == 10, 'Pmin'].sum()
    slackBusPmax = stateTable.loc[stateTable['type'] == 10, 'Pmax'].sum()

    if INCLUDE_TYPE4:
        plist = [[dt_sec, totPmeasNoSlack, totPmeas, marginGenIncAvail, measVec[numScenarioCL:(numScenarioCL+1)], totNCLoadUnserved, cmdDownLim, cmdUpLim, cmdUpMinLd, slackBusPmeas, slackBusPmin, slackBusPmax, stateTable.loc[stateTable['type']==3,'Pmeas'].iloc[0], stateTable.loc[stateTable['type']==3,'Pmin'].iloc[0], stateTable.loc[stateTable['type']==3,'Psp'].iloc[0], stateTable.loc[stateTable['type']==4,'Pmin'].iloc[0], stateTable.loc[stateTable['type']==4,'Pmeas'].iloc[0], stateTable.loc[stateTable['type']==11,'Pmeas'].iloc[0]]]
        log.info("\n" + tabulate(plist, headers=["t (s)", "totPmeas", "totPmeasSlk", "genRes", "CritLoad", "NCLUnsrv", "+CmdLim", "-CmdLim", "cmdUpMinLd", "SlkPmeas", "SlkPmin", "SlkPmax", "PVPmeas", "PVPmin", "PVPsp", "MobPmin", "MobPmeas", "gridPmeas"], floatfmt="1.0f", tablefmt="plain"))
        log.info("Gen4 SOCs: " + str(getSOCArray()))
    else:
        plist = [[dt_sec, totPmeasNoSlack, totPmeas, marginGenIncAvail, measVec[numScenarioCL:(numScenarioCL+1)], totNCLoadUnserved, cmdDownLim, cmdUpLim, cmdUpMinLd, slackBusPmeas, slackBusPmin, slackBusPmax, stateTable.loc[stateTable['type']==3,'Pmeas'].iloc[0], stateTable.loc[stateTable['type']==3,'Pmin'].iloc[0], stateTable.loc[stateTable['type']==3,'Psp'].iloc[0], stateTable.loc[stateTable['type']==11,'Pmeas'].iloc[0]]]
        log.info("\n" + tabulate(plist, headers=["t (s)", "totPmeas", "totPmeasSlk", "genRes", "CritLoad", "NCLUnsrv", "+CmdLim", "-CmdLim", "cmdUpMinLd", "SlkPmeas", "SlkPmin", "SlkPmax", "PVPmeas", "PVPmin", "PVPsp", "gridPmeas"], floatfmt="1.0f", tablefmt="plain"))
        log.info("Gen4 SOCs: " + str(getSOCArray()))

def periodicUpdateStateTableFromScenProf():
    """
    Periodically update the net-load system state (stored in stateTable) based on the scenario profile
    """

    global stateTable
    global stLoggingON
    Ts_updateStTab = 1 #[sec] how often to update
    
    upC = 0 #counter of how many times we have updated the state table via this method
    
    while threadGo:
        st = time.time()
        updateStateTableFromScenarioProf()

        if upC == 0 and stLoggingON:
            #save copy of initial state table for analysis
            stateTable.to_csv('state0.csv')
        
        upC += 1
        time.sleep(np.max((0.1,(Ts_updateStTab - (time.time()-st) - 3/1000)))) #sleep for period - time it took to run this loop so we execute at exactly the rate of period
        
def listenDispatchCmdZMQ():
    """
    Listen for dispatch commands received via ZMQ topic. For each received call dispatchCmdExec()
    """

    global stateTable
    global numDisp #number of dispatches so far
    
    numDisp = 1

    while threadGo:
        (topic, data) = dispatchCmdBus.recv_multipart()
        log.info("\n\n>>> received on dispatch cmd bus:")
        # log.debug("topic: " + str(topic))
        log.info("data: " + str(data))
        data = json.loads(data)
        log.info(">>> Received Exec Cmd Dict: " + str(data))
        
        dP_star = data['dP_star']
        finalLdIDs = data['ldIds']
        finalInvIDs = data['invIds']
        invCmds = data['invCmds']

        dispatchCmdExec(dP_star, finalLdIDs, finalInvIDs, invCmds, numDisp)

def listenDispatchCmdUDP():
    """
    Listen for dispatch commands received via UDP. For each received, decode the UDP packet into load id's to be switched on/off,
    inverter id's and associated power set points, etc. and then call dispatchCmdExec() with this information
    """
    global vecCmdIX
    global numDisp

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #Use UDP sockets
    s.bind(LOCAL_UDP_ADDR) #IP:Port here is for the computer running this script
    log.info('Bound to ' + str(LOCAL_UDP_ADDR) + ' and waiting for UDP packet-based dispatch commands')

    numDisp = 1

    while threadGo:
        data,addr = s.recvfrom(10000) #buffer size can be smaller if needed, but this works fine
        l = len(data)
        p = int(l/4) #get number of floats sent
        alllist = struct.unpack('<{}'.format('f'*p),data)
        log.debug("Received udp-based dispatch packet: " + str(alllist))

        dispVec = np.array(alllist)

        #log dispVec to file
        with open(CMDRECV_LOG_FILENAME, "a") as f:
            dt = time.time() - tProfileStart
            a = np.hstack((np.array(dt), dispVec))
            a = a.reshape((1,len(a))) #format a as a row vector for writing to file
            np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')

        #convert dispatch vector format to load ids, inv ids, inv cmds format
        dP_star = dispVec[-1]

        if dP_star >= 0:
            #decrease in net-load power - loads off, invs increase
            ldIDs = np.where(dispVec[vecCmdIX[0,0]:vecCmdIX[0,1]] == 0)[0] #assumes loads in stateTable start with id=0
        else:
            ldIDs = np.where(dispVec[vecCmdIX[0,0]:vecCmdIX[0,1]] == 1)[0] #assumes loads in stateTable start with id=0

        #add PV inverters
        invIDs = []
        invCmds = []
        for ii in range(0,len(stateTable[stateTable['type'] == 3])):
            log.debug("UDP recv: adding inverter ID=" + str(stateTable[stateTable['type'] == 3]['id'].iloc[ii]) + " with value=" + str(dispVec[(vecCmdIX[3,0] + ii):(vecCmdIX[3,0] + ii + 1)]))
            invIDs.extend([stateTable[stateTable['type'] == 3]['id'].iloc[ii]])
            invCmds.extend([int(dispVec[(vecCmdIX[3,0] + ii):(vecCmdIX[3,0] + ii + 1)][0])])

        genIDs = []
        genCmds = []
        if INCLUDE_TYPE4:
            #type 4 devices

            for ii in range(0,len(stateTable[stateTable['type'] == 4])):
                log.debug("UDP recv: adding gen ID=" + str(stateTable[stateTable['type'] == 4]['id'].iloc[ii]) + " with value=" + str(dispVec[(vecCmdIX[4,0] + ii):(vecCmdIX[4,0] + ii + 1)]))
                genIDs.extend([stateTable[stateTable['type'] == 4]['id'].iloc[ii]])
                genCmds.extend([int(dispVec[(vecCmdIX[4,0] + ii):(vecCmdIX[4,0] + ii + 1)][0])])

        log.debug("UDP dispatch vec")
        log.debug("ldIDs: " + str(ldIDs))
        log.debug("invIDs: " + str(invIDs))
        log.debug("invCmds: " + str(invCmds))

        if INCLUDE_TYPE4:
            log.debug("genIDs: " + str(genIDs))
            log.debug("genCmds: " + str(genCmds))

        dispatchCmdExec(dP_star, ldIDs, invIDs, invCmds, numDisp, genIDs, genCmds)
        numDisp += 1

def dispatchCmdExec(dP_star, finalLdIDs, finalInvIDs, invCmds, numDisp, genIDs=[], genCmds=[]):
    """
    Execute a dispatch command by updating the net-load system state accordingly. Before executing type 4 generator commands
    checks SOC to ensure the charge or discharge command is feasible.

    -- Parameters --
    dP_star : int
        decrease in net-load group total power in Watts that should be achieved with the dispatch (negative value indicates 
        increase in total power is to be achieved)
    finalLdIDs : numpy array of int's
        Array of all deferrable load id's (int) that should be turned on or off
    finalInvIDs : list of int's
        all inverter id's (int) for which a new command set point is provided
    invCmds : list of float's
        inverter set point commands (float) to apply to each inverter with an id in finalInvIDs. Ordering of
        invCmds matches the same ordering of finalInvIDs
    numDisp : int
        Counter for how many dispatch events have occurred
    genIDs : list of int's
        all batt inverters/generator id's for which a new command set point is provided
    genCmds : list of float's
        generator set point commands (float) to apply to each batt inv./generator with an id in genIDs. Ordering of
        genCmds matches the same ordering of genIDs
    """

    global stateTable
    global justDispUp #just dispatched increase in Pmeas
    global stLoggingON

    #update copied version of state table first
    nl3 = stateTable.copy()

    # log starting condition of stateTable before applying dispatch commands
    if stLoggingON:
        log.debug("writing " + 'state' + str(numDisp) + 'st.csv')
        stateTable.to_csv('state' + str(numDisp) + 'st.csv')
    
    if dP_star >= 0:
        #decrease in total power
        
        #apply P* updates from inverters
        invCmdsArr = np.array(invCmds, dtype=np.int32)
        nl3.loc[finalInvIDs,'Pmeas'] = invCmdsArr
        nl3.loc[finalInvIDs,'Psp'] = invCmdsArr

        if INCLUDE_TYPE4:
            #apply P* updates from type 4 gens
            genCmdsArr = np.array(genCmds, dtype=np.int32)
            nl3.loc[genIDs,'Pmeas'] = genCmdsArr
            nl3.loc[genIDs,'Psp'] = genCmdsArr

            #if any of these commands is to charge a full battery or discharge a depleted type 4 generator, reset Pmeas and Psp to 0
            for gid in np.array(stateTable[stateTable['type']==4]['id']):
                if GenEData[gid]['soc'] >= 1 and nl3.loc[gid,'Pmeas'] > 0:
                    log.info("LIMITED: Gen " + str(gid) + " is fully charged and cannot be further charged. Setting Pmeas = 0")
                    nl3.loc[gid,'Pmeas'] = 0
                    nl3.loc[gid,'Psp'] = 0
                elif GenEData[gid]['soc'] <= 0 and nl3.loc[gid,'Pmeas'] < 0:
                    log.info("LIMITED: Gen " + str(gid) + " is fully discharged. Setting Pmeas = 0")
                    nl3.loc[gid,'Pmeas'] = 0
                    nl3.loc[gid,'Psp'] = 0

        #check all PV inverters in this list and ensure that Pmeas/Psp is not > capacity as of now
        for pi in range(len(invCmdsArr)):
            if nl3.loc[finalInvIDs[pi], 'type'] == 3:
                if nl3.loc[finalInvIDs[pi],'Pmeas'] < nl3.loc[finalInvIDs[pi],'Pmin']:
                    log.info("pv inverter set point " + str(invCmdsArr[pi]) + " is outside Pmin=" + str(nl3.loc[finalInvIDs[pi],'Pmin']) + " for id=" + str(pi))
                    nl3.loc[finalInvIDs[pi],'Pmeas'] = nl3.loc[finalInvIDs[pi],'Pmin']
                else:
                    log.info("pv inverter set point " + str(invCmdsArr[pi]) + " is within Pmin=" + str(nl3.loc[finalInvIDs[pi],'Pmin']) + " for id=" + str(pi))

        #TODO may need to verify that inverter set points are actually changes from their present operating point
        #update status of inverters given a command in stateTable
        nl3.loc[((stateTable['id'].isin(finalInvIDs)) & (stateTable['type'] == 3) & (nl3['Pmeas'] != nl3['Pmin'])),'status'] = 0
        
        #apply Pmeas updates from loads
        nl3.loc[finalLdIDs,'Pmeas'] = 0
        
        log.info("updating load status. Original count=" + str(nl3[nl3['status']>0]['Pmeas'].count()))

        #update status of loads in stateTable
        nl3.loc[stateTable['id'].isin(finalLdIDs),'status'] = 0
        
        log.info("updating load status. New count=" + str(nl3[nl3['status']>0]['Pmeas'].count()))

        ldStatus = np.array(nl3[nl3['type'] == 0]['status']) #load status vector
        log.debug("ldStatus: " + str(ldStatus))
        
        #update other params
        finalIDsAll = finalInvIDs + list(finalLdIDs)
        nl3.loc[finalIDsAll,'Pav_down'] = nl3.loc[finalIDsAll,'Pmeas'] - nl3.loc[finalIDsAll,'Pmin']
        nl3.loc[finalIDsAll,'Pav_down_cost'] = nl3.loc[finalIDsAll,'Pav_down'] * nl3.loc[finalIDsAll,'cost_down']
        nl3.loc[finalIDsAll,'Pav_up'] = nl3.loc[finalIDsAll,'Pmax'] - nl3.loc[finalIDsAll,'Pmeas']
        nl3.loc[finalIDsAll,'Pav_up_cost'] = nl3.loc[finalIDsAll,'Pav_up'] * nl3.loc[finalIDsAll,'cost_up']
        
        log.debug("dP_star: " + str(dP_star))
        log.debug("achieved nl2->nl3: " + str(stateTable['Pmeas'].sum() - nl3['Pmeas'].sum()))
        log.debug("remaining pow: " + str(nl3['Pmeas'].sum()))
        log.debug("--")
        
        if (stateTable['Pmeas'].sum() - nl3['Pmeas'].sum()) != dP_star:
            pctErr = ((stateTable['Pmeas'].sum() - nl3['Pmeas'].sum()) - dP_star)/dP_star*100
            log.error('\n\n\n\n\n\n\n!!!Did not achieve setpoint change desired (' + str(pctErr) + '% off)')
        
    else:
        #increase in total power
        
        #apply Pmeas updates from inverters
        invCmdsArr = np.array(invCmds, dtype=np.int32)
        nl3.loc[finalInvIDs,'Pmeas'] = invCmdsArr
        nl3.loc[finalInvIDs,'Psp'] = invCmdsArr

        if INCLUDE_TYPE4:
            #apply P* updates from type 4 gens
            genCmdsArr = np.array(genCmds, dtype=np.int32)
            nl3.loc[genIDs,'Pmeas'] = genCmdsArr
            nl3.loc[genIDs,'Psp'] = genCmdsArr

            #if any of these commands is to charge a full battery or discharge a depleted type 4 generator, reset Pmeas and Psp to 0
            for gid in np.array(stateTable[stateTable['type']==4]['id']):
                if GenEData[gid]['soc'] >= 1 and nl3.loc[gid,'Pmeas'] > 0:
                    log.info("LIMITED: Gen " + str(gid) + " is fully charged and cannot be further charged. Setting Pmeas = 0")
                    nl3.loc[gid,'Pmeas'] = 0
                    nl3.loc[gid,'Psp'] = 0
                elif GenEData[gid]['soc'] <= 0 and nl3.loc[gid,'Pmeas'] < 0:
                    log.info("LIMITED: Gen " + str(gid) + " is fully discharged. Setting Pmeas = 0")
                    nl3.loc[gid,'Pmeas'] = 0
                    nl3.loc[gid,'Psp'] = 0

        #update status of inverters given a command in stateTable
        nl3.loc[((stateTable['id'].isin(finalInvIDs)) & (nl3['Pmeas'] != nl3['Pmin'])),'status'] = 0
        
        #apply Pmeas updates from loads
        nl3.loc[finalLdIDs,'Pmeas'] = nl3.loc[finalLdIDs,'Pmax'] #this assumes that load goes to Pmax initially when turned on
        
        #update status of loads in stateTable
        nl3.loc[stateTable['id'].isin(finalLdIDs),'status'] = 1
        
        log.info("updating load status. New count=" + str(nl3[nl3['status']>0]['Pmeas'].count()))

        ldStatus = np.array(nl3[nl3['type'] == 0]['status']) #load status vector
        log.debug("ldStatus: " + str(ldStatus))
        
        #update other params
        finalIDsAll = finalInvIDs + list(finalLdIDs)
        nl3.loc[finalIDsAll,'Pav_down'] = nl3.loc[finalIDsAll,'Pmeas'] - nl3.loc[finalIDsAll,'Pmin']
        nl3.loc[finalIDsAll,'Pav_down_cost'] = nl3.loc[finalIDsAll,'Pav_down'] * nl3.loc[finalIDsAll,'cost_down']
        nl3.loc[finalIDsAll,'Pav_up'] = nl3.loc[finalIDsAll,'Pmax'] - nl3.loc[finalIDsAll,'Pmeas']
        nl3.loc[finalIDsAll,'Pav_up_cost'] = nl3.loc[finalIDsAll,'Pav_up'] * nl3.loc[finalIDsAll,'cost_up']
        
        log.debug("dP_star: " + str(dP_star))
        log.debug("achieved nl2->nl3: " + str(stateTable['Pmeas'].sum() - nl3['Pmeas'].sum()))
        log.debug("remaining pow: " + str(nl3['Pmeas'].sum()))
        log.debug("--")
        
        if (stateTable['Pmeas'].sum() - nl3['Pmeas'].sum()) != dP_star:
            pctErr = ((stateTable['Pmeas'].sum() - nl3['Pmeas'].sum()) - dP_star)/dP_star*100
            log.error('\n\n\n\n\n\n\n!!!Did not achieve setpoint change desired (' + str(pctErr) + '% off)')
        
    justDispUp = True

    #set stateTable to updated version
    lockST.acquire()
    try:
        stateTable = nl3.copy()
        log.info("updated stateTable. New Pmeas = " + str(stateTable['Pmeas'].sum()) + " Pmeas (nl3) = " + str(nl3['Pmeas'].sum()))

    finally:
        lockST.release()

    #log latest stateTable Pmeas to scenario log file
    with open(SCENARIO_LOG_FILENAME, "a") as f:
        dt = time.time() - tProfileStart
        a = np.hstack((np.array([dt,totPmeasNoSlack,dP_star,-1]), np.array(stateTable['Pmeas']).T, getSOCArray()))
        a = a.reshape((1,len(a))) #format a as a row vector for writing to file
        np.savetxt(f, a, fmt='%1.3f', delimiter=',', newline='\n')

    #update state table using latest load/gen profile and send state table to dispatcher
    updateStateTableFromScenarioProf()
    
    if stLoggingON:
        stateTable.to_csv('state' + str(numDisp) + 'ed.csv')

    numDisp += 1

def initializeViaConfigFile(filename):
    """
    Loads scenario config file and runs some checks to ensure all data is correct. Initializes stateTable using data in the config file

    -- Returns --
    True if initialization was successful, False if an error occurred
    """
    global stateTable
    global scenarioConfig
    global numScenarioCL
    global vecMeasIX
    global vecCmdIX
    global INCLUDE_TYPE4
    global GenEData

    #verify that file is of XLSX format (or at least filename is...)
    if filename[-5:] != ".xlsx":
        log.error('Config file is not a XLSX file')
        return False
    
    #read in XLSX file via pandas
    df = pd.read_excel(filename, sheet_name='Sheet1')
    df.loc[:,'id'] = df.index.values
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
    
    # if file contains type 4 devices, validate efficiency and energy-related data is in correct format
    if len(df[df['type'] == 4]) > 0:
        if ("SOCstart" not in cols) or ("Eff0" not in cols) or ("Eff20" not in cols) or ("Eff0" not in cols) or ("Eff0" not in cols) or ("Eff0" not in cols) or ("Eff0" not in cols) or ("Emax" not in cols):
            log.error("Type 4 net-load devices must specify SOCstart, Emax, and Eff0-Eff100")
            return False

        for gid in np.array(df[df['type']==4]['id']):
            if df.loc[gid,'SOCstart'] < 0 or df.loc[gid,'SOCstart'] > 1:
                log.error("Net-load id=" + str(gid) + " must have 0 <= SOCstart <= 1")
                return False
            
            if df.loc[gid,'Eff0'] < 0 or df.loc[gid,'Eff0'] > 1:
                log.error("Net-load id=" + str(gid) + " must have Eff0 in pu, between 0-1")
                return False

            if df.loc[gid,'Eff20'] < 0 or df.loc[gid,'Eff20'] > 1:
                log.error("Net-load id=" + str(gid) + " must have Eff20 in pu, between 0-1")
                return False

            if df.loc[gid,'Eff40'] < 0 or df.loc[gid,'Eff40'] > 1:
                log.error("Net-load id=" + str(gid) + " must have Eff40 in pu, between 0-1")
                return False

            if df.loc[gid,'Eff60'] < 0 or df.loc[gid,'Eff60'] > 1:
                log.error("Net-load id=" + str(gid) + " must have Eff60 in pu, between 0-1")
                return False
        
            if df.loc[gid,'Eff80'] < 0 or df.loc[gid,'Eff80'] > 1:
                log.error("Net-load id=" + str(gid) + " must have Eff80 in pu, between 0-1")
                return False

            if df.loc[gid,'Eff100'] < 0 or df.loc[gid,'Eff100'] > 1:
                log.error("Net-load id=" + str(gid) + " must have Eff100 in pu, between 0-1")
                return False

    # begin state table with just the defining columns (ignore other columns that might be present) and add an autoincrementing column (0-) as id
    cfg = df[['type','status','cost_up','cost_down','Pmeas','Pmin','Pmax']]
    cfg.loc[:,'id'] = cfg.index.values

    #calculate power available in each direction and associated cost
    cfg.loc[:,'Pav_down'] = cfg['Pmeas'] - cfg['Pmin']
    cfg.loc[:,'Pav_down_cost'] = cfg['Pav_down'] * cfg['cost_down']
    cfg.loc[:,'Pav_up'] = cfg['Pmax'] - cfg['Pmeas']
    cfg.loc[(cfg['type'] == 0) & (cfg['status'] == 1), 'Pav_up'] = 0 #any loads that are status=on have no Pav_up
    cfg.loc[:,'Pav_up_cost'] = cfg['Pav_up'] * cfg['cost_up']

    cfg = cfg.astype('int32')

    log.debug('final config')
    log.debug(cfg)
    scenarioConfig = cfg.copy() #save the scenario config for later reference

    numScenarioCL = len(scenarioConfig[scenarioConfig['type'] == 0]) #log how many controllable loads there are

    #determine if any type 4 generators are in this configuration and adjust type4 flag accordingly
    if len(cfg[cfg['type'] == 4]) == 0:
        #no type 4 units defined
        INCLUDE_TYPE4 = False
    else:
        INCLUDE_TYPE4 = True

    stateTable = cfg.copy()
    #need to add power set point column that will be used for PV inverters
    stateTable['Psp'] = 0
    stateTable.loc[stateTable['type']==3, 'Psp'] = stateTable.loc[stateTable['type']==3, 'Pmin']

    #determine calculated parameters and sys margins for stateTable
    recalcStateTableMeas()

    #note starting and ending indicies for type 0,1,3,4,10,11 items in the final vector we'll use for UDP measurements
    vecMeasIX = np.zeros((12,2)) #2-D array where columns are 0=starting index, 1=ending index and rows correspond to type so row 0 will have start and end for type 0 net-load. Unused types will just keep 0's
    vecMeasIX[0,0] = 0 #assume type 0 always first
    vecMeasIX[0,1] = len(cfg[cfg['type'] == 0])
    vecMeasIX[1,0] = vecMeasIX[0,1]
    vecMeasIX[1,1] = vecMeasIX[1,0] + len(cfg[cfg['type'] == 1])
    vecMeasIX[3,0] = vecMeasIX[1,1]
    vecMeasIX[3,1] = vecMeasIX[3,0] + len(cfg[cfg['type'] == 3])*2 #inverters have Pmeas,Pmin for each inverter - must interleave later so 2X
    vecMeasIX[4,0] = vecMeasIX[3,1]
    vecMeasIX[4,1] = vecMeasIX[4,0] + len(cfg[cfg['type'] == 4])*3 #gens have Pmeas,Pmin,Pmax for each gen
    vecMeasIX[10,0] = vecMeasIX[4,1]
    vecMeasIX[10,1] = vecMeasIX[10,0] + len(cfg[cfg['type'] == 10])*3 #slack bus has Pmeas,Pmin,Pmax so 3X
    vecMeasIX[11,0] = vecMeasIX[10,1]
    vecMeasIX[11,1] = vecMeasIX[11,0] + len(cfg[cfg['type'] == 11])
    vecMeasIX = vecMeasIX.astype(int)

    #note starting and ending indicies for type 0,3,4 items in the final vector we'll use for UDP commands
    vecCmdIX = np.zeros((5,2)) #2-D array where columns are 0=starting index, 1=ending index and rows correspond to type so row 0 will have start and end for type 0 net-load. Unused types will just keep 0's
    vecCmdIX[0,0] = 0 #assume type 0 always first
    vecCmdIX[0,1] = len(cfg[cfg['type'] == 0])
    vecCmdIX[3,0] = vecCmdIX[0,1]
    vecCmdIX[3,1] = vecCmdIX[3,0] + len(cfg[cfg['type'] == 3])
    vecCmdIX[4,0] = vecCmdIX[3,1]
    vecCmdIX[4,1] = vecCmdIX[4,0] + len(cfg[cfg['type'] == 4])
    vecCmdIX = vecCmdIX.astype(int)

    #store energy and efficiency related variables for type 4 generation with energy input
    for gid in np.array(df[df['type']==4]['id']):
        GenEData[gid] = {}
        GenEData[gid]['soc'] = float(df.loc[gid, 'SOCstart'])
        GenEData[gid]['Emax'] = float(df.loc[gid, 'Emax'])
        GenEData[gid]['eff'] = np.array([float(df.loc[gid, 'Eff0']), float(df.loc[gid, 'Eff20']), float(df.loc[gid, 'Eff40']), float(df.loc[gid, 'Eff60']), float(df.loc[gid, 'Eff80']), float(df.loc[gid, 'Eff100'])])

    log.info("Successfully initialized with " + str(len(stateTable[stateTable['type']==0])) + " loads, " + str(len(stateTable[stateTable['type']==3])) + " PV invs, and " + str(len(stateTable[stateTable['type']==4])) + " Batt invs/Gens.")
    log.debug("GenEData: " + str(GenEData))
    #TODO: add more documentation on this method

    return True

def buildOutputMeasVecFromST():
    """
    Build up and return the output measurement vector, representing the net-load system's current state,
    in the format specified by the scenario config file. If a type 4 generator energy state
    of charge is full then update the Pmax = 0 (can't charge anymore) or if SOC is empty then update 
    the Pmin = 0 (can't discharge anymore).
    """

    global vecMeasIX

    #build up meas vector using stateTable and vecMeasIX to see if it is correct
    numGen = vecCmdIX[4,1] - vecCmdIX[4,0] #number of type 4 generators

    mesVec = np.zeros((vecMeasIX[11,1] + numGen,)) #end of measVec has numGen SOC measurements

    mesVec[vecMeasIX[0,0]:vecMeasIX[0,1]] = stateTable[stateTable['type'] == 0]['Pmeas'] #add non-crit loads
    mesVec[vecMeasIX[1,0]:vecMeasIX[1,1]] = stateTable[stateTable['type'] == 1]['Pmeas'] #add crit load
    #add invs - Pmeas first, Pmin next for each inverter one at a time
    for ii in range(0,len(stateTable[stateTable['type'] == 3])):
        mesVec[(vecMeasIX[3,0] + ii*2):(vecMeasIX[3,0] + ii*2 + 2)] = stateTable[stateTable['type'] == 3].iloc[ii][['Pmeas','Pmin']]
        # mesVec[(vecMeasIX[3,0] + ii*2 + 1):(vecMeasIX[3,0] + ii*2 + 2)] = stateTable[stateTable['type'] == 3].iloc[ii]['Pmin']

    if INCLUDE_TYPE4:
        #apply P* updates from all type 4 gens in Pmeas,Pmin,Pmax order for each gen
        ixm = 0
        for gid in np.array(stateTable[stateTable['type']==4]['id']):
            mesVec[(vecMeasIX[4,0] + ixm*3):(vecMeasIX[4,0] + ixm*3 + 3)] = stateTable.loc[gid, ['Pmeas','Pmin','Pmax']]
            
            log.debug("checking SOC(" + str(gid) + ")=" + str(GenEData[gid]['soc']))

            #SOC checks - if full (SOC=1) then Pmax = 0, if depleted (SOC=0) then Pmin = 0
            if GenEData[gid]['soc'] >= 1:
                #set Pmax = 0 in mesVec we send out
                mesVec[(vecMeasIX[4,0] + ixm*3 + 2):(vecMeasIX[4,0] + ixm*3 + 3)] = 0
                log.debug("setting Pmax = 0")
            elif GenEData[gid]['soc'] <= 0:
                #set Pmin = 0 in mesVec we send out
                mesVec[(vecMeasIX[4,0] + ixm*3 + 1):(vecMeasIX[4,0] + ixm*3 + 2)] = 0
                log.debug("setting Pmin = 0")
            
            ixm += 1

    mesVec[vecMeasIX[10,0]:vecMeasIX[10,1]] = stateTable[stateTable['type'] == 10][['Pmeas','Pmin','Pmax']] #add slack bus
    mesVec[vecMeasIX[11,0]:vecMeasIX[11,1]] = stateTable[stateTable['type'] == 11]['Pmeas'] #add grid connection

    #add SOC measurements
    ixSOC = vecMeasIX[11,1]
    for gid in np.array(stateTable[stateTable['type']==4]['id']):
        mesVec[ixSOC] = GenEData[gid]['soc']
        ixSOC += 1

    # log.debug("mesVec:")
    # log.debug(mesVec)
    return mesVec

if __name__ == "__main__":  
    ap = argparse.ArgumentParser()
    ap.add_argument("--zmq", action="store_true", help="(flag) use zmq for communications instead of default udp")
    ap.add_argument("--config", help="filename of config file (XLSX format) to define this scenario", default='-1')
    ap.add_argument("--profile", help="filename of scenario profile (XLSX format) defining the net-load profile for this scenario", default='-1')
    ap.add_argument("--debug", action="store_true", help="(flag) turn debug mode on, including additional messages")
    ap.add_argument("--localip", help="IPv4 address of this computer. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--localport", help="Port to receive UDP packets on for this computer. Default is 4000", default="4000")
    ap.add_argument("--remoteip", help="IPv4 address of the remote computer to send UDP-based commands to. Default is 127.0.0.1", default="127.0.0.1")
    ap.add_argument("--remoteport", help="Port on remote computer to send UDP-based commands to. Default is 7100", default="7100")
    ap.add_argument("--stlog", action="store_true", help="(flag) log detailed state change information before and after each dispatch")

    args = ap.parse_args()

    useUDPComm = not args.zmq
    log.info("USE UDP COMM: " + str(useUDPComm))

    debugModeON = args.debug
    #TODO: also update later instances of writing csv logs to check this var
    if debugModeON:
        log.setLevel(logging.DEBUG)
        log.info("DEBUG mode activated")
    else:
        log.setLevel(logging.INFO)
    
    stLoggingON = args.stlog

    if args.config == '-1' or args.config[-5:] != ".xlsx":
        log.error("ERROR: must define scenario config file in XLSX format")
        sys.exit()
    else:
        scenarioConfigFile = args.config

    if args.profile == '-1' or args.profile[-5:] != ".xlsx":
        log.error("ERROR: must define scenario profile file in XLSX format")
        sys.exit()
    else:
        scenarioProfileFile = args.profile

    udpPortLocal = int(args.localport)
    udpPortRemote = int(args.remoteport)

    if udpPortLocal < 100 or udpPortLocal > 100000 or udpPortRemote < 100 or udpPortRemote > 100000:
        log.error("ERROR: both local and remote ports must be in range 100-100000")
        sys.exit()

    LOCAL_UDP_ADDR = (args.localip, udpPortLocal) #IP address and port to listen for measurements on for computer running this script
    REMOTE_HOST_ADDR = (args.remoteip, udpPortRemote) #IP:Port of remote host to send cmds to

    lockST = Lock() #lock to ensure no concurrent modification of stateTable

    #initialize stateTable based on the scenario definition file
    INCLUDE_TYPE4 = False #include type 4 generation in dispatch and control
    GenEData = {} #dictionary to store generator energy data in format GenEData = {<netload id>:{"eff":<efficiency_array>,"soc":<pu efficiency>,"Emax":<energy capacity in Wh>}, ...}
    effX = [0, 0.2, 0.4, 0.6, 0.8, 1.0] #the "x-axis" of power values for efficiency vs. power level interpolation
    initSuccess = initializeViaConfigFile(scenarioConfigFile)

    if not initSuccess:
        log.error('ERROR: Did not successfully init - see previous errors')
        sys.exit()
    
    #load combined scenario profile (has loads and gen)
    scenProfile = pd.read_excel(scenarioProfileFile, sheet_name='Sheet1')

    #verify that configDispatch scenario definition file lines up with scenario profile in terms of # of columns (TODO: more robust check based on net-load types later)
    #scenProfile has 1 extra column for time index, 2 columns for Slack-Pmin and Slack-Pmax, 1 extra column per type 3 PV inverter (PV,Pmin), and 2 extra columns per
    #type 4 generator (Pmin,Pmax)
    colOffset = 1 + 2 + len(stateTable[stateTable['type'] == 3])*1
    if INCLUDE_TYPE4:
        colOffset += len(stateTable[stateTable['type'] == 4])*2

    if (len(scenProfile.columns) - colOffset) != len(stateTable):
        log.error('ERROR: Number of net-loads in scenario profile does not match number in scenario config definition')
        log.error('scenProfile (cols=' + str(len(scenProfile.columns)) + ')')
        log.error(scenProfile)
        log.error('stateTable (len=' + str(len(stateTable)) + ')')
        log.error(stateTable)
        sys.exit()
    
    #save initial state table to file for debug purposes
    if stLoggingON:
        stateTable.to_csv('stateInit.csv')

    #remove any existing log file that we appended to last time
    try:
        os.remove(SCENARIO_LOG_FILENAME)
        os.remove(CMDRECV_LOG_FILENAME)
    except FileNotFoundError as e:
        pass
    
    ## prepare var for threads
    threadGo = True #make false to stop all threads
    justDispUp = False #flag if we just dispatched to increase Pmeas (used to trigger logging)
    
    # start the scenario profile time
    tProfileStart = time.time()
    tLastEnergyCalc = tProfileStart
    
    ## startup periodic update thread that reads load/PV profiles, updates state table accordingly, triggers margin calcs, and triggers send to centralDispatcher
    thread1 = Thread(target=periodicUpdateStateTableFromScenProf)
    thread1.daemon = True
    thread1.start()

    if useUDPComm:
        ## startup thread that will listen for commands from the dispatcher and update the state table accordingly
        thread2 = Thread(target=listenDispatchCmdUDP)
        thread2.daemon = True
        thread2.start()
    else:
        #use ZMQ comm
        ## setup socket and pub/sub bus connections to centralized dispatch control
        context = zmq.Context.instance()

        #replybus - DCG sends replies/updates to centralized dispatch via this bus
        dispatchReplyBus = context.socket(zmq.PUB)
        dispatchReplyBus.connect("tcp://%s:%d" %(RELAY_SERVER_ADDR, PORT_PUB))
        time.sleep(0.1)
        
        #cmdbus - endDevice receives commands from dispatch via this bus
        dispatchCmdBus = context.socket(zmq.SUB)
        dispatchCmdBus.setsockopt_string(zmq.SUBSCRIBE, TOPIC_CMDS)
        dispatchCmdBus.connect("tcp://%s:%d" %(RELAY_SERVER_ADDR, PORT_SUB))
        time.sleep(0.1)
    
        ## startup thread that will listen for commands from the dispatcher and update the state table accordingly
        thread2 = Thread(target=listenDispatchCmdZMQ)
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
    
    ##TODO could make load profiles "responsive" so that when load is turned off they reset their timer and restart from the beginning (maybe in endDevice file)