'''
Obtain datasets of UWB measurements in TWR mode with crazyflies
Requirements: optiTrack system and crazyflie python library
Author: Sven Pfeiffer, MAVLab, TUDelft
'''
import logging
import time
from threading import Timer

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.drivers.crazyradio import Crazyradio
from cflib.crazyflie.extpos import Extpos

import ast
import sys
from NatNetClient import NatNetClient

import numpy as np
import math
import trajectories
import customUtils as util

# system configurations
cf_uri = 'radio://0/100/2M/E7E7E7E7E7'
optiTrackID = 1 # drone's ID in optiTrack
fileName = 'multiSquare_wMHE.csv'

# Trajectory
initial_pos = [0,0]
altitude = 1.5
yaw = 0
setpoints = []
setpoints.append(trajectories.takeoff(initial_pos, altitude, yaw))
setpoints.append(trajectories.xySquare(4.0, altitude, yaw))
setpoints.append(trajectories.land(initial_pos, altitude, yaw))


# enable log configs
log_ot_position = True
log_ot_attitude = True
log_cf_twr = True
log_cf_tdoa = False
log_cf_attitude = True
log_cf_kalman = False
log_cf_mhe = False

# prepare log variables
ot_position = np.zeros(3) # 3D position from optiTrack
ot_attitude = np.zeros(3) # 3D attitude from optiTrack
cf_twr = np.zeros(4)
cf_tdoa = np.zeros(6)
cf_attitude = np.zeros(3)
cf_kalman = np.zeros(3)
cf_mhe = np.zeros(3)
cf_twr = np.zeros(4) # [d0, d1, d2, d3]
timetick = 0

# prepare log file
file = open(fileName, 'w')
file.write('timeTick')
if log_ot_position:
    file.write(', otX, otY, otZ')
if log_ot_attitude:
    file.write(', otPitch, otRoll, otYaw')
if log_cf_twr:
    file.write(', twr0, twr1, twr2, twr3')
if log_cf_tdoa:
    file.write(', tdoa01, tdoa02, tdoa03, tdoa12, tdoa13, tdoa23')
if log_cf_attitude:
    file.write(', roll, pitch, yaw')
if log_cf_kalman:
    file.write(', kalX, kalY, kalZ')
if log_cf_mhe:
    file.write(', mheX, mheY, mheZ')
file.write('\n')


# This is a callback function that gets connected to the NatNet client and called once per mocap frame.
def receiveNewFrame( frameNumber, markerSetCount, unlabeledMarkersCount, rigidBodyCount, skeletonCount,
                    labeledMarkerCount, latency, timecode, timecodeSub, timestamp, isRecording, trackedModelsChanged ):
	pass

# This is a callback function that gets connected to the NatNet client. It is called once per rigid body per frame
def receiveRigidBodyFrame( id, position, rotation ):
    global ot_position, ot_attitude
    if id == optiTrackID:
        # rotate coordinates to match anchor coordinates
        ot_position[0] = position[2]
        ot_position[1] = position[0]
        ot_position[2] = position[1]
        ot_attitude = util.quat2euler(rotation)

streamingClient = NatNetClient() # Create a new NatNet client
streamingClient.newFrameListener = receiveNewFrame
streamingClient.rigidBodyListener = receiveRigidBodyFrame
streamingClient.run() # Run perpetually on a separate thread.

logging.basicConfig(level=logging.ERROR) # Only output errors from the logging framework

class LoggingExample:
    def __init__(self, crazyflie, link_uri):
        """ Initialize and run the example with the specified link_uri """
        self._cf = crazyflie
        # Connect some callbacks from the Crazyflie API
        self._cf.connected.add_callback(self._connected)
        self._cf.disconnected.add_callback(self._disconnected)
        self._cf.connection_failed.add_callback(self._connection_failed)
        self._cf.connection_lost.add_callback(self._connection_lost)
        print('Connecting to %s' % link_uri) # Try to connect to the Crazyflie
        self._cf.open_link(link_uri)
        self.is_connected = True # Variable used to keep main loop occupied until disconnect
        time.sleep(4) # wait sometime for connection

    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""
        print('Connected to %s' % link_uri)
        # add log configs
        if log_cf_twr:
            self._add_log_config_twr()
        if log_cf_tdoa:
            #self._add_log_config_tdoa()
            pass
        if log_cf_attitude:
            self._add_log_config_att()
        if log_cf_kalman:
            #self._add_log_config_kalman()
            pass
        if log_cf_mhe:
            self._add_log_config_mhe()

    def _add_log_config_twr(self):
        self._lg_twr = LogConfig(name='twr', period_in_ms=10)
        self._lg_twr.add_variable('ranging.distance0','float')
        self._lg_twr.add_variable('ranging.distance1','float')
        self._lg_twr.add_variable('ranging.distance2','float')
        self._lg_twr.add_variable('ranging.distance3','float')
        try:
            self._cf.log.add_config(self._lg_twr)
            self._lg_twr.data_received_cb.add_callback(self._twr_log_data)
            self._lg_twr.error_cb.add_callback(self._twr_log_error)
            self._lg_twr.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Distance log config, bad configuration.')           


    def _twr_log_data(self, timestamp, data, logconf):
        global cf_twr, timetick, ot_position, ot_attitude
        
        cf_twr[0] = data['ranging.distance0']
        cf_twr[1] = data['ranging.distance1']
        cf_twr[2] = data['ranging.distance2']
        cf_twr[3] = data['ranging.distance3']
        timetick = timestamp
        self._write_out_log_data()

    def _twr_log_error(self, logconf, msg):
        print('Error when logging %s: %s' % (logconf.name, msg))
    
    
    def _add_log_config_att(self):
        self._lg_att = LogConfig(name='attitude', period_in_ms=10)
        self._lg_att.add_variable('stabilizer.roll','float')
        self._lg_att.add_variable('stabilizer.pitch','float')
        self._lg_att.add_variable('stabilizer.yaw','float')
        try:
            self._cf.log.add_config(self._lg_att)
            self._lg_att.data_received_cb.add_callback(self._att_log_data)
            self._lg_att.error_cb.add_callback(self._att_log_error)
            self._lg_att.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Distance log config, bad configuration.')           


    def _att_log_data(self, timestamp, data, logconf):
        global cf_attitude
        
        cf_attitude[0] = data['stabilizer.roll']
        cf_attitude[1] = data['stabilizer.pitch']
        cf_attitude[2] = data['stabilizer.yaw']
        
    def _att_log_error(self, logconf, msg):
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _add_log_config_mhe(self):
        self._lg_mhe = LogConfig(name='mhe', period_in_ms=10)
        self._lg_mhe.add_variable('mhe.uwbX','float')
        self._lg_mhe.add_variable('mhe.uwbY','float')
        self._lg_mhe.add_variable('mhe.uwbZ','float')
        try:
            self._cf.log.add_config(self._lg_mhe)
            self._lg_mhe.data_received_cb.add_callback(self._mhe_log_data)
            self._lg_mhe.error_cb.add_callback(self._mhe_log_error)
            self._lg_mhe.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Distance log config, bad configuration.')           


    def _mhe_log_data(self, timestamp, data, logconf):
        global cf_mhe
        
        cf_mhe[0] = data['mhe.uwbX']
        cf_mhe[1] = data['mhe.uwbY']
        cf_mhe[2] = data['mhe.uwbZ']
        
    def _mhe_log_error(self, logconf, msg):
        print('Error when logging %s: %s' % (logconf.name, msg))

    def _write_out_log_data(self):
        global timetick, ot_position, ot_attitude, cf_twr, cf_tdoa, cf_attitude, cf_kalman, cf_mhe
        
        file.write('{}'.format(timetick))

        if log_ot_position:
            file.write(', {}'.format(ot_position[0]))
            file.write(', {}'.format(ot_position[1]))
            file.write(', {}'.format(ot_position[2]))
        if log_ot_attitude:
            file.write(', {}'.format(ot_attitude[0]))
            file.write(', {}'.format(ot_attitude[1]))
            file.write(', {}'.format(ot_attitude[2]))
        if log_cf_twr:
            file.write(', {}'.format(cf_twr[0]))
            file.write(', {}'.format(cf_twr[1]))
            file.write(', {}'.format(cf_twr[2]))
            file.write(', {}'.format(cf_twr[3]))
        if log_cf_tdoa:
            file.write(', {}'.format(cf_tdoa[0]))
            file.write(', {}'.format(cf_tdoa[1]))
            file.write(', {}'.format(cf_tdoa[2]))
            file.write(', {}'.format(cf_tdoa[3]))
            file.write(', {}'.format(cf_tdoa[4]))
            file.write(', {}'.format(cf_tdoa[5]))
        if log_cf_attitude:
            file.write(', {}'.format(cf_attitude[0]))
            file.write(', {}'.format(cf_attitude[1]))
            file.write(', {}'.format(cf_attitude[2]))
        if log_cf_kalman:
            file.write(', {}'.format(cf_kalman[0]))
            file.write(', {}'.format(cf_kalman[1]))
            file.write(', {}'.format(cf_kalman[2]))
        if log_cf_mhe:
            file.write(', {}'.format(cf_mhe[0]))
            file.write(', {}'.format(cf_mhe[1]))
            file.write(', {}'.format(cf_mhe[2]))

        file.write('\n')

    def _connection_failed(self, link_uri, msg):
        print('Connection to %s failed: %s' % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        print('Connection to %s lost: %s' % (link_uri, msg))

    def _disconnected(self, link_uri):
        print('Disconnected from %s' % link_uri)
        self.is_connected = False


def followSetpoints(crazyflie, setpoints):
    global ot_position

    i = 0
    current_setpoint = setpoints[i]
    crazyflie.commander.send_position_setpoint(current_setpoint[0], current_setpoint[1], current_setpoint[2], current_setpoint[3])    
    while True:
        try:
            dist_from_setpoint = np.sqrt((ot_position[0]-current_setpoint[0])**2 + (ot_position[1]-current_setpoint[1])**2 + (ot_position[2]-current_setpoint[2])**2)
            if dist_from_setpoint < 0.1:
                i = i+1
                current_setpoint = setpoints[i]
                print("New setpoint: ({},{},{})".format(current_setpoint[0],current_setpoint[1],current_setpoint[2]))
            
            crazyflie.extpos.send_extpos(ot_position[0],ot_position[1],ot_position[2])
            crazyflie.commander.send_position_setpoint(current_setpoint[0], current_setpoint[1], current_setpoint[2], current_setpoint[3])
            time.sleep(0.05)

            if i == len(setpoints)-1:
                print("Reached end of sequence \n")
                time.sleep(3)
                break

        except KeyboardInterrupt:
            print("stop")
            break

def hover(crazyflie, setpoint, time):
    global ot_position

    try:
        crazyflie.commander.send_position_setpoint(setpoint[0],setpoint[1],setpoint[2],setpoint[3])

        hovertime = 0
        while hovertime < time:
            crazyflie.extpos.send_extpos(ot_position[0],ot_position[1],ot_position[2])
            dist_from_setpoint = np.sqrt((ot_position[0]-current_setpoint[0])**2 + (ot_position[1]-current_setpoint[1])**2 + (ot_position[2]-current_setpoint[2])**2)
            if dist_from_setpoint < 0.1:
                hovertime = hovertime + 0.05
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("stop")




if __name__ == '__main__':

    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)
    cf = Crazyflie(rw_cache='./cache')
    le = LoggingExample(cf, cf_uri)
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)
    i = 0
    current_setpoint = setpoints[i]
    cf.commander.send_position_setpoint(current_setpoint[0], current_setpoint[1], current_setpoint[2], current_setpoint[3])
    while True:
        try:
            dist_from_setpoint = np.sqrt((ot_position[0]-current_setpoint[0])**2 + (ot_position[1]-current_setpoint[1])**2 + (ot_position[2]-current_setpoint[2])**2)
            if dist_from_setpoint < 0.1:
                i = i+1
                current_setpoint = setpoints[i]
                print("New setpoint: ({},{},{})".format(current_setpoint[0],current_setpoint[1],current_setpoint[2]))
            
            cf.extpos.send_extpos(ot_position[0],ot_position[1],ot_position[2])
            cf.commander.send_position_setpoint(current_setpoint[0], current_setpoint[1], current_setpoint[2], current_setpoint[3])
            time.sleep(0.05)

            if i == len(setpoints)-1:
                print("Reached end of sequence \n")
                time.sleep(3)
                break

        except KeyboardInterrupt:
            print("stop")
            break

    print("exiting... \n")
    cf.close_link()
    file.close()


