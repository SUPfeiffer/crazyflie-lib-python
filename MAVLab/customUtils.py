# customUtils.py
# 
# Contains custom utility functions used in the python scripts to control the crazyflie
#
# Author: S.Pfeiffer, MAVLab

import math
import numpy.linalg as npl

# convert optitrack quaternions into crazyflie euler angles (degrees)
def quat2euler(q):
    q = q/npl.norm(q)
    pitch = math.atan2(-2*(q[1]*q[2]-q[0]*q[3]), q[0]^2-q[1]^2+q[2]^2-q[4]^2)
    roll = math.asin(2*(q[2]*q[3]+q[0]*q[1]))
    yaw = -math.atan2(-2*(q[1]*q[3]-q[0]*q[2]), q[0]^2-q[1]^2-q[2]^2+q[3]^2)

    if pitch>0:
        pitch = pitch - 180
    else:
        pitch = pitch + 180
    
    eulerAngles = [pitch, roll, yaw]

    return eulerAngles
