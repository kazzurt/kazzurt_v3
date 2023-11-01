
import time
import numpy as np
from numpy import random as rn
from array import array

try:
    import launchpad_py as launchpad
except:
    print("Launchpad error")
print("======== Launchpad ========")

lp = launchpad.LaunchpadPro()
lpmode = None
print(lp.Open( 0 ))
#print("Launchpad Pro Mk3")
lpmode = "Pro"
print("======== Launchpad ========")

def kzlaunchpad():
    #buts = lp.ButtonStateXY( mode = 'pro')
    buts = lp.ButtonStateRaw()#returnPressure = True)
    #events = lp.ButtonStateRaw( returnPressure = True )
    #pressure = []
    #if events != []:
        #if events[0] >= 255:
        #pressure1 = events[0]
        #pressure2 = events[1]
        
    if buts != []:
        return buts
    
