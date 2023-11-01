import numpy as np
import time
from numpy import random as rn
from viz_mf import mf
import config
trs = 0
arr = np.linspace(0,99,100).astype(int)
dirr = 0
rout = 0
trs2 = 0
p44 = np.tile(1.0, (3, config.N_PIXELS))
class transition:
    
    def fade(p):
        p44*=.95
        return p
    
    def random_outage():
        global p44
        num = np.random.randint(1000,size=25)
        p44[:,num] = 0
        return p44
    
    def random_innage():
        global p44
        #p44[:,:] = 0
        num = np.random.randint(1000,size=25)
        p44[:,num] = 255
        return p44
    
    def random_colorinnage():
        global trs, p44
        trs+=1
        #p44[:,:] = 0
        p44[0,np.random.randint(1000,size=25)] = 255
        p44[1,np.random.randint(1000,size=25)] = 255
        p44[2,np.random.randint(1000,size=25)] = 255

        return p44
    
    def scrollback_fade():
        global trs, p44
        trs+=1
        num = np.linspace(0,999,1000).astype(int)
        for x in num:
            if x-trs>=0:
                p[:,x-1] = .97*p[:,x]
                p[:,x] = 0
        if trs>50:
            trs = 0
        return p
    
    def rainfall_fade(y):
        a = .99*mf.rainfall(y)
        return a
    
    def darkrain(y):
        a = -.99*mf.rainfall(y)
        return a
    
    def sparkle(p3):
        p3[0,np.random.randint(1000,size=25)]
        return a
    
    def ramp_out(p):  #BROKEN
        global trs, rout
        trs +=1
        rout += 5+trs
        p[:,:rout] = 0
        if trs>50:
            rout = 0
            trs = 0
        return p
    