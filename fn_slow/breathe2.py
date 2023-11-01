from __future__ import print_function
from __future__ import division
import time
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import config
import microphone
import dsp
import led
import sys
import numpy as np
from numpy import random as rn
import config
from color_pal import pallette
import viz_mf
import cmdfun
import kzbutfun

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
p_filt   = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS)), alpha_decay=0.1, alpha_rise=0.99)

pix      = config.N_PIXELS // 2 - 1
co2      = pallette.pal(0)
p        = np.tile(1.0, (3, config.N_PIXELS))
coo      = np.array([1,1,1]).astype(float)
oods     = np.linspace(1,config.N_PIXELS-1,config.N_PIXELS//2).astype(int)
evs      = np.linspace(0,config.N_PIXELS-2,(config.N_PIXELS-1)//2).astype(int)

rtim   = 0
rtim3  = 0
cnt3   = 0
cy     = 0
ard    = 1
cyc    = 0
bthe   = 0
thresh_bthe = 0.5

coo2 = coo

timeCount = 1
countUp   = True
arx       = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
ary       = np.linspace(0,49,50).astype(int)
coo3      = np.ones((config.N_PIXELS//50,50))
coo4      = np.ones((config.N_PIXELS//50,50))
coo5      = np.ones((config.N_PIXELS//50,50))
coo6      = np.ones((config.N_PIXELS//50,50))
coo7      = np.ones((config.N_PIXELS//50,50))
coo8      = np.ones((config.N_PIXELS//50,50))
ar_wave0  = np.ones((config.N_PIXELS//50,50))
ar_wave1  = np.ones((config.N_PIXELS//50,50))
ar_wave2  = np.ones((config.N_PIXELS//50,50))
bdir      = 1
nuu       = 50 #defines speed for kuwave2 (higher is slower)
mat_map   = 1
xn        = 14
yn        = 49
upcnt     = 0

class breathe2:
        


    def breathe2(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, bdir
        y = y**2
#         gain.update(y)
#         y /= gain.value
#         y *= 255.
#         ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        
        rtim +=.1
        rtim3+=.1
        bthe+=1

        num = 5*np.sin(rtim/8)+5
        
        #If our x and y division sine wave is at its peak, let's increment cy, which determines color remapping below
        #rtim3 prevents going into this function a bunch of times while num sits close to its maximum
        #bdir controls whether we're stepping cy upwards or downwards
        if num > 9 and rtim3 > 20:
            rtim3 = 0
            if bdir == 1: 
                cy+=1
            elif bdir == -1:
                cy-=1
                if cy<=0:
                    bdir = 1
        
        #This is the best wave function, dependent on loop number (rtim), x direction (i in arx) and y direction (ary)
        #Would be nice if both directions could be a matrix operation, but idk how to do that. Choosing smaller direction for the for loop
        if cyc==0:
            for i in arx:
                ar_wave0[i,ary] =  (.5*np.sin(rtim/7 + ary/(11-num) + i/(.5+num))+.5)*255
     
        #Lets make colors perfectly phased using 2pi/3 and 4pi/3, num2 in denom controls how quickly the change
        num2 = 25*np.sin(rtim/25)+50 #this is a sine between 25 and 75 with a relatively slow speed
        ctim = (2*np.pi*rtim)/25
        coo[0] = (.5*np.sin(ctim + 0    )+.5 )
        coo[1] = (.5*np.sin(ctim + 2*np.pi/3 )+.5)
        coo[2] = (.5*np.sin(ctim + 4*np.pi/3 )+.5)

        p[0,:] = coo[0]*viz_mf.flatMat(ar_wave0)
        p[1,:] = coo[1]*viz_mf.flatMat(ar_wave0)
        p[2,:] = coo[2]*viz_mf.flatMat(ar_wave0)

        if cy>1:
            #remap odd colors
            p[0,oods] = p[1,oods] 
            p[1,oods] = p[2,oods]
            if cy>3:
                #remap even colors differently
                p[1,evs] = p[2,evs]
                p[2,evs] = p[0,evs]
                if cy>5:
                    #remap odds again, but differently
                    p[1,oods] = p[2,oods]
                    p[2,oods] = p[1,oods]
                    if cy>7:
                        #remap evens again, but differently
                        p[0,evs] = p[2,evs]
                        p[2,evs] = p[0,evs]
                        if cy>9:
                            bdir=-1 #this will make us start stepping cy backwards
        
        return p

    