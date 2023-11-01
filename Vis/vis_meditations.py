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
gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix      = config.N_PIXELS // 2 - 1
co2      = pallette.pal(0)
p        = np.tile(1.0, (3, config.N_PIXELS))
coo = np.array([1,1,1]).astype(float)
rtim = 0
rtim3 = 0
cnt3 = 0
oods = np.linspace(1,config.N_PIXELS-1,config.N_PIXELS//4).astype(int)
evs = np.linspace(0,config.N_PIXELS-2,(config.N_PIXELS-1)//4).astype(int)
cy = 0
ard = 1
cyc=0
bthe = 0
thresh_bthe = 0.5

coo2 = coo

timeCount = 1
countUp = True
arx = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
print(arx)
ary = np.linspace(0,49,50).astype(int)
coo3 = np.ones((config.N_PIXELS//50,50))
coo4 = np.ones((config.N_PIXELS//50,50))
coo5 = np.ones((config.N_PIXELS//50,50))
ar_wave0 = np.ones((config.N_PIXELS//50,50))
ar_wave1 = np.ones((config.N_PIXELS//50,50))
ar_wave2 = np.ones((config.N_PIXELS//50,50))
nuu = 50 #defines speed for kuwave2 (higher is slower)
rtim4 = 0
class meditation:

    def brackle(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, cy, oods, evs, rtim4, num

        
        rtim +=1
        rtim3+=1
        
        #if rtim4 == 0:
            #num = 
            #rtim+=
        
        rtim4+=1
        num = .5*np.sin(rtim/30)+5
        for y in ary:
            ar_wave0[arx,y] = ((np.sin(y*np.pi/(49))/2 + np.sin(arx*np.pi/(14))/4))*(.5*np.sin(rtim/num)+.5)*255
        
        if np.mean(p[:,:])<1 and rtim3 >15:
            rtim3 = 0
            cy+=1
            coo[0] = (.5*np.sin(rtim/20)+.5)**.5
            coo[1] = (.5*np.sin(rtim/20+30/3)+.5)**.5
            coo[2] = (.5*np.sin(rtim/20+2*30/3)+.5)**.5
        
            
        p[0,:] = coo[0]*ar_wave0.flatten()
        p[1,:] = coo[1]*ar_wave0.flatten()
        p[2,:] = coo[2]*ar_wave0.flatten()
        ppm = np.mean(p[:,:])
        if cy>1:
            p[0,oods] = p[1,oods]
            p[1,oods] = p[2,oods]
            if cy>3:
                p[1,evs] = p[2,evs]
                p[2,evs] = p[0,evs]
                if cy>7:
                    p[1,evs] = p[2,evs]
                    p[2,oods] = p[1,oods]  
            #p[0,oods] = p[1,oods]
            
        p = gaussian_filter1d(p, sigma=.2)
        return p
