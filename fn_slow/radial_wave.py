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
import quadratize
gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix      = config.N_PIXELS // 2 - 1
arx      = np.linspace(0,config.ARX-1,config.ARX).astype(int)
ary      = np.linspace(0,config.ARY-1,config.ARY).astype(int)
ar_wave0 = np.ones((config.ARX,config.ARY))
ar_wave1 = np.ones((config.ARX,config.ARY))
ar_wave2 = np.ones((config.ARX,config.ARY))
evs2     = np.linspace(0,config.N_PIXELS//50-2,config.N_PIXELS//50//2).astype(int)
ods2     = np.linspace(1,config.N_PIXELS//50-1,config.N_PIXELS//50//2).astype(int)
phum     = np.array([0,25/3,2*25/3])
arby     = np.zeros((config.ARX,config.ARY))
cnt1     = 0
trig1    = 0
phw      = 10
phw2     = 5
coo      = np.array([1,1,1]).astype(float)
xdiv     = 14
ydiv     = 49
abc      = 0
dcr      = 0
coc      = 0
coc1     = 0
coc2     = 1

rtim     = 0
rtim2    = 0
rtim3    = 0

phw_gap  = 0
exc      = 8

f        = 0
co       = 1
co2      = pallette.pal(0)
p        = np.tile(1.0, (3, config.N_PIXELS))

class radial_wave:
    def radial_wave(y):
        global p, rtim, pix, arx, ary, phw, rtim3, coo, xdiv, ydiv

 
        rtim +=1/y
        rtim3+=1/y
        xph = np.abs(arx-np.mean(arx))/np.pi
        xx = np.abs(arx-14/2)/100
        
        for y in ary:
            yph = np.abs(y-np.mean(y))/100
            yy = np.abs(y-49/2)
            ar_wave0[arx,y] = (( np.sin(yy*np.pi/(ydiv))/2 + np.sin(xx*np.pi/(xdiv))/2+.5)**1)*(.5*np.sin(rtim/phw+yy/4+xx**2)+.5)*300
         
        coo[0] = (.5*np.sin(rtim/30)+.5)**.5
        coo[1] = (.5*np.sin(rtim/30+np.pi/3)+.5)**.5
        coo[2] = (.5*np.sin(rtim/30+2*np.pi/3)+.5)**.5
        
        p[0,:] = coo[0]*quadratize.flatMatQuads(ar_wave0) 
        p[1,:] = coo[1]*quadratize.flatMatQuads(ar_wave0)  
        p[2,:] = coo[2]*quadratize.flatMatQuads(ar_wave0) 
        
        ppm = np.mean(p[:,:])
        xdiv = (.5*np.sin(rtim/10)+.5)*26 + 13
        ydiv = (.5*np.sin(rtim/10)+.5)*52 + 21
        p = gaussian_filter1d(p, sigma=.2)
        return p