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
import quadratize

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



arx       = np.linspace(0,config.ARX-1, config.ARX).astype(int)
ary       = np.linspace(0,config.ARY-1, config.ARY).astype(int)

ar_wave0  = np.ones((config.ARX,config.ARY))

bdir      = 1
nuu       = 50 #defines speed for kuwave2 (higher is slower)
mat_map   = 1
xn        = 14
yn        = 49
upcnt     = 0

class breathe:
        
    def breathe(y):
        global p, p_filt, rtim, pix, arx, ary, ar_wave0, phw, rtim3, coo, xdiv, ydiv, cy, oods, evs, xn, yn, upcnt

       
        rtim +=1/y
        rtim3+=1/y
        
           
        for x in arx:
            ar_wave0[x,ary] = ((np.sin(ary*np.pi/(config.ARY))/2 + np.sin(x*np.pi/(config.ARX))/4))*(.5*np.sin(rtim/10+2*x+ary+(bs**.4)/(15-cy))+.5)*255 
            
        if rtim3 >25:
            rtim3 = 0
            cy+=1
        coo[0] = (.5*np.sin(rtim/20)+.5)
        coo[1] = (.5*np.sin(rtim/20+30/3)+.5)
        coo[2] = (.5*np.sin(rtim/20+2*30/3)+.5)
        
        p[0,:] = coo[0]*quadratize.flatMatQuads(ar_wave0)
        p[1,:] = coo[1]*quadratize.flatMatQuads(ar_wave0)
        p[2,:] = coo[2]*quadratize.flatMatQuads(ar_wave0)
        ppm    = np.mean(p[:,:])
        if cy>1:
            p[0,oods] = p[1,oods]
            p[1,oods] = p[2,oods]
            if cy>3:
                p[1,evs] = p[2,evs]
                p[2,evs] = p[0,evs]
                if cy>7:
                    p[1,evs] = p[2,evs]
                    p[2,oods] = p[1,oods]
                    if cy>10:
                        cy = 0
        #p = gaussian_filter1d(p, sigma=.2)

        return p

    