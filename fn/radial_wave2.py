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

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix      = config.N_PIXELS // 2 - 1
arx      = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
ary      = np.linspace(0,49,50).astype(int)
ar_wave0 = np.ones((config.N_PIXELS//50,50))
ar_wave1 = np.ones((config.N_PIXELS//50,50))
ar_wave2 = np.ones((config.N_PIXELS//50,50))
evs2     = np.linspace(0,config.N_PIXELS//50-2,config.N_PIXELS//50//2).astype(int)
ods2     = np.linspace(1,config.N_PIXELS//50-1,config.N_PIXELS//50//2).astype(int)
phum     = np.array([0,25/3,2*25/3])
arby     = np.zeros((config.N_PIXELS//50,50))
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

class radial_wave2:
    def radial_wave2(y):
            global p, rtim, pix, arx, ary, phw, evs2, ods2, phw2, rtim2, phw_gap, exc
           
            rtim   +=1
            rtim2  +=1
            timing = 10
            ydi = 10
            for y in ary:
                #these sin waves need to be renormalized to fit the net better
                ar_wave0[evs2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw        )+1))*255
                ar_wave0[ods2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw + np.pi)+1))*255
                
                ar_wave1[evs2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw2 + phw2/3+phw_gap)+1))*255
                ar_wave1[ods2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw2 + phw2/3 + np.pi)+1))*255
                
                ar_wave2[evs2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw + 2*phw/3)+1))*255
                ar_wave2[ods2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw2 + 2*phw2/3 + np.pi+phw_gap)+1))*255
                
            p[0,:] = viz_mf.flatMat(ar_wave0)
            p[1,:] = viz_mf.flatMat(ar_wave1)
            p[2,:] = viz_mf.flatMat(ar_wave2)

            if rtim2 > 50 and np.mean(p[:,:])<10:
                exc-=1
                rtim2 = 0
            return p
