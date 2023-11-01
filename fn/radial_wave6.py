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

class radial_wave6:
    def radial_wave6(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv
        
        rtim +=2
        rtim3+=2
        tt = np.transpose(((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1))
        
        ar_wave0[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ ary/12)+1)*255
        ar_wave1[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ 2*rtim/200*ary**2/13/3)+1)*255
        ar_wave2[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ 2*rtim/200*ary**2/13)+1)*255
        p[0,:] = ar_wave0.flatten()
        p[1,:] = ar_wave1.flatten()
        p[2,:] = ar_wave2.flatten()

        ppm = np.mean(p[:,:])
        xdiv = (.5*np.sin(rtim/50)+.5)*100 + 2-14/2
        ydiv = (.5*np.sin(rtim/50)+.5)*49 + 7-49/2    
        p = gaussian_filter1d(p, sigma=2)
        return p