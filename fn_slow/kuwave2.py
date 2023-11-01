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
p7       = np.tile(1.0, (3, config.N_PIXELS))
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

mat_map   = 1
xn        = 14
yn        = 49
upcnt     = 0
class kuwave2:
    def kuwave2(y):
        global p7, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, mat_map, \
        coo6, coo7, coo8
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        cyc = 0
        
        rtim +=1
        rtim3+=1
        bthe+=1

        xsp  = 2*np.sin(rtim*np.pi/10)+6
        ysp  = 1*(np.sin(np.pi*rtim/50))+3
        nuu = (.5*np.sin(rtim/10)+.5)*2 + 10 #+5
        #nuu = 10

        for x in arx:
            ar_wave0[x,ary] =(np.sin(rtim/nuu + x/xsp + ary/ysp))*255
            
            coo3[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x)+.75)*(.5*np.sin(rtim/(4*np.pi) /nuu           )+.5)**.5
            coo4[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x+2*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi)/nuu + 2*np.pi/3)+.5)**.5
            coo5[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x+4*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi)/nuu + 4*np.pi/3)+.5)**.5
            nn=1
            coo6[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x)+.75)*(.5*np.sin(rtim/(4*np.pi) /nuu           )+.5)**.5
            coo7[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x+2*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi)/nuu + 2*np.pi/3)+.5)**.5
            coo8[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x+4*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi)/nuu + 4*np.pi/3)+.5)**.5

        #if rtim<100:    
        a1 = ar_wave0*coo3
        a2 = ar_wave0*coo4
        a3 = ar_wave0*coo5
        a4 = ar_wave0*coo6
        a5 = ar_wave0*coo7
        a6 = ar_wave0*coo8
       
#         if mat_map == 1:
        #flipping left right so that waves come from top left corner, since other wave functions go to top right corner. 
        p7[2,:] = viz_mf.flatMat(a4) + viz_mf.flatMat(np.fliplr(a1)) + .5*viz_mf.flatMat(a1)   #+viz_mf.flatMat(a4)
        p7[1,:] = viz_mf.flatMat(a5) + viz_mf.flatMat(np.fliplr(a2))+ .5*viz_mf.flatMat(a2)#+viz_mf.flatMat(a5)
        p7[0,:] = viz_mf.flatMat(a6) + viz_mf.flatMat(np.fliplr(a3))+ .5*viz_mf.flatMat(a3)#+viz_mf.flatMat(a6)

        p7 = gaussian_filter1d(p7, sigma=2)
        p7 = 255*p7/np.max(p7)
        #p=np.fliplr(p)
        return p7