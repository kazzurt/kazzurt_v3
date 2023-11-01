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

class steve_breathe:
        
    def steve_breathe(y):
        global p, rtim, pix, arx, ary, ar_wave, ar_wave1, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        cyc = 0
        
        rtim +=1
        rtim3+=1
        bthe+=1
        if timeCount == 20:
            countUp = False
        
        if timeCount == 1:
            countUp = True
        
        if countUp:
            timeCount += 1
        else:
            timeCount -= 1
        
        for x in arx:
            ar_wave0[x,ary] = ((np.sin(ary*np.pi/((np.sin(rtim/10)+2)*5))/2
                                + np.sin(x*np.pi/((np.sin(rtim/10)+2)*25))/2))*(np.sin(rtim/10+ary/(2*np.sin(ary)+5)+x/3)+.75)*350#*(255//(timeCount/8))
            ar_wave1[x,ary] = ((np.sin((ary-5)*np.pi/((np.sin(rtim/10)+2)*5))/2
                                + np.sin(x*np.pi/((np.sin(rtim/10)+2)*25))/2))*(np.sin(rtim/10+(ary-5)/(2*np.sin((ary-5))+5)+(x+5)/3)+.75)*350#*(255//(timeCount/8))
            # ar_wave0[x,ary] = ((np.sin(ary*np.pi/(49))/2 + np.sin(x*np.pi/(14))/4))*(.5*np.sin(rtim/10+ary/5+x/5)+.5)*255
            coo3[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x)+.75)*(.5*np.sin(rtim/(4*np.pi)            )+.5)**.5
            coo4[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x+np.pi/2)+.75)*(.5*np.sin(rtim/(4*np.pi) + 2*np.pi/3)+.5)**.5
            coo5[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x+np.pi)+.75)*(.5*np.sin(rtim/(4*np.pi) + 4*np.pi/3)+.5)**.5
             
        if rtim3 >30 and ty>thresh_bthe:
            rtim3 = 0
            bthe=0
            cy+=1

        coo2[0] = (.5*np.sin(rtim/(2*np.pi)            )+.5)#**.5 #(.5*np.sin(arx)+.5)*
        coo2[1] = (.5*np.sin(rtim/(2*np.pi) + 2*np.pi/3)+.5)#**.5
        coo2[2] = (.5*np.sin(rtim/(2*np.pi) + 4*np.pi/3)+.5)#**.5
        
        a1 = ar_wave0*coo3
        a2 = ar_wave0*coo4
        a3 = ar_wave0*coo5
        a4 = ar_wave1*coo3
        a5 = ar_wave1*coo4
        a6 = ar_wave1*coo5
#         p[0,:] = a1.flatten()
#         p[1,:] = a2.flatten()
#         p[2,:] = a3.flatten()
        p[0,:] = viz_mf.flatMat(a1) + viz_mf.flatMat(a4)
        p[1,:] = viz_mf.flatMat(a2) + viz_mf.flatMat(a5)
        p[2,:] = viz_mf.flatMat(a3) + viz_mf.flatMat(a6)        
        ppm = np.mean(p[:,:])
        if cy>1:
            p[0,oods] = p[1,oods]
            p[1,oods] = p[2,oods]
            if cy>3:
                p[1,evs] = p[2,evs]
                p[2,evs] = p[0,evs]
                if cy>5:
                    p[1,oods] = p[2,oods]
                    p[2,evs] = p[1,evs]
                    if cy>7:
                        p[0,oods] = p[2,oods]
                        p[2,evs] = p[0,evs]
                        if cy>9:
                            cy = 0
        p = gaussian_filter1d(p, sigma=.5)
        return p