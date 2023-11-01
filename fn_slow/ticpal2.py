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
import cmdfun
p      = np.tile(1.0, (3, config.N_PIXELS//2))
gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix    = config.N_PIXELS // 2 - 1
ar     = np.array([21, 23, 70, 71, 72, 73,  74,  75,  76,  77,  78, 79])
colm   = np.linspace(0,config.N_PIXELS-1, 50).astype(int)
colm2  = np.linspace(0,config.N_PIXELS // 2-1, 2).astype(int)

qq     = 0
qq     = 249
qq2    = 0
thresh = 0.3

cr     = 0
hg     = 0
hg2    = 0
hg3    = 0
fwdd   = 1
gau    = .6
crg    = [2,1,0]

co     = 0
colo2 = pallette.pal(0)
color = pallette.pal(0)
colm2 = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(10,25)).astype(int)
cl = 1
nu = 50 #starting number of tics for tic 2
fwd = 0 #starting direction for tic 2
tic_yn = 0
class ticpal2:
    
    def ticpal2(y):
        global p, qq, a, ar, colm, qq2, hg, thresh, colo2, co
         
        y = y**2
        gain.update(y)
        qq +=1
        y /= gain.value
        arq = int(.5*(np.sin(qq/50)+1)*255)
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        qq2+=1
        
        
        if ty>thresh or qq>35 : 
            if qq>10:
                hg+=1
                if qq<15 and hg>3:
                    hg = 0
                    thresh*=1.1
                    print('Threshold Up, ticpal2 (auto): %5.3f.' % thresh)
                elif qq>25 and hg>3:
                    hg = 0
                    thresh*=.89
                    print('Threshold Down, ticpal2 (auto): %5.3f.' % thresh)
                p[:,:] = 0
                colm = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(25,50)).astype(int)
                qq = 0
                colo2 = pallette.pal(0)
                
            
            
        CL = int(len(colo2[:,0]))
        com = len(colm)
        
        for x in np.linspace(0,CL-1, CL).astype(int): #Going through list of color pallettes
            p[0,colm[(x-1):com//x]] = colo2[x,0]
            p[1,colm[(x-1):com//x]] = colo2[x,1]
            p[2,colm[(x-1):com//x]] = colo2[x,2]
            
        if qq>15:
            p[0, :] = gaussian_filter1d(p[0, :], sigma=.5)
            p[1, :] = gaussian_filter1d(p[1, :], sigma=.5)
            p[2, :] = gaussian_filter1d(p[2, :], sigma=.5)
        return np.concatenate((p, p[:, ::-1]), axis=1)