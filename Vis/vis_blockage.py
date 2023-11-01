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
import cmdfun
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)

#Blockage 1 globals
blk    = 1
bla01  = np.zeros((config.N_PIXELS//50,50)).astype(int) #0 by 50 matrices
bla11  = np.zeros((config.N_PIXELS//50,50)).astype(int)
bla21  = np.zeros((config.N_PIXELS//50,50)).astype(int)
ary    = np.linspace(0,49,50).astype(int) #Y vector
bly    = np.linspace(0,49,49//2).astype(int) #Y vector, half filled
btim   = 0 #generic time
blkthr = .01
blkcn  = 0
trz    = 0
trz0   = 0
bltc   = 0
bltt   = 0
#Blockage 2 globals
evens2 = np.linspace(0,config.N_PIXELS-2,config.N_PIXELS//2).astype(int)
odds2  = np.linspace(1,config.N_PIXELS-1,config.N_PIXELS//2).astype(int)
phary  = np.array([0,2*np.pi/3,5*np.pi/3])
blcol  = np.array([.5, 1, .2])
loops  = 0

p      = np.tile(1.0, (3, config.N_PIXELS))

class blockage:

    def blk1(y):
        global p, gain, blk, bla01, bla11, bla21, bly, ary, btim, blkthr, blkcn, bltc, bltt
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        blkcn+=1
        blth = np.mean(y)/np.max(y) #[len(y)//2::]
        blt = np.mean(y[2*len(y)//3::])
        
        btim+=1
        bla01[0,:] = 255*np.sin(btim/50+ ary/10)*(.5*np.sin(btim/30)+.5)**.5
        bla11[0,:] = 255*np.sin(btim/50+ ary/10*np.pi/2)*(.5*np.sin(btim/30 + ary/np.pi/2)+.5)**.5
        bla21[0,:] = 255*np.sin(btim/50+ ary/10*np.pi)*(.5*np.sin(btim/30+ary/np.pi)+.5)**.5
        blkcn+=1
    
        if blk>len(bla01[:,0])-1:
            blk= 0
        if blth>blkthr:
        
            bla01[blk,:] = bla01[blk-1,:]
            bla11[blk,:] = bla11[blk-1,:]
            bla21[blk,:] = bla21[blk-1,:]
            #threshold changes not working cuz im drunk
            #if blkcn >50:
                #blkthr *=.9
                #print("Blk Up")
            #elif blkthr<5:
                #blkthr *=1.1
                #print("Blk Down")
                #print(blkthr)
            blkcn = 0     
        blk+=1
        
        
        p[0,:] = bla01[:,:].flatten()
        p[1,:] = bla11[:,:].flatten()
        p[2,:] = bla21[:,:].flatten()
        
        p[0,:] = gaussian_filter1d(p[0,:], sigma=2)
        p[1,:] = gaussian_filter1d(p[1,:], sigma=2)
        p[2,:] = gaussian_filter1d(p[2,:], sigma=2)
        bltt+=1
        if bltc==0:
            arw = np.linspace(0,config.N_PIXELS-1,config.N_PIXELS//4).astype(int)
            p[0,arw] = p[1,arw]
            p[1,arw] = p[2,arw]
            p[2,arw] = p[0,arw]
        if bltt>10 and blt>5:
            if bltc == 0:
                bltc = 1
            elif bltc == 1:
                bltc = 0
            bltt = 0
       
        return p
        
    def blk2(y):
        global p, gain, blk, bly, ary, btim, blkthr, blkcn, bltc, bltt, evens2, odds2, phary, blcol, loops, coms
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        blkcn+=1
        blth = np.mean(y)/np.max(y) #[len(y)//2::]
        blt = np.mean(y[2*len(y)//3::])
        btim+=1
        blkcn+=1              
        blk+=1
        bltt+=1
        if bltc==0:
            p[0,evens2] = blcol[0]*255
            p[1,evens2] = blcol[1]*255
            p[2,evens2] = blcol[2]*255
            
            p[0,odds2] = 0
            p[1,odds2] = 0
            p[2,odds2] = 0
        if bltc==1:
            p[0,evens2] = 0
            p[1,evens2] = 0
            p[2,evens2] = 0
            
            p[0,odds2] = blcol[1]*255 
            p[1,odds2] = blcol[2]*255
            p[2,odds2] = blcol[0]*255
        if blk>15:
            blcol = .5*np.sin(btim/10+phary)+.5
        
        if (bltt>15 and blt>5): #coms[275]==1: #bltt is timer, blt is audio based, coms[275] is right arrow
            if bltc == 0:
                bltc = 13
                blk=0
                bltc = 0
                loops +=2
                p[:,evens2] = 0
                p[:,odds2] = 0
                evens2 = np.linspace(0,config.N_PIXELS-2,config.N_PIXELS//2-loops).astype(int)
                odds2 = np.linspace(1,config.N_PIXELS-1,config.N_PIXELS//2-loops).astype(int)
                phary+=1
            bltt = 0
            blcol = .5*np.sin(btim/10+phary)+.5
        if loops>10:
            p[0,:config.N_PIXELS//2] = p[1,config.N_PIXELS//2::]
            p[1,:config.N_PIXELS//2] = p[2,config.N_PIXELS//2::]
            p[2,:config.N_PIXELS//2] = p[0,config.N_PIXELS//2::]
        if loops>20:
            p[0,:config.N_PIXELS//4] = p[2,3*config.N_PIXELS//4::]
            p[1,:config.N_PIXELS//4] = p[0,3*config.N_PIXELS//4::]
            p[2,:config.N_PIXELS//4] = p[1,3*config.N_PIXELS//4::]
        
            p[0,config.N_PIXELS//2:3*config.N_PIXELS//4] = p[1,config.N_PIXELS//4:config.N_PIXELS//2]
            p[1,config.N_PIXELS//2:3*config.N_PIXELS//4] = p[0,config.N_PIXELS//4:config.N_PIXELS//2]
            p[2,config.N_PIXELS//2:3*config.N_PIXELS//4] = p[2,config.N_PIXELS//4:config.N_PIXELS//2]
            
        if loops>40:
            loops = 0
        p[0,:] = gaussian_filter1d(p[0,:], sigma=.5)
        p[1,:] = gaussian_filter1d(p[1,:], sigma=.5)
        p[2,:] = gaussian_filter1d(p[2,:], sigma=.5)
        return p
