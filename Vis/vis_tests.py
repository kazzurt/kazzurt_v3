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

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
p_filt   = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS)), alpha_decay=0.1, alpha_rise=0.99)
pix      = config.N_PIXELS // 2 - 1
color    = pallette.pal(0)
co       = 0
p        = np.tile(1.0, (3, config.N_PIXELS))
testc    = 0

class tests:
    def testy(y):
        global p, testc, color, co
        #Use this to test new things
        xx = range(0,20)
        ar = np.zeros((40,25))
        ar[range(0,40),22] = 255
        #This is how you draw a straight line on the net.
        #If we figure out the flatten function test it next to this to check
#     for n in np.linspace(0,900,10).astype(int): #must count in steps of 100 to draw straight line
#         
#         p[:,n:(n+2)] = 255
#         p[:,99-n] =255 #n+(50-n)*2-1
#         p[:,98-n] =255
    #p[:,30+(50-30)*2] =255
    #print(p[0,:])
        p[:,:] = viz_mf.flatMatHardMode(ar)
        return p
    #print(len(viz_mf.flatMatHardMode(ar)))
    def palettes(y):
        global p, testc, color, co
        if testc == 0:
            color = pallette.pal(co)
            print(co)
        co+=1
        
        if co==23:
            co = 1
            
        testc+=1
        
        if testc>40:
            testc = 0
        div = len(p[0,:]) // len(color[:,0])
        for i in np.linspace(0,len(color[:,0])-1,len(color[:,0])).astype(int):
            p[0,(i)*div:(i+1)*div] = color[i,0]*.4
            p[1,(i)*div:(i+1)*div] = color[i,1]*.4
            p[2,(i)*div:(i+1)*div] = color[i,2]*.4
        return p
