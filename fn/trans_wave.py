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
p      = np.tile(1.0, (3, config.N_PIXELS // 2))
gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.1, alpha_rise=0.99)
pix    = config.N_PIXELS // 2 - 1
kz = 0
kz2 = 0
k3 = 0
it = 0
ar  = np.array([21, 23, 70, 71, 72, 73,  74,  75,  76,  77,  78, 79])
ar2 = np.array([15, 17, 19, 65, 66, 67,  68,  69,  80,  81,  82, 83, 84])
yup = 1
x   =0
du = 1
ph = np.array([5, 20])
o1 = 149
o2 = 125
o3 = 240
it3 = 0
o1 = 149
o2 = 125
o3 = 240
nn = 0
trip = 0
trip2 = 0
up = np.array([172, 0, 125, 220, 100])
s = np.array([0,0,0])
y_prev = [0]
rty = 50
coll = np.array([24, 25, 26, 124, 125, 74, 75, 174]) #column of the net
#coll = np.array([24, 25, 26, 122, 123, 124, 125, 74, 75, 76, 77, 174]) #column of the net
tip = np.array([-1, 1, 1, -1, 1, -1, 1, -1]) #rement so that column moves together

class trans_wave:

    def trans_wave(y): 
        global p, p_filt, kz, ar, ar2, du, kz2, ph, x, yup, k3, it, coll, o1, o2, o3, it3, tip, trip, trip2       
        #This ones pretty different from others 
        y = np.copy(y)
        gain.update(y)
        y /= gain.value
        # Scale by the width of the LED strip
        y *= float((config.N_PIXELS // 2) - 1)
        # Map color channels according to energy in the different freq bands
        scale = 0.9
        r = 1*int(np.mean(y[:len(y) // 3]**scale))
        g = 1*int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
        b = 1*int(np.mean(y[2 * len(y) // 3:]**scale))
        # Assign color to different frequency regions
        p[0, :r] = 255.0
        p[0, r:] = 0.0
        p[1, :g] = 255.0
        p[1, g:] = 0.0
        p[2, :b] = 255.0
        p[2, b:] = 0.0
        p_filt.update(p)
        p = np.round(p_filt.value)
        # Apply substantial blur to smooth the edges
        p[0, :] = gaussian_filter1d(p[0, :], sigma=1)
        p[1, :] = gaussian_filter1d(p[1, :], sigma=1)
        p[2, :] = gaussian_filter1d(p[2, :], sigma=1)
        pl = config.N_PIXELS // 2
        trig3 = np.mean(y[:int(len(y)/4)]) #lowest 1/4
        thres3 = 50
        if kz >1: 
            
            if np.max(ar)>pl or np.max(ar2)>pl: 
                ar2 -= 1
                yup = -1
            if np.min(ar)<-pl or np.min(ar2)<-pl:
                yup = rn.randint(1,2)
            ar  = ar + yup
            ar2 = ar2 + yup
            kz = 0
            du = du+1
        arq = int(.5*(np.sin(x/20)+1)*pl)
        mm = np.mean(y[:int(len(y)/2)])/100

        tim = 50 
        p[0, arq:arq+6] = (np.sin(np.pi*x/tim)+1)       *255
        p[1, arq:arq+6] = (np.sin(np.pi*x/tim+ph[0])+1) *255
        p[2, arq:arq+6] = (np.sin(np.pi*x/tim+ph[1])+1) *255 #mm*.5*
        
        x = x+1
        kz2+=1
        #step wave
        if np.mean(y)>30 or it>0:
            p[0,k3:k3+7] = np.mean(y[0:int(len(y)/3)])/np.mean(y)*255#rn.randint(200,255)
            p[1,k3:k3+7] = np.mean(y[int(len(y)/3):int(2*len(y)/3)])/np.mean(y)*255#rn.randint(100,255)
            p[2,k3:k3+7] = np.mean(y[int(2*len(y)/3):])/np.mean(y)*255#rn.randint(100,255)
            it += 1
            if it>50:
                k3 += 8
                it = 1
        if k3>int((config.N_PIXELS / 2) - 1):
            k3=0
            it = 0
        # end step wave
        #if mm>1:
            #p[:,:] = 0
        if mm<.2:
            #p[:,:] = 0
            p[0,coll] = o1
            p[1,coll] = o2
            p[2,coll] = o3
            it3 +=1
                
            if trig3>thres3 or it3>10:
                
                coll = coll + tip
                it3 = 0
                if coll[0] == -25:
                    tip = -tip
                    trip   = 1
                    trip2 += 1
                    o1 = rn.randint(100,250)
                    o2 = rn.randint(100,250)
                    o3 = rn.randint(100,250)
                if coll[0] == 24 and trip == 1:
                    tip = -tip
                    trip = 0
                    trip2 +=1
                    o1 = rn.randint(1,250)
                    o2 = rn.randint(1,250)
                    o3 = rn.randint(1,250)
        #Change up the mapping symmetry
        p2 = np.reshape(p[0,:],(config.N_PIXELS//50//2,50))
        p3 = np.reshape(p[1,:],(config.N_PIXELS//50//2,50))
        p4 = np.reshape(p[2,:],(config.N_PIXELS//50//2,50))
        
        p2 = np.transpose(p2)
        p3 = np.transpose(p3)
        p4 = np.transpose(p4)
        
        p2 = p2.flatten()
        p3 = p3.flatten()
        p4 = p4.flatten()
        p[0,:] = p2
        p[1,:] = p3
        p[2,:] = p4
        
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin