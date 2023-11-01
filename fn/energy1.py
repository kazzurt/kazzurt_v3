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
pix    = config.N_PIXELS // 2 - 1
s = np.array([0, 0, 0])
a = np.zeros((1,32))
m1 = 0
m2 = 0
c5 = 0
ar  = np.array([21, 23, 70, 71, 72, 73,  74,  75,  76,  77,  78, 79])
ar2 = np.array([15, 17, 19, 65, 66, 67,  68,  69,  80,  81,  82, 83, 84])
yup = 1
x=0
du = 0
kz2 = 0

cntb =0
cntb2 = 21
base_e = np.linspace(0,config.N_PIXELS-100,config.N_PIXELS//100).astype(int)
base_o = np.linspace(99,config.N_PIXELS-1, config.N_PIXELS//100).astype(int)
up_e = np.linspace(50,config.N_PIXELS-50,config.N_PIXELS//100).astype(int)
up_o = np.linspace(49,config.N_PIXELS-51,config.N_PIXELS//100).astype(int)
bb = np.linspace(0,9,10).astype(int)
upp = 1
eth = .2

thresh = .2
hg = 0
qq2 = 0
colm = np.linspace(0,config.N_PIXELS // 2-1, 50).astype(int)
sz_on = 0
kz = 0
en1 = 0
c1 = 0
c2 = 0
red = rn.randint(100,255)
gr = rn.randint(100,255)
bl = rn.randint(100,255)
red2 = rn.randint(0,255)
gr2 = rn.randint(0,255)
bl2 = rn.randint(0,255)
u2 = rn.randint(0,120)
w2 = rn.randint(1,3)
it = 0
k3 = 1
it2 = 0
v1 = 0
v2 = 0
v3 = 0
v4 = 0

coll = np.array([24, 25, 26, 124, 125, 74, 75, 174]) #column of the net
tip = np.array([-1, 1, 1, -1, 1, -1, 1, -1]) #rement so that column moves together
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.1, alpha_rise=0.99)
it3 = 0
o1 = 149
o2 = 125
o3 = 240
nn = 0
trip2 = 0
up = np.array([172, 0, 125, 220, 100])
s = np.array([0,0,0])
y_prev = [0]
rty = 50
pix = (config.N_PIXELS / 2) - 1
odds = np.linspace(1,374,374//2).astype(int)
evens = np.linspace(1,372,374//2).astype(int)
qq = 0
cnt1 = 0
phum = np.array([0,25/3,2*25/3])
trig1 = 0

class energy1:
    
    def energy1(y):
        #This one is ridiculous kurt. seriously, clean this shit up
        global p, pnorm, k3, cinc, kk, kz, c1, u, w, red, bl, gr, c2, u2, w2, red2, bl2, gr2, \
               y_prev, it, t1, it2, point, v1, v2, v3, v4, en1, coll, tip, it3, trip, o1, o2, o3, trip2, nn, up1, s
        
        kz +=1 #LOOP COUNTER
        #print(type(kz))
        y = np.copy(y)
        y2 = y**2
        gain.update(y2)
        y /= gain.value
        
        # Scale by the width of the LED strip
        y *= float((config.N_PIXELS // 2) - 1)
        #print(y)
        # Map color channels according to energy in the different freq bands
        scale = 0.7
        r = int(np.mean(y[:len(y) // 3]**scale))
        g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
        b = int(np.mean(y[2 * len(y) // 3:]**scale))
        
        #triggers - are you triggered yet?
        #taking the mean of the frequency vs power bins, number of bins = len(y)
        trig1 = np.mean(y[int(len(y)/2):]) #Lower half
        trig2 = np.mean(y[:len(y)]) #higher half
        trig3 = np.mean(y[:int(len(y)/4)]) #lowest 1/4
        trig4 = np.mean(y[int(len(y)/4):]) #highest 1/4 
        
        #thresholds - we should normalize these eventually (or does the mic already do that)
        thres1 = 20
        thres2 = 150
        thres3 = 50
        
        y2 /= gain.value
        y2 *= 255.0
        r = int(np.max(y2[:len(y2) // 3]))
        g = int(np.max(y2[len(y2) // 3: 2 * len(y2) // 3]))
        b = int(np.max(y2[2 * len(y2) // 3:]))
        
        #Inward outward wave
        if trig1 > thres1 or en1>0: #Trigger on the lower half or if its already going
            if en1<150: #en1 is just a counter
                p[:, 1:] = p[:, :-1] #Scrolling effect window
                #p *= 0.98 
                p = gaussian_filter1d(p, sigma=0.2)
                p[0, 0] = g
                p[1, 0] = r
                p[2, 0] = b
                en1+=1
            if en1>149:
                p[:,:] = 0
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
                    if trip2>2:
                        en1 = 0
        # End Inward Outward wave
        
        # START  in2out train
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
        #End In2out train
            
        # Semi Standard Energy #
        if trig2>thres2:#140: #Trigger on the upper half 
            p[0, 0:r] = rn.randint(150,255) 
            p[0, r:] = 0
            p[1, 0:g] = rn.randint(150,255) 
            p[1, g:] = 0
            p[2, 0:b] = rn.randint(150,255)
            p[2, b:] = 0
            p_filt.update(p)
            p = np.round(p_filt.value)
            en1 = 0
        #End Semi Standard Energy

        #Upward rain, up = np.array([172, 0, 125, 220, 100])
        if trig3 > 50 or trig4 > 50: #Trigger easily
            if up[1]>30:
                up[0] -=50
                up[1] = 0
                up[2:4] = rn.randint(150,250)

            else:
                up[1] +=1
            if up[0]<25:
                up[0] = 174
            
            p[0,up[0]-2:up[0]+2] = up[2]*255
            p[1,up[0]-2:up[0]+2] = up[3]*255
            p[2,up[0]-2:up[0]+2] = up[4]*255
        
        y_norm = y/np.max(y)
       
        if np.mean(y[0:8]>20) or it2 == 0:
            if it2 == 0:
                point = rn.randint(125,170)
                v1 = np.mean(y_norm[:int(len(y_norm)/3)])*255
                v2 = np.mean(y_norm[int(len(y_norm)/3):int(len(y_norm)*2/3)])*255
                v3 = np.mean(y_norm[int(len(y_norm)*2/3):])*255
                it2 = 1
                v4 = rn.randint(3,7)
            else:
                it2+=1
                if it2>50:
                    it2 = 0
            p[0,point:point+v4] = v1
            p[1,point:point+v4] = v2
            p[2,point:point+v4] = v3
        
        # START  in2out train
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
        #End In2out train
       
        # High note block tracker
        if trig2 > 50: 
            if c1<1: #Define color and shape
                u = rn.randint(0,120)
                w = rn.randint(1,3)
                red = rn.randint(0,255)
                gr = rn.randint(0,255)
                bl = rn.randint(0,255)
                nn=rn.randint(3,9) #Used to be 5 only
            p[0,u:u+nn] = red
            p[1,u:u+nn] = gr
            p[2,u:u+nn] = bl
            c1+=1
            if c1>50: #when to reset
                c1=0
                
        #Low block tracker        
        if trig1 > 20:
            if c2<1:
                u2 = rn.randint(0,120)
                w2 = rn.randint(1,3)
                red2 = rn.randint(0,255)
                gr2 = rn.randint(0,255)
                bl2 = rn.randint(0,255)
            p[0,u2] = 255
            p[1,u2] = 255 
            p[2,u2] = 255 
            nn2=3
            p[0,u2:u2+nn2] = red2
            p[1,u2:u2+nn2] = gr2
            p[2,u2:u2+nn2] = bl2
            c2+=1
            
            if c2>50:
                c2=0
                
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
