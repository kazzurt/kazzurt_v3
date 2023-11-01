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
bb = np.linspace(0,config.N_PIXELS//100-1,config.N_PIXELS//100).astype(int)
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
#coll = np.array([24, 25, 26, 122, 123, 124, 125, 74, 75, 76, 77, 174]) #column of the net
tip = np.array([-1, 1, 1, -1, 1, -1, 1, -1]) #rement so that column moves together
#tip = np.array([-1,  1,  1, 1, -1, 1, -1, 1, -1, 1, -1 ,1, -1]) #increment so that column moves together
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.3, alpha_rise=0.99)
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

cnt1 = 0
phum = np.array([0,25/3,2*25/3])
trig1 = 0

class energies_exp:
    
    def energy_base(y):
        global p, cntb, cntb2, base_e, base_o, up_e, up_o, bb, upp, eth
        y = np.copy(y)
        gain.update(y)
        y /= gain.value
        y *= float((config.N_PIXELS // 2) - 1)

        scale = 0.9
        cntb+=1
        cntb2+=1
        etr = np.mean(y)/np.max(y)

        rrr = 1*int(np.mean(y[:len(y) // 3]**scale))//3
        ggg = 1*int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))//3
        bbb = 1*int(np.mean(y[2 * len(y) // 3:]**scale))//3
        
        if upp == 1:# and etr>eth: 
            for i in bb:
                p[0, base_e[i]:base_e[i]+rrr] = int(.5*(np.sin(cntb/25+0+base_e[i]/10)+1)*255)
                p[1, base_e[i]:base_e[i]+ggg] = int(.5*(np.sin(cntb/25+25/3+2*base_e[i]/10)+1)*255)
                p[2, base_e[i]:base_e[i]+bbb] = int(.5*(np.sin(cntb/25+25*2/3+3*base_e[i]/10)+1)*255)
                
                p[0, up_e[i]+rrr:up_e[i]+50] = int(.5*(np.sin(cntb/25+0+up_e[i]/10)+1)*255)
                p[1, up_e[i]+ggg:up_e[i]+50] = int(.5*(np.sin(cntb/25+25/3+2*up_e[i]/10)+1)*255)
                p[2, up_e[i]+bbb:up_e[i]+50] = int(.5*(np.sin(cntb/25+25*2/3+3*up_e[i]/10)+1)*255)
                
                p[0, base_e[i]+rrr:base_e[i]+50] = 0
                p[1, base_e[i]+ggg:base_e[i]+50] = 0
                p[2, base_e[i]+bbb:base_e[i]+50] = 0
                
                p[0, up_e[i]:up_e[i]+rrr] = 0
                p[1, up_e[i]:up_e[i]+ggg] = 0
                p[2, up_e[i]:up_e[i]+bbb] = 0
            cntb2 = 0
        p_filt.update(p)
        p = np.round(p_filt.value)
        p[0, :] = gaussian_filter1d(p[0, :], sigma=3)
        p[1, :] = gaussian_filter1d(p[1, :], sigma=3)
        p[2, :] = gaussian_filter1d(p[2, :], sigma=3)
        
        return np.concatenate((p, p[:, ::-1]), axis=1)
    
   
    
    

