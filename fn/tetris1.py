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
gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
p        = np.tile(1.0, (3, config.N_PIXELS//2))
p_filt2  = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.1, alpha_rise=0.99)
rtim = 0
arx = np.linspace(0,14,15).astype(int)
ary = np.linspace(0,49,50).astype(int)
ar_wave0 = np.ones((15,50))
ar_wave1 = np.ones((15,50))
ar_wave2 = np.ones((15,50))

phw = 10
rtim3 =0
coo = np.array([1,1,1]).astype(float)
xdiv = 14
ydiv = 49
abc = 0
dcr = 0
kz = 0

arby = np.zeros((config.N_PIXELS//50,50))
rr = rn.randint(2,13)
ry = rn.randint(2,47)
#xxs = np.array([rr, rr+1, rr-1]).astype(int)
#yys = np.array([ry, ry, ry]).astype(int)
xxs = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
yys = np.zeros((1,config.N_PIXELS//50)).astype(int)
yys2 = np.zeros((1,config.N_PIXELS//50)).astype(int)+49
yys3 = np.zeros((1,config.N_PIXELS//50)).astype(int)+24

it = 0
trig1 = 0
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
en1 = 0
coll = np.array([24, 25, 26, 124, 125, 74, 75, 174]) #column of the net
#coll = np.array([24, 25, 26, 122, 123, 124, 125, 74, 75, 76, 77, 174]) #column of the net
tip = np.array([-1, 1, 1, -1, 1, -1, 1, -1]) #rement so that column moves together
#tip = np.array([-1,  1,  1, 1, -1, 1, -1, 1, -1, 1, -1 ,1, -1]) #increment so that column moves together

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
pix = int((config.N_PIXELS / 2) - 1)
odds = np.linspace(1,pix,pix//2).astype(int)
evens = np.linspace(0,pix-1,pix//2).astype(int)

cnt1 = 0
phum = np.array([0,25/3,2*25/3])
trig1 = 0
cnt3 = 0
clr = pallette.pal(0)
rtim4 = 0
p_prev = 0
#a = np.zeros((1,350))
mn = np.zeros((1,350))
c = np.zeros((1,350))
qe1 = 25
qe2 = 26
qe3 = 27
qew1 = np.linspace(0,pix, qe1).astype(int)
qew2 = np.linspace(0,pix, qe2).astype(int)
qew3 = np.linspace(0,pix, qe3).astype(int)
sthresh = .2
ewb = 25 #Speeed of our bump
ph2 = 20
kz2 = 0
x = 0
qq = 0
qq2 = 0
hg = 0
cnt2 = 0
sl = 0
cnt4 = 0
per = np.array([0,0,0])

#Tetris
cnt1t = 0
cnt3t = 0
tethresh = 5
class tetris1:
    def tetris1(y):
        global p, rty, cnt1t, odds, phum, evens, pix, cnt3t, clr, tethresh
        y2 = y**2
        gain.update(y2)
        y2 /= gain.value
        y2 *= 255.0
        m2 = np.mean(y2[28:])
        
        cnt1t +=1
        cnt3t+=1
        div = len(p[0,:]) // len(clr[:,0]) // 2
        
        
        
        
        if cnt3t>5:
            p[:,evens] = 0
            for i in np.linspace(0,len(clr[:,0])-1,len(clr[:,0])).astype(int):
                p[0,odds[(i)*div:(i+1)*div]] = clr[i,0]
                p[1,odds[(i)*div:(i+1)*div]] = clr[i,1]
                p[2,odds[(i)*div:(i+1)*div]] = clr[i,2]
            evens = np.linspace(0,pix,rn.randint(17,178)).astype(int)
            cnt3t = 0
        if cnt1t>20 and m2<tethresh:# and cnt3>5:
            p[:,odds] = 0
            odds = np.linspace(0,pix,rn.randint(17,178)).astype(int)
            for i in np.linspace(0,len(clr[:,0])-1,len(clr[:,0])).astype(int):
                p[0,evens[(i)*div:(i+1)*div]] = clr[i,0]
                p[1,evens[(i)*div:(i+1)*div]] = clr[i,1]
                p[2,evens[(i)*div:(i+1)*div]] = clr[i,2]
            cnt3t = 0
        if cnt3t>25:
            tethresh *=.95
            print("Tetris Thresh Down")
            
        if cnt1t>=20 and m2<tethresh:
            p[:, 1:] = p[:, :-1]
            if cnt1t>35:
                cnt1t = 0
                clr = pallette.pal(0)
                evens = np.linspace(0,pix,rn.randint(17,178)).astype(int)
            
        return np.concatenate((p, p[:, ::-1]), axis=1) #mirrored color symmetry