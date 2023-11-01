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
class spectrum:
    def spectrum(y):
        global p, gain, qq, cr, hg, hg2, sthresh, qq2, colm, qew1, qew2, qew3, qe1, qe2, qe3, per
        y = y**2
        gain.update(y)
        qq +=1
        y /= gain.value
        arq = int(.5*(np.sin(qq/50)+1)*255)
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        qq2+=1
        per0 = np.mean(y[0:8])/np.max(y[0:8])
        per1 = np.mean(y[8:16])/np.max(y[8:16])
        per2 = np.mean(y[16:24])/np.max(y[16:24])
        
        if ty>sthresh:
            hg+=1
            if hg>3:
                per = (255*np.array([per0,per1,per2]))+50
                hg = 0
            if qq>20:
                if qq<25:
                    sthresh*=1.125
                    print("Threshold up, spectrum")
                    print(sthresh)
                elif qq>50:
                    sthresh*=.875
                    print("Threshold down, spectrum")
                    print(sthresh)
                p[:,:] = 0
                qew1 = np.linspace(0,pix, qe1).astype(int)
                qew2 = np.linspace(0,pix, qe2).astype(int)
                qew3 = np.linspace(0,pix, qe3).astype(int)
                qe1+=1
                qe2+=1
                qe3+=1
                colm = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(25,75)).astype(int)
               
                qq = 0
        for i in np.array([0,1,2]).astype(int):    
            p[i,qew1] = per[i]
            p[i,qew2] = per[i]
            p[i,qew3] = per[i]
        p[0, :] = gaussian_filter1d(p[0, :], sigma=.35)
        p[1, :] = gaussian_filter1d(p[1, :], sigma=.35)
        p[2, :] = gaussian_filter1d(p[2, :], sigma=.35)
        return np.concatenate((p[:, ::-1], p), axis=1)