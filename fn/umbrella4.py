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
import pygame
import kzbutfun
from fn.colorwave0 import colorwave0
from fn.colorwave1 import colorwave1
from fn.energy_base import energy_base
cnt2 = np.array([0,0,0])
p        = np.tile(1.0, (3, config.N_PIXELS))

U = np.array([16, 18, 20, 21, 22, 23, 24, 25, 26, 27])+49

u2 = np.array([64, 65, 66, 67, 68, 69, 70, 71, 72, 73])
U = np.append(U, u2, axis=0)
u3 = np.array(np.linspace(23,35,35-23+1).astype(int)+100)
U = np.append(U, u3, axis=0)
u4 = np.append(np.linspace(10,19,10).astype(int)+150,np.array([19+150]),axis=0)
u4 = np.append(u4,np.linspace(21,25,5).astype(int)+150,axis=0)
U = np.append(U,u4,axis=0)
u5 = np.array([24,25,26,27,28,34,35,36,37,38,39,40,41])+200
U = np.append(U,u5,axis=0)
u6 = np.array([6,7,8,9,10,11,20,21,22,23,24,25,26])+250
U = np.append(U,u6,axis=0)
u7 = np.array([23,24,25,26,27,28,29,30,31,33,35,40,41,42,43,44])+300
U = np.append(U,u7,axis=0)
u8 = np.append(np.array([2,4,5,6,7,8,9])+350,np.linspace(10,28,19).astype(int)+350,axis=0)
U = np.append(U,u8,axis=0)
u9 = np.array([5,15,17,20,21,22,23,24,25,26,29,30,31,32,33,34,37,38,39,40,41,42,43,44,45,46,47,48,49])+400
U = np.append(U,u9,axis=0)
u10 = np.append(np.array([0,2,3,4,5,6,7,8,9,10,16,17,18,19])+450,np.linspace(23,48,48-23+1).astype(int)+450,axis=0)
U = np.append(U,u10,axis=0)
u11 = np.append(np.linspace(2,26,25).astype(int)+500,np.array([29,30,31,32,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48])+500,axis=0)
U = np.append(U,u11,axis=0)
u12 = np.array([4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,27,29,30,31,32,33,34,35,37,39,41,43,45])+550
U = np.append(U,u12)
u13 = np.array([16,17,18,19,20,24,25,26,27,28,29,30,31,32,33,34,40,41,42,43,44])+600
U = np.append(U,u13)
u14 = np.array([5,6,7,8,9,10,17,19,21,22,23,24,25,30,31,32,33,34])+650
U = np.append(U,u14)
u15 = np.array([24,25,26,27,28,29,35,36,37,38,39,40,41,42])+700
U = np.append(U,u15)
u16 = np.array([9,10,11,12,13,14,15,16,18,21,22,23,24])+750
U = np.append(U,u16)
u17 = np.linspace(25,36,12).astype(int)+800
U = np.append(U,u17)
u18 = np.array([15,17,18,19,20,21,22,23,24,25])+850
U = np.append(U,u18)


cnt2 = 0
print(U)
umbrella = U
#print(umbrella)

cnt3=0

ttop2 = np.array([49 , 149, 249, 349, 449, 549,  649, 749])

ttop = np.array([50, 150, 250, 350, 450, 550, 650])

umb_thresh = 15
hit = 0
phum = np.array([0,25/3,2*25/3])

dec = 49
cnt4 = 0
ind = np.array([0,1,2])
cnt5 = 0
drop = rn.randint(0,len(ttop)-1)
drop2 = rn.randint(0,len(ttop)-1)
cnt6=0
lp2=0
olap = 1
cnt8 = 0
cnt9 = 0
trig8 = 50
it8 = 0
class umbrella4:
    def umbrella4(y):
            global p, cnt2, cnt3, umbrella, tipp, umb_thresh, hit, phum, ttop, dec, cnt4, ind, cnt5, drop, cnt6,drop2,cnt6,lp2,ttop2,olap , cnt8,cnt9,phum8, kz8, it8,trig8

            cnt5+=1
            p = .5*energy_base.energy_base(y)
           
            p[0,umbrella] = 150*(.5*np.sin(cnt5/20+np.pi/6)+.5)
            p[2,umbrella] = 200*(.5*np.sin(cnt5/20)+.5)

            return p



