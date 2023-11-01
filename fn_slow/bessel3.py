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
import scipy.special as sp
import quadratize
gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix      = config.N_PIXELS 
p        = np.tile(1.0, (3, config.N_PIXELS))
coo      = np.array([1,1,1]).astype(float)


pl = np.linspace(0,pix-1,pix).astype(int)*4

arx         = np.linspace(0,config.ARX-1,config.ARX).astype(int)
ary         = np.linspace(0,config.ARY-1,config.ARY).astype(int)
red_ar      = np.zeros((config.ARX,config.ARY))
gre_ar      = np.zeros((config.ARX,config.ARY))
blu_ar      = np.zeros((config.ARX,config.ARY))
tim=0
k = 2
ki = 1
ph = [0,np.pi/3,2*np.pi/3]
class bessel3:
    
    def bessel3(y):
            global p, pix, arx, ary, red_ar, gre_ar, blu_ar, pl, tim, k, ki, ph
#             y = y**2
#             gain.update(y)
#             y /= gain.value
#             y *= 255.
#             r = 2*int(np.max(y[:len(y) // 3]))
#             g = 2*int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
#             b = 2*int(np.max(y[2 * len(y) // 3:]))
#             ty = np.mean(y[len(y)//2::])
            tim+=k/y

            B = ((sp.jv(1,pl)+.4)/1.4)*255
#             for i in range(40):
#                 for j in range(25):
#                     red_ar[i,j] = (sp.jv(1,(i-20)*(j-11)/20+ (i-20)/5+(j-12)/5+tim/10)+.33)*255
#                     blu_ar[i,j] = (sp.jv(2,(i-20)*(j-11)/20+ (i-20)/5+(j-12)/5+tim/10)+.33)*255
#                     gre_ar[i,j] = (sp.jv(3,(i-20)*(j-12)/20+ (i-20)/5+(j-12)/5+tim/10)+.33)*255
            ys = 5*np.sin(tim/50)+10
            xs = 5*np.sin(tim/50)+15
            nu = 20
            for j in ary:
                red_ar[:,j] = (sp.jv(3,(arx-xs)*(j-ys)/nu+ (arx-xs)/2 + (j-ys)/5  +tim/10+ph[0])+.33)
                blu_ar[:,j] = (sp.jv(3,(arx-xs)*(j-ys)/nu+ (arx-xs)/2+(j-ys)/5 +tim/10+ph[1])+.33)
                gre_ar[:,j] = (sp.jv(3,(arx-xs)*(j-ys)/nu+ (arx-xs)/2+(j-ys)/5 +tim/10+ph[2])+.33)
                    #gre_ar = sp.jv(1,j)
#             p[0,:] = viz_mf.flatMatHardMode(red_ar)/np.max(red_ar)*(100*np.sin(tim/25+ph[0])+150)
#             p[1,:] = viz_mf.flatMatHardMode(gre_ar)/np.max(blu_ar)*(100*np.sin(tim/25+ph[1])+150)
#             p[2,:] = viz_mf.flatMatHardMode(blu_ar)/np.max(gre_ar)*(100*np.sin(tim/25+ph[2])+150)
            p[0,:] = quadratize.flatMatQuads(red_ar)/np.max(red_ar)*(100*np.sin(tim/25+ph[0])+150)
            p[1,:] = quadratize.flatMatQuads(gre_ar)/np.max(blu_ar)*(100*np.sin(tim/25+ph[1])+150)
            p[2,:] = quadratize.flatMatQuads(blu_ar)/np.max(gre_ar)*(100*np.sin(tim/25+ph[2])+150)
            #p = gaussian_filter1d(p, sigma=.3)
            ph[1] += np.pi/6/100
            ph[2] += np.pi/3/100
            #if tim>150 or tim <= 0:
                #k *= -1
                
 
            return p




