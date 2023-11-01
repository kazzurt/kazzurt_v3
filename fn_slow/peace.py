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
from fn.colorwave22 import colorwave22
#import image_proc
import PIL
from PIL import Image, ImageOps
from fn.energy_base2 import energy_base2
import quadratize
from fn_slow.colorwave0 import colorwave0
from fn_slow.colorwave01 import colorwave01
from fn_slow.colorwave1 import colorwave1
from fn_slow.colorwave02 import colorwave02
from fn_slow.radial_wave import radial_wave
from fn_slow.radial_wave2 import radial_wave2
from fn_slow.radial_wave3 import radial_wave3
from fn_slow.radial_wave4 import radial_wave4
from fn_slow.radial_wave5 import radial_wave5
from fn_slow.radial_wave6 import radial_wave6
from fn_slow.radial_pal import radial_pal
from fn_slow.pointwave import pointwave
from fn_slow.bessel1 import bessel1
from fn_slow.bessel2 import bessel2
from fn_slow.bessel3 import bessel3
from fn_slow.bessel4 import bessel4

p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.3, alpha_rise=0.99)
im       = np.tile(1.0, (3, config.N_PIXELS))
p        = np.tile(1.0, (3, config.N_PIXELS))
im2      = np.tile(1,(3,config.N_PIXELS))
img = Image.open("/home/pi/kz3/peace3.png") #load in da image ya
img = img.rotate(90)

#img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
#img = img.rotate(cnt)
resized = Image.fromarray(np.array(img)).resize(size=(config.ARY,config.ARX)) #resize for 20 strands of 50 pix

img_arr = np.array((resized)) 
ovs = ["colorwave01","bessel1","bessel2","colorwave02","radial_wave","radial_wave3","radial_wave4","radial_wave5","radial_wave6","pointwave"]
# im[0,:] = img_arr[:,:,0].flatten()  
# im[1,:] = img_arr[:,:,1].flatten()
# im[2,:] = img_arr[:,:,2].flatten()
# 
# for j in range(0,3): #flip every other column up/down cause wiring
#     for i in range(0,19):
#         if i%2 == 0:
#             im[j,i*50:i*50+50] = np.fliplr([im[j,i*50:i*50+50]])[0]
# for i in range(0,1000):
#     if im[2,i]>150:
#         im2[:,i] = 0
#     else:
#         im2[:,i] = im[:,i]
cnt = 0
cnt2 =0
ph = [0,0, 0]#2*np.pi/3,4*np.pi/3]
mu = 1
class peace:
    def peace(y, overlay, overlay2):
        global p, p_filt, im, cnt, cnt2, ph, mu, resized, im2,img_arr, ovs
    
        cnt+=1/y
        cnt2+=1/y
        #overlay=1
        p[0,:] = quadratize.flatMatQuads(img_arr[:,:,0])
        p[1,:] = quadratize.flatMatQuads(img_arr[:,:,0])
        p[2,:] = quadratize.flatMatQuads(img_arr[:,:,0])
        #select overlay function
        nam = '{}'.format(ovs[overlay])
        p2 = getattr(globals()[nam],nam)(y)
        nam2 = '{}'.format(ovs[overlay2])
        p3 = getattr(globals()[nam2],nam2)(y)
        #p*=(.5*np.cos(cnt/15)+.5)
        for i in range(0,len(p[0,:])):
            if (p[0,i]+p[1,i]+p[2,i])/3 > 100*np.sin(cnt/10)+150:
                p[:,i] = 255/2*np.sin(cnt/15)+255/2
            else:#if i%2 == 1:
                p[:,i] = p2[:,i]*.75#*(.5*np.cos(cnt/15)+.5)
            #elif i%1 == 0:
                #p[:,i] = p3[:,i]*.5#*(.5*np.cos(cnt/15)+.5)
        return p




