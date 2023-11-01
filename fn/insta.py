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
import quadratize
import PIL
from PIL import Image, ImageOps

p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.3, alpha_rise=0.99)
im       = np.tile(1.0, (3, config.N_PIXELS))
p        = np.tile(1.0, (3, config.N_PIXELS))


img = Image.open("instagram.png") #load in da image ya
img = img.rotate(90)
#img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
resized = Image.fromarray(np.array(img)).resize(size=(26,52)) #resize for 20 strands of 50 pix

cnt = 0
cnt2 =1
ph = [0,0,0]
mu = 1
class insta:
    def insta(y):
        global p, p_filt, im, cnt, cnt2, ph, mu, resized
        cnt+=1

   
        img_arr = np.array((resized)) 

        p[0,:] = quadratize.flatMatQuads(img_arr[:,:,0])
        p[1,:] = quadratize.flatMatQuads(img_arr[:,:,1])
        p[2,:] = quadratize.flatMatQuads(img_arr[:,:,2])
        for i in range(0,len(p[0,:])):
            if (p[0,i]+p[1,i]+p[2,i])/3 >200:
                p[:,i] = (.5*np.cos(cnt/15)+.5)*255
#             else:
#                 p[0,i] *= (.5*np.sin(cnt/15 )+.5)
#                 p[1,i] *= (.5*np.sin(cnt/15 )+.5)
#                 p[2,i] *= (.5*np.sin(cnt/15 )+.5)
#         for j in range(0,3): #flip every other column up/down cause wiring
#             for i in range(0,19):
#                 if i%2 == 0:
#                     im[j,i*50:i*50+50] = np.fliplr([im[j,i*50:i*50+50]])[0]
#         for i in range(0,3):
#             p[i,:] = im[i,:]*(np.abs(np.sin(cnt/50+ph[i])))
#         ph[1] +=np.pi/100
#         ph[2] +=np.pi/50

        return p


