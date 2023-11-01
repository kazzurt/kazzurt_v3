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
#import image_proc
import PIL
from PIL import Image, ImageOps
import quadratize 
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.3, alpha_rise=0.99)
im       = np.tile(1.0, (3, config.N_PIXELS))
p        = np.tile(1.0, (3, config.N_PIXELS))


img = Image.open("sublogo.png") #load in da image ya
img = img.rotate(90)
#img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
#resized = Image.fromarray(np.array(img)).resize(size=(50,20)) #resize for 20 strands of 50 pix
resized = Image.fromarray(np.array(img)).resize(size=(config.ARX,config.ARY)) #resize for 20 strands of 50 pix


cnt = 0
cnt2 =0
ph = [0,0, 0]#2*np.pi/3,4*np.pi/3]
mu = 1
class subs2:
    def subs2(y):
        global p, p_filt, im, cnt, cnt2, ph, mu, resized, im, img

        L1 = config.ARX
        L2 = config.ARY

        img_arr = np.array((resized))
        xd = 2*np.pi*np.sin(cnt/10) + 2*np.pi + .2
        yd = 2*np.pi*np.sin(cnt/10 + np.pi/2*(1+cnt/20)) + 2*np.pi + .2
        for i in range(L1):
            for j in range(L2):
                img_arr[i,j] = (.5*np.sin(cnt/20 + (i-L2/2)/xd - (j-L1/2)/yd)+.5)*img_arr[i,j]
                

        im[0,:] = quadratize.flatMatQuads(img_arr[:,:,0])  
        im[1,:] = quadratize.flatMatQuads(img_arr[:,:,1])
        im[2,:] = quadratize.flatMatQuads(img_arr[:,:,2])

        for j in range(0,3): #flip every other column up/down cause wiring
            for i in range(0,19):
                if i%2 == 0:
                    im[j,i*50:i*50+50] = np.fliplr([im[j,i*50:i*50+50]])[0]     
        cnt+=1
        cnt2+=1

        for i in range(0,3):
            p[i,:] = im[i,:] #*(np.abs(np.sin(cnt/30+ph[i])))



        return p


