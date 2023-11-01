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

p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.3, alpha_rise=0.99)
im       = np.tile(1.0, (3, config.N_PIXELS))
p        = np.tile(1.0, (3, config.N_PIXELS))


img = Image.open("sublogo.png") #load in da image ya
img = img.rotate(90)
img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
resized = Image.fromarray(np.array(img)).resize(size=(50,20)) #resize for 20 strands of 50 pix



cnt = 0
cnt2 =0
ph = [0,0, 0]#2*np.pi/3,4*np.pi/3]
mu = 1
class subs1:
    def subs1(y):
        global p, p_filt, im, cnt, cnt2, ph, mu,resized, im,img
        num = np.sin(cnt/5)+1
        img = img.rotate(num)
        resized = Image.fromarray(np.array(img)).resize(size=(50,20)) #resize for 20 strands of 50 pix
        #resized = resized.rotate(cnt/50)
        img_arr = np.array((resized)) 
        im[0,:] = img_arr[:,:,0].flatten()  
        im[1,:] = img_arr[:,:,1].flatten()
        im[2,:] = img_arr[:,:,2].flatten()

        for j in range(0,3): #flip every other column up/down cause wiring
            for i in range(0,19):
                if i%2 == 0:
                    im[j,i*50:i*50+50] = np.fliplr([im[j,i*50:i*50+50]])[0]     
        cnt+=1
        cnt2+=1
#         if cnt2>2:
#             cnt2 = 0
#             mu = 5*np.mean(y[len(y)//2:])+.5
#             if mu > 1:
#                 mu = 1
#         print(mu)
#         Q = colorwave0.colorwave0(y) eh it didnt really add anything
        for i in range(0,3):
            p[i,:] = im[i,:]*(np.abs(np.sin(cnt/30+ph[i])))
#             for j in range(0,1000):
#                 if p[i,j] ==0:
#                     p[i,j] = Q[i,j]
        temp  = p[0,:]
        
#         if cnt/30>np.pi and cnt2/30>np.pi:
#             ph[1]+=np.pi/6
#             ph[2]+=np.pi/3
#             cnt2=0
        #p[0,:] = p[1,:]
        #p[1,:] = p[2,:]
        #p[2,:] = temp#         p_filt.update(p[0:499])
#         p = np.round(p_filt.value[0:499])  
        return p

