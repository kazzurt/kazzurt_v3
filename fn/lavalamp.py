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
import image_proc
import PIL
from PIL import Image, ImageOps
import viz_mf
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.3, alpha_rise=0.99)
im       = np.zeros((47, 3, config.N_PIXELS))
p        = np.tile(1.0, (3, config.N_PIXELS))

for k in range(1,47):
    st = "lavalamp2/Untitled_Artwork-" + str(k) + ".png"
    img = Image.open(st) #load in da image ya
    img = img.rotate(90)
#img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.transpose(method=Image.FLIP_TOP_BOTTOM)

    resized = Image.fromarray(np.array(img)).resize(size=(50,20)) #resize for 20 strands of 50 pix
    print(resized)
    img_arr = np.array((resized)) 

    im[k,0,:] = img_arr[:,:,0].flatten()  
    im[k,1,:] = img_arr[:,:,1].flatten()
    im[k,2,:] = img_arr[:,:,2].flatten()
    
    for j in range(0,3): #flip every other column up/down cause wiring
        for i in range(0,19):
            if i%2 == 0:
                im[k,j,i*50:i*50+50] = np.fliplr([im[k,j,i*50:i*50+50]])[0]

cnt = 1
cnt2 =0
ph = [0,0, 0]#2*np.pi/3,4*np.pi/3]
mu = 1
class lavalamp:
    def lavalamp(y):
        global p, p_filt, im, cnt, cnt2, ph, mu
        Q = colorwave22.colorwave22(y)
        if cnt2 == 0:
            cnt+=1
        elif cnt2 == 1:
            cnt-=1
        
        for i in range(0,3):
            p[i,:] = im[cnt,i,:]/np.max(im[i,:])*255
            for j in range(0,1000):
                if p[i,j] > 150:
                    p[i,j] = Q[i,j]
        if cnt >=46:
            cnt2=1
        elif cnt == 1:
            cnt2 = 0
            
        
        return p




