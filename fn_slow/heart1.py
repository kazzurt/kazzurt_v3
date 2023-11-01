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
from fn_slow.colorwave0 import colorwave0
from fn_slow.colorwave1 import colorwave1
from fn_slow.colorwave22 import colorwave22
#import image_proc
import PIL
from PIL import Image, ImageOps
from fn.energy_base2 import energy_base2
import quadratize
from fn_slow.colorwave01 import colorwave01
from fn_slow.colorwave02 import colorwave02
from fn_slow.colorwave25 import colorwave25
from fn_slow.colorwave26 import colorwave26
from fn_slow.radial_wave import radial_wave
from fn_slow.radial_wave2 import radial_wave2
from fn_slow.radial_wave3 import radial_wave3
from fn_slow.radial_wave4 import radial_wave4
from fn_slow.radial_wave5 import radial_wave5
from fn_slow.radial_wave6 import radial_wave6
from fn_slow.radial_pal import radial_pal
from fn_slow.bessel1 import bessel1
from fn_slow.bessel2 import bessel2
from fn_slow.bessel3 import bessel3
from fn_slow.bessel4 import bessel4
from fn_slow.pointwave import pointwave

p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.3, alpha_rise=0.99)
im       = np.tile(1.0, (3, config.N_PIXELS))
p        = np.tile(1.0, (3, config.N_PIXELS))
im2      = np.tile(1,(3,config.N_PIXELS))
img = Image.open("/home/pi/kz3/heart.png") #load in da image ya
img = img.rotate(90)

#img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
#img = img.rotate(cnt)
resized = Image.fromarray(np.array(img)).resize(size=(config.ARY,config.ARX)) #resize for 20 strands of 50 pix

img_arr = np.array((resized)) 
ovs = quadratize.funktions

cnt = 0
cnt2 =0
cnt3 = 0
ph = [0,0, 0]#2*np.pi/3,4*np.pi/3]
mu = 1
class heart1:
    def heart1(y, overlay):
        global p, p_filt, im, cnt, cnt2, ph, mu, resized, im2,img_arr, ovs, cnt3
    
        cnt+=1/y
        cnt2+=1/y
        cnt3+=1
        #overlay=1
        p[0,:] = quadratize.flatMatQuads(img_arr[:,:,0])
        p[1,:] = quadratize.flatMatQuads(img_arr[:,:,1])
        p[2,:] = quadratize.flatMatQuads(img_arr[:,:,2])
        #select overlay function
        nam = '{}'.format(ovs[overlay])
        p2 = getattr(globals()[nam],nam)(y)
        nn = (.25*np.sin(cnt/20)+.3)
        for i in range(0,len(p[0,:])):
            if (p[0,i]+p[1,i]+p[2,i])/3 > 100*np.sin(cnt/10)+150:
                p[:,i] = nn*p2[:,i]
        if nn<0.06 and cnt3>50:
            overlay=rn.randint(0,len(quadratize.funktions)-1)
            cnt3=0
            print(ovs[overlay])
        return p



