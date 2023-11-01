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
from fn.colorwave01 import colorwave01
from fn.colorwave1 import colorwave1
from fn.colorwave02 import colorwave02
from fn.radial_wave import radial_wave
from fn.radial_wave2 import radial_wave2
from fn.radial_wave3 import radial_wave3
from fn.radial_wave4 import radial_wave4
from fn.radial_wave5 import radial_wave5
from fn.radial_wave6 import radial_wave6
from fn.radial_pal import radial_pal
from fn.bessel1 import bessel1
from fn.bessel2 import bessel2
from fn.bessel3 import bessel3
from fn.bessel4 import bessel4
import quadratize
import PIL
from PIL import Image, ImageOps
p        = np.tile(1.0, (3, config.N_PIXELS))

img = Image.open("sublogo.png")
img = img.rotate(90)
t = 0
img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
#resized = Image.fromarray(np.array(img)).resize(size=(config.ARY,config.ARX))

ovs = ["bessel1","bessel2","colorwave01","colorwave02","radial_wave","radial_wave3","radial_wave4","radial_wave5","radial_wave6"]
def allfuns2(y, fun, functs):
    nam = '{}.{}'.format(functs[fun],functs[fun])
    return  globals()[nam](y)

class subs_new:
    def subs_new(y):
            global p, ovs, t, img
            img = img.rotate(np.sin(t/5)+1)
            resized = Image.fromarray(np.array(img)).resize(size=(config.ARY,config.ARX))
            img_arr = np.array((resized))
            p[0,:] = quadratize.flatMatQuads(img_arr[:,:,0])
            p[1,:] = quadratize.flatMatQuads(img_arr[:,:,1])
            p[2,:] = quadratize.flatMatQuads(img_arr[:,:,2])
            
            t +=1
            #select overlay function
#             nam = '{}'.format(ovs[overlay])
#             p2 = getattr(globals()[nam],nam)(y)
#             
#             for i in range(0,len(p[0,:])):
#                 if p[0,i] >10:
#                     p[:,i] = p2[:,i]
                    
            return p

