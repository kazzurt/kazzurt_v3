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
from fn_slow.bessel1 import bessel1
from fn_slow.bessel2 import bessel2
from fn_slow.bessel3 import bessel3
from fn_slow.bessel4 import bessel4
from fn_slow.pointwave import pointwave

import quadratize
import PIL
from PIL import Image, ImageOps
p        = np.tile(1.0, (3, config.N_PIXELS))

img = Image.open("/home/pi/kz3/fn_slow/umbrella.png")
img = img.rotate(90)

img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
resized = Image.fromarray(np.array(img)).resize(size=(config.ARY,config.ARX))
img_arr = np.array((resized))
#ovs = ["colorwave01","bessel1","bessel2","colorwave02","radial_wave","radial_wave3","radial_wave4","radial_wave5","radial_wave6"]
ovs = ["colorwave01","bessel1","bessel2","colorwave02","radial_wave","radial_wave3","radial_wave4","radial_wave5","radial_wave6","pointwave"]

t = 0
def allfuns2(y, fun, functs):
    nam = '{}.{}'.format(functs[fun],functs[fun])
    return  globals()[nam](y)

class umbrella_dark:
    def umbrella_dark(y, overlay, overlay2):
            global p, img_arr, ovs, t
            p[0,:] = quadratize.flatMatQuads(img_arr[:,:,0])
            p[1,:] = quadratize.flatMatQuads(img_arr[:,:,1])
            p[2,:] = quadratize.flatMatQuads(img_arr[:,:,2])
            t+=1
            #select overlay function
            nam = '{}'.format(ovs[overlay])
            nam2 = '{}'.format(ovs[overlay2])
            p2 = getattr(globals()[nam],nam)(4)
            p3 = getattr(globals()[nam2],nam2)(4)
            for i in range(0,len(p[0,:])):
                if p[0,i] <150:
                    p[:,i] = .5*p2[:,i]
                else:
                    p[:,i] = p3[:,i]
#                     p[0,i] = (255*np.sin(t/25)+255)/2
#                     p[1,i] = (255*np.sin(t/25+2*np.pi/3)+255)/2
#                     p[2,i] = (255*np.sin(t/25+4*np.pi/3)+255)/2
            return p

