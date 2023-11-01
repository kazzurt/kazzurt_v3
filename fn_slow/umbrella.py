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

import kzbutfun
from fn_slow.colorwave0 import colorwave0
from fn_slow.colorwave01 import colorwave01
from fn_slow.colorwave1 import colorwave1
from fn_slow.colorwave4 import colorwave4
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

import quadratize
import PIL
from PIL import Image, ImageOps
p        = np.tile(1.0, (3, config.N_PIXELS))

img = Image.open("/home/pi/kz3/fn_slow/umbrella.png")
img = img.rotate(90)

img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
resized = Image.fromarray(np.array(img)).resize(size=(config.ARY,config.ARX))
img_arr = np.array((resized))
ovs = quadratize.funktions

def allfuns2(y, fun, functs):
    nam = '{}.{}'.format(functs[fun],functs[fun])
    return  globals()[nam](y)

class umbrella1:
    def umbrella1(y, overlay):
            global p, img_arr, ovs
            p[0,:] = quadratize.flatMatQuads(img_arr[:,:,0])
            p[1,:] = quadratize.flatMatQuads(img_arr[:,:,1])
            p[2,:] = quadratize.flatMatQuads(img_arr[:,:,2])
        
            #select overlay function
            nam = '{}'.format(ovs[overlay])
            p2 = getattr(globals()[nam],nam)(2)
            #print(ovs[overlay])
            for i in range(0,len(p[0,:])):
                if p[0,i] >50:
                    p[:,i] = p2[:,i]
                    
            return p
