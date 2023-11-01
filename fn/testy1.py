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

p      = np.tile(1.0, (3, config.N_PIXELS))




class testy1:
    
    def testy1(y):
        #This one is ridiculous kurt. seriously, clean this shit up
        global p
        p[:,48] = 255
        p[:,49] = 255
        p[:,50] = 255
        return p

#     def testy2(y):
#         global p
#         p[:,49] = 255
#         return p  