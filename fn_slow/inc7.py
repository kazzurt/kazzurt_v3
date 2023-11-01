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
gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix    = config.N_PIXELS // 2 - 1

fwddd = 1
ghh = 0
gh2 =0
jk = 0
qq = 0
thresh_inc = .2
cr = 0
hg2 = 0
hg = 0
fg = 0
class inc7:
#     def __init__(self, incvar1, incvar2):
#         self.mainloop = 
    
    def inc7(y):
        global p, qq, cr, hg, hg2, thresh_inc, pix, fg
        
        
        fg +=1
        if fg>4:
            p[0,np.random.randint(len(p[0,:]),size=5)] = rn.randint(150,255)
            p[1,np.random.randint(len(p[0,:]),size=5)] = rn.randint(150,255)
            p[2,np.random.randint(len(p[0,:]),size=5)] = rn.randint(150,255)
            fg = 0
#             p[0,np.random.randint(len(p[0,:]),size=1)] = 0
#             p[1,np.random.randint(len(p[0,:]),size=1)] = 0
#             p[2,np.random.randint(len(p[0,:]),size=1)] = 0
        
    
#         p[0, :] = gaussian_filter1d(p[0, :], sigma=1)
#         p[1, :] = gaussian_filter1d(p[1, :], sigma=1)
#         p[2, :] = gaussian_filter1d(p[2, :], sigma=1)
        return p


    
 
