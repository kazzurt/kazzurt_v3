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
import quadratize
from viz_mf import flatMatHardMode

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
p        = np.tile(1.0, (3, config.N_PIXELS))
arx      = np.linspace(0,len(p[0,:])//50,15).astype(int)
ary      = np.linspace(0,49,50).astype(int)



# rainfall globals init
trip_reset = True
arby_loc = np.zeros((config.ARX,config.ARY))
n_points = np.random.poisson(15)

init_x = [np.random.randint(config.ARX) for i in range(n_points)]
init_y = [config.ARY-1 for i in range(n_points)]

new_x = init_x
new_y = init_y

x_old = new_x
y_old = new_y

pas = 0
rtim = 0
coo      = np.array([1,1,1]).astype(float) #initialize color array, r, g, b
class rainfall: 
    def rainfall(y):
        global p, rtim, rtim4, coo, xdiv, ydiv, arby2, abc, dcr, xxs, yys, yys2, yys3, oods, trip_reset, arby_loc, x_old, new_x, y_old, new_y, n_points, pas
        
        pas+=1
        
        if pas>=2:
            pas = 0
            rtim+=1
            
            n_points = 50
            #time.sleep(.02)
     
            arby_loc[x_old,y_old] = 0
            arby_loc[new_x,new_y] = 255
            x_old = new_x
            y_old = new_y

            new_x = [x for x in new_x]
            new_y = [max(y-np.random.randint(1,5),-1) for y in new_y]

            new_x = [new_x[i] for i in range(len(new_x)) if new_y[i] > -1]
            new_y = [new_y[i] for i in range(len(new_y)) if new_y[i] > -1]

            add_rain = np.random.exponential(1)

            if add_rain > 0.5:
                n_points = np.random.poisson(15)
                new_rain_x = [np.random.randint(40) for i in range(n_points)]
                new_rain_y = [24 for i in range(n_points)]

                new_x.extend(new_rain_x)
                new_y.extend(new_rain_y)

            coo[0] = (.5*np.sin(rtim/10)+.5)**.5
            coo[1] = (.5*np.sin(rtim/10+2*np.pi/3)+.5)**.5
            coo[2] = (.5*np.sin(rtim/10+4*np.pi/3)+.5)**.5
            
            p[0,:] = coo[0]*quadratize.flatMatQuads(arby_loc)
            p[1,:] = coo[1]*quadratize.flatMatQuads(arby_loc)
            p[2,:] = coo[2]*quadratize.flatMatQuads(arby_loc)

#             if rtim4>2:
#                 p[0,oods] = p[1,oods]
#                 p[1,oods] = p[2,oods]
#                 p[2,oods] = p[0,oods]
#             if rtim4>4:
#                 rtim4 = 0
            p[0,:] = gaussian_filter1d(p[0,:], sigma=.5)
            p[1,:] = gaussian_filter1d(p[1,:], sigma=.5)
            p[2,:] = gaussian_filter1d(p[2,:], sigma=.5)
        return p
