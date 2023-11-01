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

from viz_mf import flatMatHardMode
gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
p        = np.tile(1.0, (3, config.N_PIXELS))
arx      = np.linspace(0,len(p[0,:])//50,15).astype(int)
ary      = np.linspace(0,49,50).astype(int)
rtim     = 0
rtim4    = 0
coo      = np.array([1,1,1]).astype(float) #initialize color array, r, g, b
xdiv     = 14
ydiv     = 49
abc      = 0
dcr      = 0
kz       = 0

# arby = np.zeros((config.N_PIXELS//50,50))
arby     = np.zeros((20,50))
arby2    = np.zeros((40,25))
arby_sky = np.ones((40,25))
rr       = rn.randint(2,13)
ry       = rn.randint(2,47)

xxs      = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
yys      = np.zeros((1,config.N_PIXELS//50)).astype(int)
yys2     = np.zeros((1,config.N_PIXELS//50)).astype(int)+49
yys3     = np.zeros((1,config.N_PIXELS//50)).astype(int)+24
SS       = config.N_PIXELS-1
coll2    = np.linspace(0,SS-100,rn.randint(50,150)).astype(int)
jit      = 0
fwd      = 1
sl       = 0
ccn      = 0
fwd2     = 1
qq2      = 0
qq       = 0
hg       = 0
ffi      = 0.3
thresh7  = 3
oods     = np.linspace(1,config.N_PIXELS-1,config.N_PIXELS//2).astype(int)
fl       = 0
blu      = 0
bluu     = 0

# rainfall globals init
trip_reset = True
arby_loc = np.zeros((40,25))
n_points = np.random.poisson(15)

init_x = [np.random.randint(40) for i in range(n_points)]
init_y = [24 for i in range(n_points)]

new_x = init_x
new_y = init_y

x_old = new_x
y_old = new_y

# sunrise globals init
og_min     = -7
arby_new   = np.zeros((40,25))
arby_sun   = np.zeros((40,25))
center_x   = 19
center_y   = 13
sunset_min = 14
mirror_val = 2
sun_dict   = {}
iris_x     = []
iris_y     = []
star_x     = []
star_y     = []
loop_counter = 0
disperse = 0
disperse_count = 20
new_max = 0
reset_rise = False
gre = 0
gree  = 0

# Fireworks globals init
center_x = 19
center_y = 13
max_rise = np.random.randint(8,23)
init_x = np.random.randint(4,28)
launches = 1
launch_i_list = [0]
max_rise_list = [max_rise]
init_x_list = [init_x]
max_scatter_list = [6]
pop_x = [ 0,-1,-1,-1, 0, 1, 1, 1]
pop_y = [-1,-1, 0, 1, 1, 1, 0,-1]
fwork_tail_list = [np.random.randint(1,5)]
scat_i_list = [-1]
r_val = np.random.randint(250)
g_val = np.random.randint(250)
b_val = np.random.randint(250)
r_list = [r_val]
g_list = [g_val]
b_list = [b_val]
fw = 0
# end fireworks globals
class whiterain: 
    def whiterain(y):
        global p, rtim, arx, ary, rtim4, coo, xdiv, ydiv, arby2, abc, dcr, xxs, yys, yys2, yys3, oods, trip_reset, arby_loc, x_old, new_x, y_old, new_y, n_points
        
#         y = y**2
#         gain.update(y)
#         y /= gain.value
#         y *= 255.0
        abc+=1
        rtim+=1
        
        n_points = 100
        time.sleep(.05)
 
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
            n_points = np.random.poisson(10)
            new_rain_x = [np.random.randint(40) for i in range(n_points)]
            new_rain_y = [24 for i in range(n_points)]

            new_x.extend(new_rain_x)
            new_y.extend(new_rain_y)

        coo[0] = (.5*np.sin(rtim/10)+.5)**.5
        coo[1] = (.5*np.sin(rtim/10+2*np.pi/3)+.5)**.5
        coo[2] = (.5*np.sin(rtim/10+4*np.pi/3)+.5)**.5
        
        p[:,:] = flatMatHardMode(arby_loc)
        #p[1,:] = coo[1]*flatMatHardMode(arby_loc)
        #p[2,:] = coo[2]*flatMatHardMode(arby_loc)

        
        #p[0,:] = gaussian_filter1d(p[0,:], sigma=.5)
        #p[1,:] = gaussian_filter1d(p[1,:], sigma=.5)
        #p[2,:] = gaussian_filter1d(p[2,:], sigma=.5)
        return p
