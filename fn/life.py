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


rtim  = 0

rtim4 = 0


coo  = np.array([1,1,1]).astype(float) #initialize color array, r, g, b
xdiv = 14
ydiv = 49
abc  = 0
dcr  = 0
kz   = 0

# arby = np.zeros((config.N_PIXELS//50,50))
arby = np.zeros((20,50))
arby2 = np.zeros((40,25))
rr = rn.randint(2,13)
ry = rn.randint(2,47)

xxs = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
yys = np.zeros((1,config.N_PIXELS//50)).astype(int)
yys2 = np.zeros((1,config.N_PIXELS//50)).astype(int)+49
yys3 = np.zeros((1,config.N_PIXELS//50)).astype(int)+24
SS = config.N_PIXELS-1
coll2 = np.linspace(0,SS-100,rn.randint(50,150)).astype(int)
jit = 0
fwd = 1
sl = 0
ccn = 0
fwd2 = 1
qq2 = 0
qq = 0
hg = 0
ffi = 0.3
thresh7 = 3
oods = np.linspace(1,config.N_PIXELS-1,config.N_PIXELS//2).astype(int)
fl = 0


###### Life globals #######

gen_counter = 0

old_pairs = []

all_pairs = {}
cntr = 0
for x_pos in range(40):
    for y_pos in range(25):
        all_pairs[cntr] = (x_pos,y_pos)
        cntr += 1

### random seed ###
glob_gen_x = [np.random.randint(40) for i in range(200)]
glob_gen_y = [np.random.randint(25) for i in range(200)]

old_gen_x_random = [i for i in glob_gen_x]
old_gen_y_random = [i for i in glob_gen_y]

### set seed ###

# Lightning
init_list_x = [i for i in range(0,40)]
init_list_y = [i for i in range(0,25)]
init_list_x_rev = [i for i in range(39,-1,-1)]
init_list_y_rev = [i for i in range(0,25)]

lightning_x = list(np.repeat(init_list_x,len(init_list_y)))
lightning_y = list(np.repeat(init_list_y,len(init_list_x)))
lightning_x_rev = list(np.repeat(init_list_x_rev,len(init_list_y)))
lightning_y_rev = list(np.repeat(init_list_y_rev,len(init_list_x)))

lightning_x.extend(lightning_x_rev)
lightning_y.extend(lightning_y_rev)

# Triads 1
glob_pattern_x = [19,19,18,17,20,21,17,21,18,20,17,21]
glob_pattern_y = [11,12,13,13,13,13,14,14,15,15,15,15]

triad_1_x = []
triad_1_y = []

offset=0
for xi in range(-10,20,10):
    for yi in range(-10,10,6):
        triad_1_x.extend([xi + x for x in glob_pattern_x])
        triad_1_y.extend([yi + y for y in glob_pattern_y])
    offset+=1

# Triads 2
glob_pattern_x = [19,19,18,17,20,21,17,21,19,17,21]#,16,22]#,15,23]#,14,24]#,14,24,14,24,14,24]
glob_pattern_y = [11,12,13,13,13,13,14,14,15,15,15]#,16,16]#,16,16]#,17,17]#,16,16,15,15,14,14]

triad_2_x = []
triad_2_y = []

offset=0
for xi in range(-10,20,10):#range(-10,18,20):
    for yi in range(-10,10,6):
        triad_2_x.extend([xi + x for x in glob_pattern_x])
        triad_2_y.extend([yi + y for y in glob_pattern_y])
    offset+=1

# uncomment for lightning
#old_gen_x_det = [i for i in lightning_x]
#old_gen_y_det = [i for i in lightning_y]

# # uncomment for triad 1
old_gen_x_det = [i for i in triad_1_x]
old_gen_y_det = [i for i in triad_1_y]

# # uncomment for triad 2
# old_gen_x_det = [i for i in triad_2_x]
# old_gen_y_det = [i for i in triad_2_y]
phase_offran = np.pi/6
phase_off    = np.pi/12
gc = 0
gc2 = 0
class life:
    def deterministic(y):
        global p, rtim, arx, ary, rtim4, coo, xdiv, ydiv, arby, arby2, abc, dcr, xxs, yys, yys2, yys3, oods, fl, \
        old_pairs, all_pairs, gen_counter, old_gen_x_det, old_gen_y_det, phase_off, gc
        
        
       
        arby_life = np.zeros((40,25))
        
        if gen_counter == 0 or gc == 0:
            for cur_ix in range(len(old_gen_x_det)):
                cur_pair = (old_gen_x_det[cur_ix],old_gen_y_det[cur_ix])
                old_pairs.append(cur_pair)
            for ix in range(len(old_pairs)):
                arby_life[old_pairs[ix]] = 1    
        else:
            new_pairs = []
            for key,value in all_pairs.items():
                x_val = value[0]
                y_val = value[1]
                radi_x = [x_val-1, x_val-1, x_val-1, x_val  , x_val  , x_val+1, x_val+1, x_val+1]
                radi_y = [y_val-1, y_val  , y_val+1, y_val-1, y_val+1, y_val-1, y_val  , y_val+1]
                live_cell = False
                sum_neigh = 0
                
                if value in old_pairs:
                    live_cell = True

                for radi_ix in range(len(radi_x)):
                    if (radi_x[radi_ix], radi_y[radi_ix]) in old_pairs:
                            sum_neigh += 1

                if live_cell == True and sum_neigh in [2,3]:
                    new_pairs.append(value)

                if live_cell == False and sum_neigh == 3:
                    new_pairs.append(value)

            for ix in range(len(new_pairs)):
                arby_life[new_pairs[ix]] = 1

            old_pairs = []
            for cur_ix in range(len(new_pairs)):
                cur_pair = new_pairs[cur_ix]
                old_pairs.append(cur_pair)
        
        red_val = .5*np.sin(np.pi*gen_counter/20)+.5
        green_val = .5*np.sin(np.pi*gen_counter/20 + 2*np.pi/3 + phase_off)+.5
        blue_val = .5*np.sin(np.pi*gen_counter/20+ 4*np.pi/3 + 2*phase_off)+.5
        
        p[0,:] = flatMatHardMode(arby_life) * red_val * 255
        p[1,:] = flatMatHardMode(arby_life) * green_val * 255
        p[2,:] = flatMatHardMode(arby_life) * blue_val * 255
        
        gen_counter += 1
        gc += 1
  
        if gc>=100:
            gc = 0
            phase_off += np.pi/4
        return p