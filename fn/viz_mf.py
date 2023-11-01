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

def flatMat(pixel_mat):
    flattened_mat = pixel_mat[0,:].tolist()
    for i in range(1,pixel_mat.shape[0]):
        init_row = pixel_mat[i,:].tolist()
        rev_row = list(reversed(init_row))
        if i%2 == 1:
            flattened_mat.extend(rev_row)
        else:
            flattened_mat.extend(init_row)
    return np.array(flattened_mat)

def flatMatHardMode(pixel_mat):
    flattened_mat = []
    #flattened_mat = pixel_mat[0,:].tolist()
    ref_dict = {}
    n_rows = pixel_mat.shape[0]
    n_cols = pixel_mat.shape[1]
    col_range = range(n_cols)
    keys = range(int(np.ceil(pixel_mat.shape[0]/2)))
    max_val = 0
    for ikey in keys:
        if max_val+1 < n_rows:
            if ikey%2 == 0:
                dict_entry = [max_val,max_val+1]
            else:
                dict_entry = [max_val+1,max_val]
        else:
            dict_entry = [max_val]
        ref_dict[ikey] = dict_entry
        max_val = max_val + 2

    for key, value in ref_dict.items():
        modrem = key%2
        dict_entry_len = len(value)
        if dict_entry_len>1:
            init_row = value[0]
            zip_row  = value[1]
            row1_list = pixel_mat[init_row,:]
            row2_list = pixel_mat[zip_row,:]
            
            if modrem == 1:
                row1_list = list(reversed(row1_list))  #not using col1_list or col2_list
                row2_list = list(reversed(row2_list))

            for mat_col in col_range:
                flattened_mat.extend([row1_list[mat_col]])
                flattened_mat.extend([row2_list[mat_col]])
        else:
            flattened_mat.extend(pixel_mat[value[0],:])
                        
    return np.array(flattened_mat)

class mf:
    def spaceship2(y):
        global p, rtim, arx, ary, rtim4, coo, xdiv, ydiv, arby, arby2, abc, dcr, xxs, yys, yys2, yys3, oods, fl
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        abc+=1
        rtim+=1
        
        if abc>0: 
            abc = 0
            
            if dcr == 0:
                arby[xxs,yys] = 0
                #arby[xxs,yys2] = 0
                #arby[xxs,yys3] = 0
                yys += 2
                
                yys3 -= 2
            elif dcr == 1:
                arby[xxs,yys] = 0
                #arby[xxs,yys2] = 0
                #arby[xxs,yys3] = 0
                yys -= 2
                yys2 += 2
                yys3 += 2
            if np.max(yys)>=48: #np.max(xxs)>= 12 or 
                dcr = 1
                #fl = 1
            elif np.min(yys)<=1: #np.min(xxs)<= 2 or 
                dcr = 0
                #fl = 0
            arby[xxs,yys] = 255
            arby[xxs,yys3] = 255
            
            
#         coo[0] = (.5*np.sin(rtim/30)+.5)**.5
#         coo[1] = (.5*np.sin(rtim/30+30/3)+.5)**.5
#         coo[2] = (.5*np.sin(rtim/30+2*30/3)+.5)**.5
#         coo[0] = (.5*np.sin(rtim*np.pi/40)+.5)**.5
#         coo[1] = (.5*np.sin(rtim*np.pi/40+np.pi/3)+.5)**.5
#         coo[2] = (.5*np.sin(rtim*np.pi/40+2*np.pi/3)+.5)**.5
        
        coo[1] = (.5*np.sin(rtim/(2*np.pi)/5            )+.5)**.5 
        coo[2] = (.5*np.sin(rtim/(2*np.pi)/5 + 2*np.pi/3)+.5)**.5
        coo[0] = (.5*np.sin(rtim/(2*np.pi)/5 + 4*np.pi/3)+.5)**.5
        if fl == 0:
            p[0,:] = coo[0]*flatMat(arby)#.flatten()
            p[1,:] = coo[1]*flatMat(arby)#.flatten()
            p[2,:] = coo[2]*flatMat(arby)#.flatten()
        else:
            p[0,:] = coo[0]*arby.flatten()
            p[1,:] = coo[0]*arby.flatten()
            p[2,:] = coo[0]*arby.flatten()
        #p[0,:] = coo[1]*flatMat(arby2)
        #p[1,:] = coo[2]*flatMat(arby2)
        #p[2,:] = coo[0]*flatMat(arby2)
        
        sig = np.sin(rtim/25)+1
        p[0,:] = gaussian_filter1d(p[0,:], sigma=sig)#**1.5
        p[1,:] = gaussian_filter1d(p[1,:], sigma=sig)#**1.5
        p[2,:] = gaussian_filter1d(p[2,:], sigma=sig)#**1.5
        print(np.max(p))
        return p
    def spaceship(y):
        global p, rtim, arx, ary, rtim4, coo, xdiv, ydiv, arby, abc, dcr, xxs, yys, yys2, yys3, oods
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        abc+=1
        rtim+=1
        
        player1_x = xxs
        player1_y = yys
        
        
        if np.mean(y[2*len(y)//3::])>0 and abc>2 or abc>20: #on at 5 
            abc=0
            
            if dcr == 0:
                arby[player1_x,player1_y] = 0
#                 arby[xxs,yys2] = 0
#                 arby[xxs,yys3] = 0
                #xxs += 1
                player1_y = [i+1 for i in player1_y]
#                 yys2-=2
#                 yys3-=2
            elif dcr == 1:
                arby[player1_x,player1_y] = 0
#                 arby[xxs,yys2] = 0
#                 arby[xxs,yys3] = 0
                #xxs -= 1
                player1_y = [i-1 for i in player1_y]
#                 yys2+=2
#                 yys3+=2
            if np.max(yys)>=48: #np.max(xxs)>= 12 or 
                dcr = 1
            elif np.min(yys)<=1: #np.min(xxs)<= 2 or 
                dcr = 0
                rtim4+=1
            arby[xxs,yys] = 255


        coo[0] = (.5*np.sin(rtim/30)+.5)**.5
        coo[1] = (.5*np.sin(rtim/30+30/3)+.5)**.5
        coo[2] = (.5*np.sin(rtim/30+2*30/3)+.5)**.5
        
        p[0,:] = coo[0]*flatMat(arby)#.flatten()
        p[1,:] = coo[1]*flatMat(arby)#.flatten()
        p[2,:] = coo[2]*flatMat(arby)#.flatten()

        sig = np.sin(rtim/10)+2 
#         p[0,:] = gaussian_filter1d(p[0,:], sigma=sig)**1.5
#         p[1,:] = gaussian_filter1d(p[1,:], sigma=sig)**1.5
#         p[2,:] = gaussian_filter1d(p[2,:], sigma=sig)**1.5
        
        return p
    
    
    def rainfall():
        global p, rtim, arx, ary, rtim4, coo, xdiv, ydiv, arby2, abc, dcr, xxs, yys, yys2, yys3, oods, trip_reset, arby_loc, x_old, new_x, y_old, new_y, n_points
        
#         y = y**2
#         gain.update(y)
#         y /= gain.value
#         y *= 255.0
        abc+=1
        rtim+=1
        
        n_points = 100
        time.sleep(.02)
 
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
        
        p[0,:] = coo[0]*flatMatHardMode(arby_loc)
        p[1,:] = coo[1]*flatMatHardMode(arby_loc)
        p[2,:] = coo[2]*flatMatHardMode(arby_loc)

        if rtim4>2:
            p[0,oods] = p[1,oods]
            p[1,oods] = p[2,oods]
            p[2,oods] = p[0,oods]
        if rtim4>4:
            rtim4 = 0
        p[0,:] = gaussian_filter1d(p[0,:], sigma=.5)
        p[1,:] = gaussian_filter1d(p[1,:], sigma=.5)
        p[2,:] = gaussian_filter1d(p[2,:], sigma=.5)
        return p
    
    def sunrise(y):

        global p, rtim, arx, ary, rtim4, coo, xdiv, ydiv, arby2, abc, dcr, xxs, yys, yys2, yys3, oods, arby_new, center_x, center_y, mirror_val, iris_x, iris_y, \
               disperse, disperse_count, og_min, center_x, center_y, sun_dict, loop_counter, new_max, arby_sun, gre, arby_sky, blu
        
        og_min+=1
        og_max = og_min+2

        new_min = og_min
        new_max = og_max
        arby_new = np.zeros((40,25))
        time.sleep(0.25)

        if og_min <= 12: #rising sun

            keys = range(11,19+9)
            for ikey in keys:
                loop_counter += 1
                if ikey <= 19:
                    displace = int(ikey-10)
                    max_inc = 1
                    min_inc = -1
                    n_ys = displace+1
                    if displace % 2 == 1:
                        new_min = new_min + min_inc
                    else:
                        new_max = new_max + max_inc
                    val_range = range(new_min, new_max+2)
                    dict_entry = [i for i in val_range]
                    sun_dict[ikey] = dict_entry
                else:
                    displace = int(ikey - 10 - mirror_val)
                    max_inc = -1
                    min_inc = 1
                    mirror_val+=2
                    n_ys = displace+1
                    if displace % 2 == 1:
                        new_max = new_max + max_inc
                    else:
                        new_min = new_min + min_inc
                    val_range = range(new_min, new_max+2)
                    dict_entry = [i for i in val_range]
                    sun_dict[ikey] = dict_entry

#             print(new_min)
#             print(new_max)
            iris_x = []
            iris_y = []
            gre += 8
            blu +=.05
            for key, value in sun_dict.items():
                iris_x.extend([key for i in range(len(value)) if value[i]>-1])
                iris_y.extend([val for val in value if val > -1])
            
        elif og_min == 13:
            arby_sun[iris_x,iris_y] = 1
        else:
            if disperse < disperse_count:
                #arby_new = np.zeros((40,25))
                iris_x = [x + np.random.randint(-3,4) for x in iris_x]
                iris_y = [y + np.random.randint(-3,4) for y in iris_y]

                iris_x = [0 if x < 0 else x for x in iris_x]
                iris_y = [0 if y < 0 else y for y in iris_y]
                iris_x = [39 if x > 39 else x for x in iris_x]
                iris_y = [24 if y > 24 else y for y in iris_y]
                
                
                # arby_new[pupil_x,pupil_y] = 1

#                 color_x, color_y, white_x, white_y, p_output, output_mat = viz_graph_free(input_mat = arby_new,mat_struc = 'new')

#                 plt.scatter(color_x, color_y, c='blue')
#                 plt.scatter(white_x, white_y, c='#c8c8c8')
#                 plt.show()

                disperse += 1
            else:
                my_rands = [np.random.randint(2) for i in range(len(iris_x))]
                iris_x = [iris_x[i] for i in range(len(iris_x)) if my_rands[i]==1]
                iris_y = [iris_y[i] for i in range(len(iris_y)) if my_rands[i]==1]
        
        arby_new[iris_x,iris_y] = 1
        #arby_new += arby_sun
        arby_sky = np.ones((40,25))
        arby_sky -=arby_new
        
        coo[0] = 247 #r
        coo[1] = 50 + gre #g
        coo[2] = 50 #b
        coosun = [245,200,50]
        coosky = [50, 50+gre, 245]
        #print(coo[1])
        p[0,:] = coo[0]*flatMatHardMode(arby_new) + coosun[0]*flatMatHardMode(arby_sun) + blu*coosky[0]*flatMatHardMode(arby_sky)#.flatten()
        p[1,:] = coo[1]*flatMatHardMode(arby_new) + coosun[1]*flatMatHardMode(arby_sun) + blu*coosky[1]*flatMatHardMode(arby_sky)#.flatten()
        p[2,:] = coo[2]*flatMatHardMode(arby_new) + coosun[2]*flatMatHardMode(arby_sun) + blu*coosky[2]*flatMatHardMode(arby_sky)#.flatten()
        if rtim4>2:
            p[0,oods] = p[1,oods]
            p[1,oods] = p[2,oods]
            p[2,oods] = p[0,oods]
        if rtim4>4:
            rtim4 = 0
        sig = np.sin(rtim/10)+2 
#         p[0,:] = gaussian_filter1d(p[0,:], sigma=sig)**1.5
#         p[1,:] = gaussian_filter1d(p[1,:], sigma=sig)**1.5
#         p[2,:] = gaussian_filter1d(p[2,:], sigma=sig)**1.5
        
        return p
    
    
    def sunsAndStars(y):

        global p, rtim, arx, ary, rtim4, coo, xdiv, ydiv, arby2, abc, dcr, xxs, yys, yys2, yys3, oods, arby_new, center_x, center_y, mirror_val, iris_x, iris_y, \
               disperse, disperse_count, og_min, center_x, center_y, sun_dict, loop_counter, new_max, arby_sun, gree, arby_sky, bluu, star_x, star_y, reset_rise, sunset_min
        
        og_min+=1
        og_max = og_min+2

        new_min = og_min
        new_max = og_max
        arby_new = np.zeros((40,25))
        time.sleep(0.25)

        if og_min <= 12: #rising sun
            new_min = og_min
            new_max = og_max
            keys = range(11,19+9)
            for ikey in keys:
                loop_counter += 1
                if ikey <= 19:
                    displace = int(ikey-10)
                    max_inc = 1
                    min_inc = -1
                    n_ys = displace+1
                    if displace % 2 == 1:
                        new_min = new_min + min_inc
                    else:
                        new_max = new_max + max_inc
                    val_range = range(new_min, new_max+2)
                    dict_entry = [i for i in val_range]
                    sun_dict[ikey] = dict_entry
                else:
                    displace = int(ikey - 10 - mirror_val)
                    max_inc = -1
                    min_inc = 1
                    mirror_val+=2
                    n_ys = displace+1
                    if displace % 2 == 1:
                        new_max = new_max + max_inc
                    else:
                        new_min = new_min + min_inc
                    val_range = range(new_min, new_max+2)
                    dict_entry = [i for i in val_range]
                    sun_dict[ikey] = dict_entry

#             print(new_min)
#             print(new_max)
            iris_x = []
            iris_y = []
            gree += 8
            bluu +=.05
            for key, value in sun_dict.items():
                iris_x.extend([key for i in range(len(value)) if value[i]>-1])
                iris_y.extend([val for val in value if val > -1])
            star_x = [x for x in iris_x]
            star_y = [y for y in iris_y]
            
        elif og_min > 12:
            sunset_min-=1
            sunset_max = sunset_min+2
            new_min = sunset_min
            new_max = sunset_max
            keys = range(11,19+9)
            for ikey in keys:
                loop_counter += 1
                if ikey <= 19:
                    displace = int(ikey-10)
                    max_inc = 1
                    min_inc = -1
                    n_ys = displace+1
                    if displace % 2 == 1:
                        new_min = new_min + min_inc
                    else:
                        new_max = new_max + max_inc
                    val_range = range(new_min, new_max+2)
                    dict_entry = [i for i in val_range]
                    sun_dict[ikey] = dict_entry
                else:
                    displace = int(ikey - 10 - mirror_val)
                    max_inc = -1
                    min_inc = 1
                    mirror_val+=2
                    n_ys = displace+1
                    if displace % 2 == 1:
                        new_max = new_max + max_inc
                    else:
                        new_min = new_min + min_inc
                    val_range = range(new_min, new_max+2)
                    dict_entry = [i for i in val_range]
                    sun_dict[ikey] = dict_entry

#             print(new_min)
#             print(new_max)
            iris_x = []
            iris_y = []
            gree -= 8
            bluu -=.05
            for key, value in sun_dict.items():
                iris_x.extend([key for i in range(len(value)) if value[i]>-1])
                iris_y.extend([val for val in value if val > -1])
                
#         else:
            if disperse < disperse_count:
                #arby_new = np.zeros((40,25))
                star_x = [x + np.random.randint(-3,4) for x in star_x]
                star_y = [y + np.random.randint(-3,4) for y in star_y]

                star_x = [0 if x < 0 else x for x in star_x]
                star_y = [0 if y < 0 else y for y in star_y]
                star_x = [39 if x > 39 else x for x in star_x]
                star_y = [24 if y > 24 else y for y in star_y]
                
                arby_new[star_x,star_y] = 1
                # arby_new[pupil_x,pupil_y] = 1

#                 color_x, color_y, white_x, white_y, p_output, output_mat = viz_graph_free(input_mat = arby_new,mat_struc = 'new')

#                 plt.scatter(color_x, color_y, c='blue')
#                 plt.scatter(white_x, white_y, c='#c8c8c8')
#                 plt.show()

                disperse += 1
            else:
                if disperse >= disperse_count and reset_rise is False:
                    my_rands = [np.random.randint(2) for i in range(len(star_x))]
                    star_x = [star_x[i] for i in range(len(star_x)) if my_rands[i]==1]
                    star_y = [star_y[i] for i in range(len(star_y)) if my_rands[i]==1]
                    arby_new[star_x,star_y] = 1
                    reset_rise = len(star_x) < 2
                else:
                    og_min = -7
                    arby_new = np.zeros((40,25))
                    arby_sun = np.zeros((40,25))
                    center_x = 19
                    center_y = 13

                    sunset_min = 14

                    mirror_val = 2
                    sun_dict = {}
                    iris_x = []
                    iris_y = []
                    star_x = []
                    star_y = []
                    loop_counter = 0
                    disperse = 0
                    disperse_count = 20
                    reset_rise = False
                    new_max = 0
                    #
                    gree  = 0
                    bluu  = 0
        arby_new[iris_x,iris_y] = 1
        #arby_new += arby_sun
        arby_sky = np.ones((40,25))
        arby_sky -=arby_new
        
        coo[0] = 247 #r
        coo[1] = 50 + gree #g
        coo[2] = 50 #b
        coosun = [245,200,50]
        coosky = [50, 50+gree, 245]

        p[0,:] = coo[0]*flatMatHardMode(arby_new) + coosun[0]*flatMatHardMode(arby_sun) + blu*coosky[0]*flatMatHardMode(arby_sky)#.flatten()
        p[1,:] = coo[1]*flatMatHardMode(arby_new) + coosun[1]*flatMatHardMode(arby_sun) + blu*coosky[1]*flatMatHardMode(arby_sky)#.flatten()
        p[2,:] = coo[2]*flatMatHardMode(arby_new) + coosun[2]*flatMatHardMode(arby_sun) + blu*coosky[2]*flatMatHardMode(arby_sky)#.flatten()
        if rtim4>2:
            p[0,oods] = p[1,oods]
            p[1,oods] = p[2,oods]
            p[2,oods] = p[0,oods]
        if rtim4>4:
            rtim4 = 0
        sig = np.sin(rtim/10)+2 
#         p[0,:] = gaussian_filter1d(p[0,:], sigma=sig)**1.5
#         p[1,:] = gaussian_filter1d(p[1,:], sigma=sig)**1.5
#         p[2,:] = gaussian_filter1d(p[2,:], sigma=sig)**1.5
        
        return p
    
    def fireworks(y):
        global p, rtim, arx, ary, rtim4, coo, xdiv, ydiv, arby, abc, dcr, xxs, yys, yys2, yys3, oods, firework_global_dict, center_x, center_y, launches, \
               launch_i_list, max_rise_list, init_x_list, max_scatter_list, fwork_tail_list, scat_i_list, pop_x, pop_y, init_trail_x, init_trail_y,r_list,g_list,b_list, fw
        
        fw += 1
        if fw >1:  #Acts as a pause without pausing the run
            fw = 0
            base_color_list = [50,250]
            
            firework_dict = {}
            color_dict = {}
            
            arby_fwork_r = np.zeros((40,25))
            arby_fwork_g = np.zeros((40,25))
            arby_fwork_b = np.zeros((40,25))

            add_work = np.random.randint(5)

            if add_work <= 1:
                launches += 1
                
                launch_i_list.extend([0])

                max_rise_work = [np.random.randint(8,23)]
                max_rise_list.extend(max_rise_work)

                init_x_work = [np.random.randint(4,37)]
                init_x_list.extend(init_x_work)

                fwork_tail = [np.random.randint(1,5)]
                fwork_tail_list.extend(fwork_tail)

                scat_i_list.extend([-1])

                max_scatter_list.extend([np.random.randint(2,9)])
                
                col_choice = np.random.randint(4)
                if col_choice == 0:
                    r_val = 250
                    g_val = 50 + np.random.randint(201)
                    b_val = 50 + np.random.randint(-50,201)
                elif col_choice == 1:
                    b_val = 250
                    r_val = 50 + np.random.randint(201)
                    g_val = 50 + np.random.randint(-50,201)
                elif col_choice == 2:
                    g_val = 250
                    b_val = 50 + np.random.randint(201)
                    r_val = 50 + np.random.randint(-50,201)
                else:
                    r_val = np.random.randint(250)
                    g_val = np.random.randint(250)
                    b_val = np.random.randint(250)
                    
                r_val = [r_val]
                g_val = [g_val]
                b_val = [b_val]
                
                r_list.extend(r_val)
                g_list.extend(g_val)
                b_list.extend(b_val)
                

            for this_work in range(launches):
                launch_i = launch_i_list[this_work]
                max_rise = max_rise_list[this_work]
                init_x = init_x_list[this_work]
                this_work_tail = fwork_tail_list[this_work]
                scat_i = scat_i_list[this_work]
                max_scatter = max_scatter_list[this_work]
                
                r_v = r_list[this_work]
                g_v = g_list[this_work]
                b_v = b_list[this_work]

                launch_i_list[this_work]+=1
                if launch_i < max_rise:
                    init_trail_y = [max(launch_i-disp,0) for disp in range(this_work_tail)]
                    init_trail_x = [init_x for i in range(this_work_tail)]

                elif launch_i == max_rise:
                    init_trail_x = [init_x]
                    init_trail_y = [max_rise]

                elif launch_i > max_rise and launch_i <= (max_rise + max_scatter):
                    scat_i_list[this_work] = scat_i + 1
                    scat_i+=1

                    for local_scat in range(0,scat_i+1):
                        init_trail_x.extend(
                            [init_x + pop_x[i]+(-1*local_scat) 
                             if pop_x[i] < 0 else init_x + pop_x[i]-(-1*local_scat) if pop_x[i] > 0 else init_x 
                             for i in range(len(pop_x))]
                        )
                        init_trail_y.extend(
                            [max_rise + pop_y[i]+(-1*local_scat) 
                             if pop_y[i] < 0 else max_rise + pop_y[i]-(-1*local_scat) if pop_y[i] > 0 else max_rise 
                             for i in range(len(pop_y))]
                        )

                else:
                    init_trail_x = []
                    init_trail_y = []

                copy_x = [x for x in init_trail_x]
                copy_y = [y for y in init_trail_y]

                x_rang = range(len(copy_x))
                y_rang = range(len(copy_y))

                temp_trail_x = [init_trail_x[i] for i in x_rang if (copy_y[i] > -1 and copy_y[i] < 25)]
                final_out_x = [x for x in temp_trail_x if (x > -1 and x < 40)]
                temp_trail_y = [init_trail_y[i] for i in y_rang if (copy_x[i] > -1 and copy_x[i] < 40) ]
                final_out_y = [y for y in temp_trail_y if (y > -1 and y < 25) ]

                firework_dict[this_work] = (final_out_x,final_out_y) 

                color_dict[this_work] = [r_v,g_v,b_v]

            for key, value in firework_dict.items():
                arby_fwork_r[value] = color_dict[key][0]
                arby_fwork_g[value] = color_dict[key][1]
                arby_fwork_b[value] = color_dict[key][2]
                
        
    #         coo[0] = (.5*np.sin(rtim/30)+.5)**.5
    #         coo[1] = (.5*np.sin(rtim/30+30/3)+.5)**.5
    #         coo[2] = (.5*np.sin(rtim/30+2*30/3)+.5)**.5
            #print(arby_fwork_r)
            p[0,:] = flatMatHardMode(arby_fwork_r)#.flatten()
            p[1,:] = flatMatHardMode(arby_fwork_g)#.flatten()
            p[2,:] = flatMatHardMode(arby_fwork_b)#.flatten()
    #         if rtim4>2:
    #             p[0,oods] = p[1,oods]
    #             p[1,oods] = p[2,oods]
    #             p[2,oods] = p[0,oods]
    #         if rtim4>4:
    #             rtim4 = 0
    #         sig = np.sin(rtim/10)+2 
            p[0,:] = gaussian_filter1d(p[0,:], sigma=.5)
            p[1,:] = gaussian_filter1d(p[1,:], sigma=.5)
            p[2,:] = gaussian_filter1d(p[2,:], sigma=.5)
        
        return p