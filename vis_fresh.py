from __future__ import print_function
from __future__ import division
import time
import numpy as np
from numpy import random as rn
from array import array
import PIL
from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter1d
import config
import led
import sys
import dsp
import microphone
from color_pal import pallette
from Vis.vis_breathe import breathing
from fn.colorwave0 import colorwave0
from fn.colorwave01 import colorwave01
from fn.colorwave0 import colorwave0
from fn.colorwave1 import colorwave1
from fn.colorwave2 import colorwave2
from fn.colorwave3 import colorwave3
from fn.colorwave4 import colorwave4
from fn.colorwave01 import colorwave01
from fn.colorwave02 import colorwave02
from fn.colorwave22 import colorwave22
from fn.colorwave23 import colorwave23
from fn.colorwave24 import colorwave24
from fn.colorwave25 import colorwave25
from fn.colorwave26 import colorwave26
from fn.colorwave6 import colorwave6
from fn.colorwave7 import colorwave7
from Vis.vis_tic import tics
from Vis.vis_ticfull import ticsfull
from fn.radial_wave import radial_wave
from fn.radial_wave2 import radial_wave2
from fn.radial_wave3 import radial_wave3
from fn.radial_wave4 import radial_wave4
from fn.radial_wave5 import radial_wave5
from fn.radial_wave6 import radial_wave6
from fn.bessel1 import bessel1
from fn.bessel2 import bessel2
from fn.bessel3 import bessel3
from fn.bessel4 import bessel4
from Vis.vis_scroll import scroll
from fn.umbrella import umbrella1
from fn.umbrella_dark import umbrella_dark
from fn.insta import insta
from fn.fract1 import fract1
from fn.pointwave import pointwave
from fn.colorscroll import colorscroll
from Vis.vis_incs import incs
from fn.heart1 import heart1
from fn.peace import peace
import viz_mf
from gpiozero import Button
button = Button(17)

import kzbutfun
from viz_mf import mf

import cmdfun 

functs = ["colorwave01","colorwave02","colorwave25","colorwave26","colorwave7","radial_wave","radial_wave5","bessel1","bessel2","bessel3","pointwave", \
          "umbrella_dark","insta", "fract1","tic2", "breathe", "sweetscroll","scroll4","radial_wave2","radial_wave4","radial_wave6","inc","heart1","peace" \
          ]
#"colorwave26", radial_wave5 umbrella

visualization_type = sys.argv[1]

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)

"""The low-pass filter used to estimate frames-per-second"""
kz   = int(0)
k    = 0
cinc = 0
kk   = 120
c    = 0
S    = np.zeros(((config.N_PIXELS // 2) - 1))
SS   = ((config.N_PIXELS // 2) - 1)
if kz>10:
    _fps = dsp.ExpFilter(50, alpha_decay=0.2, alpha_rise=0.2)
def frames_per_second():   
    global _time_prev, _fps, kz, k
    kz += 1
    k=1
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)

def memoize(function):
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv
    return wrapper

#@memoize


#Filters n shit
r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.1, alpha_rise=0.5)
common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.99, alpha_rise=0.01)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)


p3 = np.tile(1.0, (3, config.N_PIXELS))    
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
swt  = 0

def funch(typ):
    #Random function chooser
    global p, fun, fun_total, fun_cut, lpp, tran, ct
    fun = rn.randint(1,fun_total) # last_fun+1#
    lpp +=1
    ct = 0
    tran = 0
    if typ == 0:
        print('Forced change, Function: {:.0f}'.format(fun))
    elif typ == 1:
        print('Timed change, Function: {:.0f}'.format(fun))
    print("===========")
    kz3 = 0
    if fun<=fun_cut and fun>0:
        p = np.tile(1.0, (3, config.N_PIXELS//2))
    elif fun>0:
        p = np.tile(1.0, (3, config.N_PIXELS))
    return fun
scrl = [2, 6, 12]
rwlo = [13, 14, 21]
enes = [1, 7, 10, 15]
incy = [5, 19, 20]
pong = [43, 44, 47]
spnk = [3, 16, 17, 23, 24, 25]
tiks = [8, 9, 20, 22]
mffn = [47, 48, 49, 53, 54]
brth = [28, 30, 36, 37, 38, 41, 42]
hwav = [4, 18]
cwav = [45, 46, 50, 51, 52]
rwav = [26, 29, 34, 35, 40]
blka = [31, 33]


fxns     = {}

fxns[1]  = scrl
fxns[2]  = rwlo
fxns[3]  = pong
fxns[4]  = incy
fxns[5]  = spnk
fxns[6]  = tiks
fxns[7]  = mffn 
fxns[8]  = brth 
fxns[9]  = hwav 
fxns[10] = cwav
fxns[11] = rwav
fxns[12] = blka
fxns[13] = enes
lfun = {}
fncnt = 0



Y         = [0]
i         = 1
bb        = config.N_FFT_BINS 
loop      = 10
a         = np.array([1,bb])

fun_total = 57 #Total number of functions that kurt will loop through
fun_cut   = 22 #separates into p (3x500) and p (3x1000)

ct        = 0
s         = [0, 0, 0]
kz3       = 0
mp        = 200
m4        = 0
bby       = 1
lst_time  = time.time()
lpp        = 0
transc    = 0
csp       = 0 #For color sparkle
tran      = 0
cho_cnt   = [0 for i in range(16)] #for countin chords breh
### Note - this will need to be changed if the grid on the launchpad changes
cpal      = pallette.pal(0)
ga        = 0 #commanded gaussian filter counter
butarry   = np.zeros((40,25))
bc        = 0
cdc       = 1
lpad      = np.zeros((2,64))


tri = np.linspace(1,8,8).astype(int)
np.random.shuffle(tri)


comfun    = [116, 101, 107, 98, 99, 102, 127, 8] #keyboard coms, t=tics, energy=e, spunk=k, breathe=b, colorwave=c, freestyle=f
counter   = True
spark     = 0
cho_chk   = 0
chocn     = 0
dim = 1

tt = 0
funt = 1

p22 = np.tile(1.0, (3, config.N_PIXELS)) #Transition p

#nergy1"
#unused: "rowl3",sweetscroll
# functs = ["bump","scroll","energy_gaps","rowl",\
#           "spunk","inc2",\
#           "scroll","simplescroll","scroll4",\
#           "trans_wave","half_wave",\
#           "incpal","tic_pal","spectrum","tetris1","tetris2",\
#           "pong","breathe","breathe2","blockage","energy_base","blockage2","becca_breathe",\
#           "steve_breathe","kuwave2","palwave","slide",\
#           "radial_wave","radial_wave2","radial_wave4","radial_wave5","radial_wave6","radial_pal",\
#           "newt","newt2","newt3","newt4",\
#           "colorwave1","colorwave22","colorwave23","colorwave24","colorwave26","colorwave6",\
#           "fireworks","rainfall","liferan",\
#           "ticsfull1","ticsfull2", "ticpal2",\
#           "pointwave2",\
#           "umbrella","umbrella2",
#           "insta","colorwave0","tetris1","subs1","subs2","fract1","fract2","heart1",\
#           "drop", \
#           "bessel1","bessel2","lavalamp","bessel3","bessel4","bessel1","bessel2","bessel3","bessel4","subs1","subs2",\
#           "umbrella3","umbrella4","umbrella5","umbrella7"]   "blockage","blockage2"

#Create array of shuffled function numbers
funs  = np.linspace(0,len(functs)-1,len(functs)).astype(int)
np.random.shuffle(funs)
ft    = 4
fun   = funs[ft]

print('Starting Function: ', functs[fun])
print('Total Functions: {:.0f}'.format(len(functs)))
trcn   = False
ling   = True
bltyp  = 1
umb_init = rn.randint(0,8)
umb_init2 = 9#rn.randint(0,8)
#Main function that calls all others. Each subfunction can be called independently

ni= 0
t11 = 20/2+5
t12 = 40/2+5
t13 = 60/2+5
t14 = 80/2+5
t15 = 100/2+5
butc = 0
sto2 = -1
cntlo = 0
y=5

def visualize_kurt(y):
    global p, loop, a, bb, fun, ct, s, kz3, mp, m2, m3, m4, p2, bby, lst_time, fun_total, fun_cut, transc, trs, tran, last_fun, csp, \
           arry, ind,  cpal, ga, bc, cdc, lpad, comfun, counter, spark, dim, funs, ft, tri, tt, funt, t01, t02, t03, t11, t12, t13, t14, t15, \
           p3, p11, p22, trns, functs, \
           trcn, ling, bltyp, butc, umb_init, umb_init2, sto, sto2,cntlo, ni
    cntlo+=1
    if button.is_pressed:
        print("BUTTON")
        
    #ni += .1#20*np.sin(cntlo/200)+20
    t11 += ni
    t12 += ni
    t13 += ni
    t14 += ni
    t15 += ni
    ct += 1  #Primary counter
    butc+=1
    if ft>=len(funs)-1: #If we've gone through all functions, reshuffle
        print("===========================================Reshuffle funs")
        sto2 = funs[ft]
        np.random.shuffle(funs)
        ft = 0 #reset function index
        
    
    
    
    

#     if funt>=2:
#         p11      = allfuns(y, sto, functs)
#         print('== Function 1: ', functs[sto])
#     else:
    if sto2>-1:
        funs[ft] = sto2
        sto2=-1
        #p11 = allfuns(y, sto2, functs)
    
    p11      = allfuns(y, funs[ft], functs)
    #print('== Function 1: ', functs[funs[ft]])
        
    #print('== Function 2: ', functs[funs[ft+1]])
    if ct >= t11 and ct < t12:       #first half of linger
        p22 = allfuns(y, funs[ft+1], functs)  #pick the next function, co1 down to 0.5, co2 up to 0.5
        co1 = (.5-1)/(t12-t11) * (ct-t11) + 1
        co2 = (.5-0)/(t12-t11) * (ct-t11) + 0
        p3 = zeroup(co1*p11 + co2*p22)

    elif ct >= t12 and ct <t13:  #linger up
        p22 = allfuns(y, funs[ft+1], functs)
        co1 = .5
        co2 = .5
        p3 = zeroup(co1*p11 + co2*p22)
        
        #print(np.max(p22))
    elif ct>=t13 and ct<t14:  #plateu overlap
        p22 = allfuns(y, funs[ft+1], functs)
        co1 = (0-.5)/(t14-t13)*(ct-t13) + .5
        co2 = (.5-1)/(t13-t14)*(ct-t13) + .5
        p3 = zeroup(co1*p11 + co2*p22)
        sto = funs[ft+1]
    elif ct >=t14 and ct<t15: #linger down
        
        p3 = allfuns(y, sto, functs)
        
       
        
        #p3 = p22
    elif ct>=t15:
        if functs[fun] == "umbrella" or functs[fun] == "umbrella_dark" or functs[fun] == "fract1":
            umb_init = rn.randint(0,9)
            umb_init2 = rn.randint(0,9)
        ct = t11
        funt=2
        ft += 1        #increment function index
        print('=========================================New Function: ', functs[fun])

    else:
        p3 = p11

#     p3 = zeroup(p3)            #No negative numbers being sent to net
#     if np.max(p3) > 255:
#         p3 = 255*p3/np.max(p3)     #renormalize to 255 just in case
#         
    #p3 *= 1  #brightness fraction
    
    return p3
    



def zeroup(fg):
    return (np.abs(fg)+fg)/2
def zerodown(fg):
    return (fg - np.abs(fg))/2


def lpfunselect(lpad):
    global p, fun, fun_total, fun_cut, lp, ct, tran, trs, fxns, lastfun, lfun, fncnt

    for i in range(1,14):
        if lpad[0,i] == i+1 and lpad[1,i] == 1:
            #fun = fxns[i][rn.randint(0,len(fxns[i][:]))]
            if fncnt>=len(fxns[i]):
                fncnt = 0
            fun = fxns[i][fncnt]
            fncnt+=1
            print('Launchpad chosen function: {:.0f}'.format(fun))
            
    if fun<=fun_cut and fun>0:
        p = np.tile(1.0, (3, config.N_PIXELS//2))
    elif fun>0:
        p = np.tile(1.0, (3, config.N_PIXELS))
    tran = 0
    ct = 0
    trs = 0
    return fun
coms = 0
def funselect(command):
    global p, fun, fun_total, fun_cut, lp, coms, ct, tran, trs
    tics = [8, 9, 20, 22]
    enes = [1, 7, 10, 15]
    spnk = [3, 16, 17, 23, 24, 25]
    cwav = [45, 46]
    brth = [28, 30, 36, 37, 38, 41, 42]
    
    if command == 116:
        coms[116] = 0
        fun = tics[rn.randint(0,len(tics))] #Tics obv
    elif command == 101:
        coms[101] = 0
        fun = enes[rn.randint(0,len(enes))] #energies
    elif command == 107:
        coms[107] = 0
        fun = spnk[rn.randint(0,len(spnk))] #Spunk
    elif command == 99:
        coms[99] = 0
        fun = cwav[rn.randint(0,len(cwav))] #colorwave
    elif command == 98:
        coms[98] = 0
        fun = brth[rn.randint(0,len(brth))] 
    if fun<=fun_cut and fun>0:
        p = np.tile(1.0, (3, config.N_PIXELS//2))
    elif fun>0:
        p = np.tile(1.0, (3, config.N_PIXELS))
    tran = 0
    ct = 0
    trs = 0
    return fun
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z

def allfuns(y, fun, functs):
    nam = '{}{}'.format('visualize_',functs[fun])
    return  globals()[nam](y)

#########################################################################
def visualize_colorwave0(y):
    global p
    p = colorwave0.colorwave0(y)
    return p
def visualize_colorwave01(y):
    global p
    p = colorwave01.colorwave01(y)
    return p
def visualize_colorwave02(y):
    global p
    p = colorwave02.colorwave02(y)
    return p
def visualize_colorwave1(y):
    global p
    p = colorwave1.colorwave1(y)
    return p
def visualize_colorwave22(y):
    global p
    p = colorwave22.colorwave22(y)
    return p
def visualize_colorwave23(y):
    global p
    p = colorwave23.colorwave23(y)
    return p
def visualize_colorwave24(y):
    global p
    p = colorwave24.colorwave24(y)
    return p
def visualize_colorwave25(y):
    global p
    p = colorwave25.colorwave25(y)
    return p
def visualize_colorwave26(y):
    global p
    p = colorwave26.colorwave26(y)
    return p
def visualize_colorwave6(y):
    global p
    p = colorwave6.colorwave6(y)
    return p
def visualize_colorwave7(y):
    global p
    p = colorwave7.colorwave7(y)
    return p
def visualize_pointwave(y):
    global p
    p = pointwave.pointwave(y)
    return p
def visualize_pointwave2(y):
    global p
    p = pointwave2.pointwave2(y)
    return p
def visualize_colorscroll(y):
    global p
    p = colorscroll.colorscroll(y)
    return p
#########################################################################
#The blockage family. A small but growing bunch of kiddos
#kinda unique I guess? 
def visualize_blockage(y):
    global p
    p = blockage.blk1(y)
    return p

def visualize_blockage2(y):
    global p
    p = blockage.blk2(y)
    return p

#########################################################################

## The pong family
def visualize_pong(y):
    global p
    p = pongy.pong(y)
    return p
def visualize_slide(y):
    global p
    p = pongy.slide(y)
    return p 

#########################################################################
#Test scripts 
def visualize_testy(y):
    global p
    p = tests.testy(y)
    return p

def visualize_palettes(y):
    global p
    p = tests.palettes(y)
    return p

#########################################################################
#The wave family. Started as an inward outward radial wave.
#But its just kinda wavin all over the place now
def visualize_radial_wave(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, evs2, ods2, phw2,rtim2, phw_gap, exc
    p = radial_wave.radial_wave(y)
    return p

def visualize_radial_wave2(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, evs2, ods2, phw2,rtim2, phw_gap, exc
    p = radial_wave2.radial_wave2(y)
    return p

def visualize_radial_wave3(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, phum
    p = radial_wave3.radial_wave3(y)
    return p

def visualize_radial_wave4(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv
    p = radial_wave4.radial_wave4(y)
    return p

def visualize_radial_wave5(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv
    p = radial_wave5.radial_wave5(y)
    return p

def visualize_radial_pal(y):
    p = radial_pal.radial_pal(y)
    return p
#########################################################################
def visualize_meditation(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, evs2, ods2, phw2,rtim2, phw_gap, exc, co2, coc, coc1, coc2, f
    p = meditation.circle(y)
    return p

#########################################################################
def visualize__wave(y):
    global p
    p = umb.umb_wave(y)
    return p
###############################################################
#The breathing family. breathe isn't the meditation one anymore
def visualize_breathe(y):
    global p
    p = breathing.breathe(y)
    return p

def visualize_breathe2(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, lpad
    p = breathe2.breathe2(y)
    return p

def visualize_becca_breathe(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe
    p = becca_breathe.becca_breathe(y)
    return p

def visualize_steve_breathe(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5
    p = steve_breathe.steve_breathe(y)
    return p

def visualize_staggerwave(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, nuu
    p = breathing.stag(y)
    return p

##################################################
#The wave family. Some of the best ones. 
def visualize_kuwave(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5
    p = kuwave.kuwave(y)
    return p
def visualize_kuwave2(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, nuu
    p = kuwave2.kuwave2(y)
    return p
def visualize_radial_wave6(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, nuu
    #p = breathing.rwave6(y)
    p = radial_wave6.radial_wave6(y)
    return p
def visualize_palwave(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, nuu
    p = breathing.palwave(y)
    return p

#########################################################3

def visualize_rowl(y):
    global p, rowl, cnt, sl_rowl, dire, rowl_thresh, rowl_time, SS
    p = rowl.vrowl(y)
    return p
def visualize_rowl2(y):
    global p, rowl2, cnt, sl, dire, phl
    p = rowl.vrowl2(y)
    return p#np.concatenate((p, p[:, ::-1]), axis=1)
def visualize_rowl3(y): 
    global p, rowl3, cnt, sl, dire, phl, pix
    p = rowl.vrowl3(y)
    return np.concatenate((p, np.fliplr(p)), axis=1)

#########################################################################
"""Effect that originates in the center and scrolls outwards"""
def visualize_scroll(y):
    global p
    p = scroll.scrolly(y)
    return p
def visualize_simplescroll(y):
    global p
    p = scroll.simplescroll(y)
    return p
def visualize_sweetscroll(y):
    global p
    p = scroll.sweetscroll(y)
    return p
def visualize_launchscroll(y, lpad):
    global p, cnt2, pix, cnt3, sl, adr,cnt4
    p = scroll.launchscroll(y, lpad)
    return p           
def visualize_scroll4(y):
    global p
    p = scroll.scroll4(y)
    return p    
#########################################################################
#The energy family. These ones got hella vibes
def visualize_energy1(y):
    global p
    p = energy1.energy1(y)
    return p
def visualize_energy2(y):
    global p
    p = energies.energy2(y)
    return p
def visualize_energy_base(y):
    global p
    #This is in a different class because it uses the full net vector (ie doesn't mirror half)
    p = energy_base.energy_base(y)
    return p
def visualize_energy_classic(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    p = energies.energy_classic(y)
    return p
def visualize_energy_gaps(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    p = energies.energy_gaps(y)
    return p
#########################################
umb_cnt = 0

def visualize_umbrella(y):
    global p, umb_cnt, umb_init
    p = umbrella1.umbrella1(y,umb_init)
    return p

def visualize_umbrella_dark(y):
    global p, umb_cnt, umb_init, umb_init2
    p = umbrella_dark.umbrella_dark(y,umb_init, umb_init2)
    return p
#########################################################################
# The tic family
def visualize_tic(y):
    global p #, lpad 
    p = tics.tic1(y)
    return p
def visualize_tic2(y):
    global p
    p = tics.tic2(y)
    return p
def visualize_tic_pal(y):
    global p
    p = tics.ticpal(y)
    return p
def visualize_ticpal2(y):
    global p
    p = ticpal2.ticpal2(y)
    return p
################## Full tics
def visualize_ticsfull1(y):
    global p
    p = ticsfull.tic1(y) 
    return p
def visualize_ticsfull2(y):
    global p
    p = ticsfull.tic2(y) 
    return p
#########################################################################
#The inc family. Random pixel filling and unfilling. Pretty simple 
def visualize_inc(y):
    global p #, a
    p = incs.inc(y) #y,a
    return p
def visualize_inc2(y):
    global p, qq, cr, hg, hg2, thresh_inc, pix, fwddd, ghh, gh2, jk
    p = incs.inc2(y)
    return p
def visualize_incpal(y):
    global p, qq, cr, hg, hg2, thresh_inc, pix, fwddd, ghh, gh2, jk
    p = incs.incpal(y)
    return p
#########################################################################
#The spunk family
def visualize_spunk(y):
    global p
    p = spunky.spunk(y)
    return p 
def visualize_spectrum(y):
    global p
    p = spectrum.spectrum(y)
    return p
def visualize_tetris1(y):
    global p
    p = tetris1.tetris1(y)
    return p
def visualize_tetris2(y):
    global p
    p = tetris2.tetris2(y)
    return p
def visualize_slow_wave(y):
    global p
    p = spunky.slow_wave(y) 
    return p
def visualize_bump(y):
    global p
    p = spunky.bump(y)
    return p

#########################################################################
#Matts funktions
def visualize_rainfall(y):
    global p
    p = rainfall.rainfall(y)
    return p
def visualize_sunrise(y):
    global p
    p = mf.sunrise(y)
    return p
def visualize_fireworks(y):
    global p
    p = mf.fireworks(y)
    return p
def visualize_sunsandstars(y):
    global p
    p = sunsandstars.sunsandstars(y)
    return p
def visualize_life(y):
    global p
    p = life.deterministic(y)
    return p
def visualize_liferan(y):
    global p
    p = life_random.life_random(y)
    return p 

#########################################################################            
def visualize_half_wave(y):
    global p
    p = half_wave.half_wave(y)
    return p
def visualize_trans_wave(y):
    global p    
    p = trans_wave.trans_wave(y)
    return p
def visualize_spunk2(y):
    global p, cnt2, pix, cnt3, sl, adr,cnt4
    p = spunky.spunk2(y)
    return p      

#################################################################
def visualize_teeth(y):
    global p, cnt2, pix, cnt3, sl, adr,cnt4
    p = special.teeth(y)
    return p
####
def visualize_newt(y):
    global p
    p = newt.newt(y)
    return p
def visualize_newt2(y):
    global p
    p = newt2.newt2(y)
    return p
def visualize_newt3(y):
    global p
    p = newt3.newt3(y)
    return p
def visualize_newt4(y):
    global p
    p = newt4.newt4(y)
    return p
def visualize_energy_base2(y):
    p = energy_base2.energy_base2(y)
    return p
#### functions that use images
def visualize_subs1(y):
    p = subs1.subs1(y)
    return p
def visualize_subs_new(y):
    p = subs_new.subs_new(y)
    return p
def visualize_insta(y):
    p = insta.insta(y)
    return p
def visualize_fract1(y):
    ovs = ["colorwave01","bessel1","bessel2","colorwave02","radial_wave","radial_wave3","radial_wave4","radial_wave5","radial_wave6","pointwave"]

    p = fract1.fract1(y,umb_init, umb_init2)
    return p
def visualize_fract2(y):
    p = fract2.fract2(y)
    return p
def visualize_heart1(y):
    p = heart1.heart1(y,umb_init)
    return p
def visualize_peace(y):
    p = peace.peace(y,umb_init, umb_init2)
    return p
def visualize_lavalamp(y):
    p = lavalamp.lavalamp(y)
    return p
def visualize_drop(y):
    p = drop.drop(y)
    return p
def visualize_bessel1(y):
    p = bessel1.bessel1(y)
    return p
def visualize_bessel2(y):
    p = bessel2.bessel2(5)
    return p
def visualize_bessel3(y):
    p = bessel3.bessel3(y)
    return p
def visualize_bessel4(y):
    p = bessel4.bessel4(y)
    return p
fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS),
                         alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD,
                       alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()
pq = 0

def microphone_update(audio_samples):
    global y_roll, prev_rms, prev_exp, prev_fps_update, pq, p
    # Normalize samples between 0 and 1
    y = .1#audio_samples / 2.0**15
    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)
    
    vol = np.max(np.abs(y_data))
   
    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
        led.pixels = np.tile(0, (3, config.N_PIXELS))
        led.update()
    else:
        # Transform audio input into the frequency domain
#         N = len(y_data)
#         N_zeros = 2**int(np.ceil(np.log2(N))) - N
#         # Pad with zeros until the next power of two
#         y_data *= fft_window
#         y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
#         YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
#         # Construct a Mel filterbank from the FFT data
#         mel = np.atleast_2d(YS).T * dsp.mel_y.T
        # Scale data to values more suitable for visualization
        # mel = np.sum(mel, axis=0)
#         mel = np.sum(mel, axis=0)
#         mel = mel**2.0
        # Gain normalization
#         mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
#         mel /= mel_gain.value
#         mel = mel_smoothing.update(mel)
        
        mel=0
        # Map filterbank output onto LED strip
        output = visualization_effect(mel)  
        led.pixels = output
        
        led.update()
        if config.USE_GUI:
            # Plot filterbank output
            x = np.linspace(config.MIN_FREQUENCY, config.MAX_FREQUENCY, len(mel))
            mel_curve.setData(x=x, y=fft_plot_filter.update(mel))
            # Plot the color channels
            r_curve.setData(y=led.pixels[0])
            g_curve.setData(y=led.pixels[1])
            b_curve.setData(y=led.pixels[2])
    if config.USE_GUI:
        app.processEvents()
    
    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 0.5 > prev_fps_update:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))
            #print(kz)

# Number of audio samples to read every time frame
samples_per_frame = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16

if sys.argv[1] == "spectrum":
        visualization_type = visualize_spectrum
elif sys.argv[1] == "energy1":
        visualization_type = visualize_energy1
elif sys.argv[1] == "energy2":
        visualization_type = visualize_energy2
elif sys.argv[1] == "scroll":
        visualization_type = visualize_scroll
elif sys.argv[1] == "kurt":
        visualization_type = visualize_kurt
elif sys.argv[1] == "bump":
        visualization_type = visualize_bump
elif sys.argv[1] == "inc":
        visualization_type = visualize_inc
elif sys.argv[1] == "inc2":
        visualization_type = visualize_inc2
elif sys.argv[1] == "incpal":
        visualization_type = visualize_incpal
elif sys.argv[1] == "tic":
        visualization_type = visualize_tic
elif sys.argv[1] == "tic2":
        visualization_type = visualize_tic2
elif sys.argv[1] == "tic3":
        visualization_type = visualize_tic3
elif sys.argv[1] == "tic_pal":
        visualization_type = visualize_tic_pal
elif sys.argv[1] == "half_wave":
        visualization_type = visualize_half_wave
elif sys.argv[1] == "sweetscroll":
        visualization_type = visualize_sweetscroll
elif sys.argv[1] == "energy_classic":
        visualization_type = visualize_energy_classic
elif sys.argv[1] == "energy_gaps":
        visualization_type = visualize_energy_gaps
elif sys.argv[1] == "energy_base":
        visualization_type = visualize_energy_base
elif sys.argv[1] == "energy_base2":
        visualization_type = visualize_energy_base2
elif sys.argv[1] == "slide":
        visualization_type = visualize_slide
elif sys.argv[1] == "simplescroll":
        visualization_type = visualize_simplescroll
elif sys.argv[1] == "rowl":
        visualization_type = visualize_rowl
elif sys.argv[1] == "rowl2":
        visualization_type = visualize_rowl2
elif sys.argv[1] == "spunk":
        visualization_type = visualize_spunk
elif sys.argv[1] == "spunk2":
        visualization_type = visualize_spunk2
elif sys.argv[1] == "umbrella":
        visualization_type = visualize_umbrella
elif sys.argv[1] == "tetris1":
        visualization_type = visualize_tetris1
elif sys.argv[1] == "tetris2":
        visualization_type = visualize_tetris2
elif sys.argv[1] == "radial_wave":
        visualization_type = visualize_radial_wave
elif sys.argv[1] == "radial_wave2":
        visualization_type = visualize_radial_wave2
elif sys.argv[1] == "radial_wave3":
        visualization_type = visualize_radial_wave3
elif sys.argv[1] == "radial_wave4":
        visualization_type = visualize_radial_wave4
elif sys.argv[1] == "umbrella_wave":
        visualization_type = visualize_umbrella_wave
elif sys.argv[1] == "radial_wave5":
        visualization_type = visualize_radial_wave5
elif sys.argv[1] == "radial_wave6":
        visualization_type = visualize_radial_wave6
elif sys.argv[1] == "trans_wave":
        visualization_type = visualize_trans_wave
elif sys.argv[1] == "breathe":
        visualization_type = visualize_breathe
elif sys.argv[1] == "breathe2":
        visualization_type = visualize_breathe2
elif sys.argv[1] == "becca_breathe":
        visualization_type = visualize_becca_breathe
elif sys.argv[1] == "pong":
        visualization_type = visualize_pong
elif sys.argv[1] == "blockage":
        visualization_type = visualize_blockage
elif sys.argv[1] == "slow_wave":
        visualization_type = visualize_slow_wave
elif sys.argv[1] == "rowl3":
        visualization_type = visualize_rowl3
elif sys.argv[1] == "transition":
        visualization_type = visualize_transition
elif sys.argv[1] == "testy":
        visualization_type = visualize_testy
elif sys.argv[1] == "steve_breathe":
        visualization_type = visualize_steve_breathe
elif sys.argv[1] == "kuwave":
        visualization_type = visualize_kuwave
elif sys.argv[1] == "kuwave2":
        visualization_type = visualize_kuwave2
elif sys.argv[1] == "staggerwave":
        visualization_type = visualize_staggerwave
elif sys.argv[1] == "blockage2":
        visualization_type = visualize_blockage2
elif sys.argv[1] == "radial_pal":
        visualization_type = visualize_radial_pal
elif sys.argv[1] == "palwave":
        visualization_type = visualize_palwave
elif sys.argv[1] == "imgconv":
        visualization_type = visualize_imgconv
elif sys.argv[1] == "palettes":
        visualization_type = visualize_palettes
elif sys.argv[1] == "meditation":
        visualization_type = visualize_meditation
elif sys.argv[1] == "colorwave0":
        visualization_type = visualize_colorwave0
elif sys.argv[1] == "colorwave01":
        visualization_type = visualize_colorwave01
elif sys.argv[1] == "colorwave02":
        visualization_type = visualize_colorwave02
elif sys.argv[1] == "colorwave1":
        visualization_type = visualize_colorwave1
elif sys.argv[1] == "colorwave22":
        visualization_type = visualize_colorwave22
elif sys.argv[1] == "colorwave23":
        visualization_type = visualize_colorwave23
elif sys.argv[1] == "colorwave25":
        visualization_type = visualize_colorwave25
elif sys.argv[1] == "colorwave26":
        visualization_type = visualize_colorwave26
elif sys.argv[1] == "colorwave3":
        visualization_type = visualize_colorwave3
elif sys.argv[1] == "colorwave6":
        visualization_type = visualize_colorwave6
elif sys.argv[1] == "colorwave7":
        visualization_type = visualize_colorwave7
elif sys.argv[1] == 'rainfall':
        visualization_type = visualize_rainfall
elif sys.argv[1] == 'sunrise':
        visualization_type = visualize_sunrise
elif sys.argv[1] == 'fireworks':
        visualization_type = visualize_fireworks
elif sys.argv[1] == 'sunsandstars':
        visualization_type = visualize_sunsandstars
elif sys.argv[1] == 'teeth':
        visualization_type = visualize_teeth
elif sys.argv[1] == 'launchscroll':
        visualization_type = visualize_launchscroll
elif sys.argv[1] == 'life':
        visualization_type = visualize_life
elif sys.argv[1] == 'liferan':
        visualization_type = visualize_liferan
elif sys.argv[1] == 'newt':
        visualization_type = visualize_newt
elif sys.argv[1] == 'newt2':
        visualization_type = visualize_newt2
elif sys.argv[1] == 'newt3':
        visualization_type = visualize_newt3
elif sys.argv[1] == 'newt4':
        visualization_type = visualize_newt4
elif sys.argv[1] == 'ticpal2':
        visualization_type = visualize_ticpal2
elif sys.argv[1] == 'pointwave':
        visualization_type = visualize_pointwave
elif sys.argv[1] == 'pointwave2':
        visualization_type = visualize_pointwave2
elif sys.argv[1] == 'umbrella2':
        visualization_type = visualize_umbrella2
elif sys.argv[1] == 'umbrella3':
        visualization_type = visualize_umbrella3
elif sys.argv[1] == 'umbrella4':
        visualization_type = visualize_umbrella4
elif sys.argv[1] == 'umbrella5':
        visualization_type = visualize_umbrella5
elif sys.argv[1] == 'umbrella6':
        visualization_type = visualize_umbrella6
elif sys.argv[1] == 'umbrella7':
        visualization_type = visualize_umbrella7        
elif sys.argv[1] == 'subs1':
        visualization_type = visualize_subs1
elif sys.argv[1] == 'subs_new':
        visualization_type = visualize_subs_new
elif sys.argv[1] == 'insta':
        visualization_type = visualize_insta
elif sys.argv[1] == 'fract1':
        visualization_type = visualize_fract1
elif sys.argv[1] == 'fract2':
        visualization_type = visualize_fract2
elif sys.argv[1] == 'heart1':
        visualization_type = visualize_heart1
elif sys.argv[1] == 'lavalamp':
        visualization_type = visualize_lavalamp
elif sys.argv[1] == 'drop':
        visualization_type = visualize_drop       
elif sys.argv[1] == 'scroll4':
        visualization_type = visualize_scroll4
elif sys.argv[1] == 'bessel1':
        visualization_type = visualize_bessel1
elif sys.argv[1] == 'bessel2':
        visualization_type = visualize_bessel2
elif sys.argv[1] == 'bessel3':
        visualization_type = visualize_bessel3
elif sys.argv[1] == 'bessel4':
        visualization_type = visualize_bessel4
elif sys.argv[1] == 'umbrella_dark':
        visualization_type = visualize_umbrella_dark
elif sys.argv[1] == 'colorscroll':
        visualization_type = visualize_colorscroll
elif sys.argv[1] == 'peace':
        visualization_type = visualize_peace
#else: #sys.argv[1] == "tetris":
        #visualization_type = visualize_slow_wave
visualization_effect = visualization_type
"""Visualization effect to display on the LED strip"""

if __name__ == '__main__':

    # Initialize LEDs
    #while True:
    led.update()
    # Start listening to live audio stream
    microphone.start_stream(microphone_update)
 
  




