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
from numpy import random as rn
from array import array


visualization_type = sys.argv[1]

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)

"""The low-pass filter used to estimate frames-per-second"""
kz = int(0)
k = 0
cinc = 0
kk = 120
c = 0
S = np.zeros(((config.N_PIXELS // 2) - 1))
SS = ((config.N_PIXELS // 2) - 1)
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


@memoize
def _normalized_linspace(size):
    return np.linspace(0, 1, size)


def interpolate(y, new_length):
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    z = np.interp(x_new, x_old, y)
    return z


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
if sys.argv[1] == "umbrella" or sys.argv[1] == "energy_base" or sys.argv[1] == "radial_wave" or sys.argv[1] == "radial_wave2" or sys.argv[1] == "radial_wave3" \
   or sys.argv[1] == "breathe" or sys.argv[1] == "pong" or sys.argv[1] == "blockage" or sys.argv[1] == "breathe2" or sys.argv[1] == "radial_wave4" \
   or sys.argv[1] == "umbrella_wave":
    p = np.tile(1.0, (3, config.N_PIXELS))
else:
    p = np.tile(1.0, (3, config.N_PIXELS // 2))
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)

Y=[0]
i = 1
bb = config.N_FFT_BINS 
loop = 10
a = np.array([1,bb])
fun_total = 30
fun = rn.randint(1,fun_total)
#fun=8
print(fun)
ct = 0
s = [0, 0, 0]
kz3 = 0
mp = 200
m4 = 0
bby =1
lst_time = time.time()
lp = 0
fun_cut = 20

def visualize_kurt(y):
    global p, loop, a, bb, fun, ct, s, kz3, mp, m2, m3, m4, p2, bby, t1, t2, lst_time, lp, fun_total, fun_cut
    y2 = y**2
    gain.update(y2)
    y2 /= gain.value
    y2 *= 255.0
    m2 = np.mean(y2[28:])
    gth = np.mean(y)/np.max(y)
    if m2 > 250 and ct>20:
        fun = rn.randint(1,fun_total)
        lp+=1
        print("FORCED CHANGE")
        kz3 = 0
        if fun<=fun_cut:
            p = np.tile(1.0, (3, config.N_PIXELS//2))
        else:
            p = np.tile(1.0, (3, config.N_PIXELS))
    if ct == 0:
        if fun<=fun_cut:
            p = np.tile(1.0, (3, config.N_PIXELS//2))
        else:
            p = np.tile(1.0, (3, config.N_PIXELS))
    
    
    if fun   == 1:
        visualize_energy(y)
    elif fun == 2:
        visualize_scroll(y)
    elif fun == 3:
        visualize_bump(y)
    elif fun == 4:
        visualize_wave(y)
    elif fun == 5:
        visualize_inc(y)
    elif fun == 6:
        visualize_sweetscroll(y)
    elif fun == 7:
        visualize_energy_classic(y)
    elif fun == 8:
        visualize_tic(y)
    elif fun == 9:
        visualize_tic2(y)
    elif fun == 10:
        visualize_energy_gaps(y)
    elif fun == 11:
        visualize_slide(y)
    elif fun == 12:
        visualize_simplescroll(y)        
    elif fun == 13:
        visualize_rowl(y)
    elif fun == 14:
        phl = rn.randint(10,30)
        visualize_rowl2(y)
    elif fun == 15:
        visualize_energy2(y)
    elif fun == 16:
        visualize_spunk(y)
    elif fun == 17:
        visualize_spunk2(y)
    elif fun == 18:
        visualize_trans_wave(y)
    elif fun == 19:
        visualize_inc2(y)
    elif fun == 20:
        visualize_tic3(y)
    elif fun == 21:
        visualize_rowl3(y)
    elif fun == 22:
        visualize_radial_wave2(y)
    elif fun == 23:
        visualize_pong(y)
    elif fun == 24:
        visualize_breathe(y)
    elif fun == 25:
        visualize_radial_wave3(y)
    elif fun == 26:
        visualize_breathe2(y)
    elif fun == 27:
        visualize_blockage(y)
    elif fun == 28:
        visualize_energy_base(y)
    elif fun == 29:
        visualize_umbrella(y)
    elif fun == 30:
        visualize_umbrella_wave(y)
    #elif fun == 31:
        #visualize_transition(y)
    #elif fun == 16:
        #p = np.tile(1.0, (3, config.N_PIXELS))
        #lp = 0
        #visualize_umbrella(y)
    ct+=1
    
        
    if ct>200 and m2>40:  #based on count and threshold high freq
        ct = 0
        last_fun = fun
        fun = rn.randint(1,fun_total)
        lp+=1
        print("----")
        print(fun)
        if fun<=fun_cut and last_fun>fun_cut:
            p = np.tile(1.0, (3, config.N_PIXELS//2))
        elif fun>fun_cut and last_fun<=fun_cut:
            p = np.tile(1.0, (3, config.N_PIXELS))
    if ct>500:
        ct=0
        fun = rn.randint(1,fun_total)
        kz3 = 0
        lp+=1
        print(fun)
        if fun<=fun_cut:
            p = np.tile(1.0, (3, config.N_PIXELS//2))
        else:
            p = np.tile(1.0, (3, config.N_PIXELS))
    kz3+=1
    #if gth >.6:
        #p = np.fliplr(p)
        #print("flip")
    
    #Change up the mapping symmetry
    #if m2 > 75: #high frequency response
        #print("color change")
        #Shake things up a bit
        #print(len(p[2,:])//2)
        #p[0,:187] = p[1,186:]
        #p[1,:186] = p[2,186:]
        #p[2,:len(p[0,:])//2] = p[0,len(p[0,:])//2:]
    
    if kz3 <mp: #based on pure counter
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    elif kz3 > mp-1 and kz3 < 2*mp or m1>2: #Based on pure counter

        p2 = 0*p
        #Reorder colors for half the net
        p2[0,:] = p[1,:]
        p2[1,:] = p[2,:]
        p2[2,:] = p[0,:]
        return np.concatenate((p[:, ::-1], p2), axis=1) #reversed symmetry
        #kz3 = mp
        
    else:
        #sym3 += 1
        #print("sym 3")
        #Reorder colors for half the net
        p2 = 0*p
        p2[0,:] = p[1,:]
        p2[1,:] = p[2,:]
        p2[2,:] = p[0,:]
        s[0] += 1
        #print("sym 3")
        if s[0] > 2*mp-1:
            kz3 = 0
        return np.concatenate((p, p2         ), axis=1) #no symmetry
cnt2 = np.array([0,0,0])

#umbrella = np.tile(1.0, (3, config.N_PIXELS))
u1 = [ 27, 29]
u2 = [64, 65, 66, 67, 68, 69, 70, 71, 72, 73]
u3 = [127, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139]
u4 = [158, 159, 160, 161, 162, 163, 170, 171, 172]
u5 = [226, 227, 228, 229, 230, 231, 232, 238, 239, 240, 241, 242, 243]
u6 = [256, 257, 258, 259, 260, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 274]
u7 = [ 325, 326, 327, 328, 332, 333, 334, 336, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347] #320, 321, 322, 419, 421,423, 417
u8 = [352, 353, 354, 355, 356, 357, 358, 359, 360, 365, 366, 367, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, \
      384, 385, 386, 387, 388, 389, 390]
u9 = [410, 412, 414, 416,  418, 419, 420, 421,  422, 424, 425, 426, 427, 428, 429, 431, 432, 433, 434, 435, 436, 437, 438, 439, \
      440, 441, 442, 443, 444, 446]
u10 = [455, 456, 457, 458, 463,464,  465, 466, 467, 468, 469, 470, 471, 472, 477, 478, 479, 480]
u11 = [527, 528, 529, 530, 534,539, 540, 541, 542]
u12 = [559, 560, 561, 562, 564, 568, 569, 570, 571]
u13 = [628, 629, 630, 631, 633, 634, 635, 636, 637, 638]
u14 = [663, 665, 666, 667, 668, 669, 670, 671, 672]         
u15 = [726, 728, 730]


cnt2 = 0
umbrella = np.array(u1+u2+u3+u4+u5+u6+u7+u8+u9+u10+u11+u12+u13+u14+u15)
umbl = np.linspace(0,198,100).astype(int)
umbl2 = np.linspace(1,199,100).astype(int)


cnt3=0
ttop2 = np.array([49 , 149, 249, 349, 449, 549,  649, 749])
ttop = np.array([50, 150, 250, 350, 450, 550, 650])
umb_thresh = 15
hit = 0
phum = np.array([0,25/3,2*25/3])

dec = 49
cnt4 = 0
ind = np.array([0,1,2])
cnt5 = 0
drop = rn.randint(0,len(ttop)-1)
drop2 = rn.randint(0,len(ttop)-1)
cnt6=0
lp2=0
dwn = 0
def visualize_umbrella(y):
    global p, cnt2, cnt3, umbrella, tipp, umb_thresh, hit, phum, ttop, dec, cnt4, ind, cnt5, drop, cnt6, drop2, cnt6, lp2, ttop2, umbl, umbl2, dwn
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    cnt2+=1
    
    
    cnt5+=1
    cnt6+=1
    m2 = int(np.mean(y)/np.max(y)*255)
    m3 = int(np.mean(y[len(y)//2::])/np.max(y)*255)
    if m3>50:
        print(m3)

   
        
    #attempted rain, not a huge fan
    #if cnt2 > 100:
        #p[2,ttop2] = .5*int(.5*(np.sin(cnt2/25+phum[0])+1)*255)
        #p[1,ttop2] = .5*int(.5*(np.sin(cnt2/25+phum[1])+1)*255)
        #p[0,ttop2] = .5*int(.5*(np.sin(cnt2/25+phum[2])+1)*255)
        #p[1,ttop] = .5*int(.5*(np.sin(cnt2/25+phum[0])+1)*255)
        #p[2,ttop] = .5*int(.5*(np.sin(cnt2/25+phum[1])+1)*255)
        #p[0,ttop] = .5*int(.5*(np.sin(cnt2/25+phum[2])+1)*255)
        #if cnt2>200:
            #cnt2 = 0
    
    if ttop[0]//100>=1:
        ttop = np.array([50, 150, 250, 350, 450, 550, 650])
        ttop2 = np.array([48,49 , 148,149,248, 249, 348,349, 448,449, 548,549, 648, 649,748, 749])
        #p[:,:] = 0
    p[0,umbrella] = .5*(np.sin(cnt2/25+phum[0])+1)*255 #[umbl]
    p[1,umbrella] = .5*(np.sin(cnt2/25+phum[1])+1)*255
    p[2,umbrella] = .5*(np.sin(cnt2/25+phum[2])+1)*255
    
    #p[2,umbrella[umbl2]] = .5*(np.sin(cnt2/25+phum[0])+1)*255
    #p[1,umbrella[umbl2]] = .5*(np.sin(cnt2/25+phum[1])+1)*255
    #p[0,umbrella[umbl2]] = .5*(np.sin(cnt2/25+phum[2])+1)*255
    
    if m2>75:
        cnt3+=1
        if cnt3>10:
            #p[1,umbrella[umbl2]] = .5*(np.sin(cnt2/25+phum[0])+1)*255
            #p[2,umbrella[umbl2]] = .5*(np.sin(cnt2/25+phum[1])+1)*255
            #p[0,umbrella[umbl2]] = .5*(np.sin(cnt2/25+phum[2])+1)*255
            cnt3=0
    if m3>500:
        cnt4+=1
        if cnt4>10:
            #p[1,umbrella[umbl]] = .5*(np.sin(cnt2/25+phum[0])+1)*255
            #p[2,umbrella[umbl]] = .5*(np.sin(cnt2/25+phum[1])+1)*255
            #p[0,umbrella[umbl]] = .5*(np.sin(cnt2/25+phum[2])+1)*255
            cnt4=0
    
            Por  = np.reshape(p[0],(15,50))
            Por1 = np.reshape(p[1],(15,50))
            Por2 = np.reshape(p[2],(15,50))
    
            pltr = np.linspace(1,48,48).astype(int)
            
            if dwn == 1 and cnt5>20:
                cnt5 = 0
                dwn = 0
                for x in pltr:
                    Por[:,x-1] = Por[:,x]
                    Por1[:,x-1] = Por1[:,x]
                    Por2[:,x-1] = Por2[:,x]
        
                    Por[:,x] = 0
                    Por1[:,x] = 0
                    Por2[:,x] = 0
                p[0,:] = Por.flatten()
                p[1,:] = Por.flatten()
                p[2,:] = Por.flatten()
            elif dwn == 0 and cnt5>20:
                dwn = 1
                cnt5 = 0
                for x in pltr:
                    Por[:,x+1] = Por[:,x]
                    Por1[:,x+1] = Por1[:,x]
                    Por2[:,x+1] = Por2[:,x]
        
                Por[:,x] = 0
                Por1[:,x] = 0
                Por2[:,x] = 0
        #p[0,:] = Por.flatten()
        #p[1,:] = Por1.flatten()
        #p[2,:] = Por2.flatten()
    return p
blk=1
bla0 = np.zeros((15,50)).astype(int)
bla1 = np.zeros((15,50)).astype(int)
bla2 = np.zeros((15,50)).astype(int)
#print(bla)
bly = np.linspace(0,49,49//2).astype(int)
btim =0
blkthr = .01
blkcn = 0
trz = 0
trz0 = 0
def visualize_transition(y):
    global p, trz, trz0, btim, ary
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    trn = np.mean(y[len(y)//2::])/np.max(y)
    trz0 = trz
    print(trn)
    if trn>.2:
        btim+=1
    #p[:,trz+1] = p[:,trz]
        trz=int((trz+1)**1.001)
     
        p[0,trz0:trz] = 255*(.5*np.sin(btim/30)+.5)**.5
        p[1,trz0:trz] = 255*(.5*np.sin(btim/30+30/3)+.5)**.5
        p[2,trz0:trz] = 255*(.5*np.sin(btim/30+2*30/3)+.5)**.5
        print(btim)
    return p
    
    
def visualize_blockage(y):
    global p, blk, bla0, bla1, bla2, bly, ary, btim, blkthr, blkcn
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    blkcn+=1
    blth = np.mean(y)/np.max(y) #[len(y)//2::]
    
    btim+=1
    bla0[0,:] = 255*np.sin(btim/50+ ary/10)*(.5*np.sin(btim/30)+.5)**.5
    bla1[0,:] = 255*np.sin(btim/50+ ary/10*np.pi/2)*(.5*np.sin(btim/30 + ary/np.pi/2)+.5)**.5
    bla2[0,:] = 255*np.sin(btim/50+ ary/10*np.pi)*(.5*np.sin(btim/30+ary/np.pi)+.5)**.5
    blkcn+=1

    if blth>blkthr:
        
        bla0[blk,:] = bla0[blk-1,:]
        bla1[blk,:] = bla1[blk-1,:]
        bla2[blk,:] = bla2[blk-1,:]
        #threshold changes not working cuz im drunk
        if blkcn >50:
            #blkthr *=.9
            print("Blk Up")
        elif blkthr<5:
            #blkthr *=1.1
            print("Blk Down")
            print(blkthr)
        blkcn = 0     
    blk+=1

    if blk>14:
        blk= 0
        
    p[0,:] = bla0[:,:].flatten()
    p[1,:] = bla1[:,:].flatten()
    p[2,:] = bla2[:,:].flatten()
    
    p[0,:] = gaussian_filter1d(p[0,:], sigma=2)
    p[1,:] = gaussian_filter1d(p[1,:], sigma=2)
    p[2,:] = gaussian_filter1d(p[2,:], sigma=2)
    
    p = np.fliplr(p)
    
    return p


rtim = 0
arx = np.linspace(0,14,15).astype(int)
ary = np.linspace(0,49,50).astype(int)
ar_wave0 = np.ones((15,50))
ar_wave1 = np.ones((15,50))
ar_wave2 = np.ones((15,50))

phw = 10
rtim3 =0
coo = np.array([1,1,1]).astype(float)
xdiv = 14
ydiv = 49
abc = 0
dcr = 0
def visualize_radial_wave(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    

        
    rtim +=1
    rtim3+=1
   
    xph = np.abs(arx-np.mean(arx))/np.pi
    xx = np.abs(arx-14/2)/100
    for y in ary:
        yph = np.abs(y-np.mean(y))/100
        yy = np.abs(y-49/2)
        ar_wave0[arx,y] = (( np.sin(yy*np.pi/(ydiv))/2 + np.sin(xx*np.pi/(xdiv))/2+.5)**1)*(.5*np.sin(rtim/phw+yy/4+xx**2)+.5)*300
        #for x in arx:
            #ar_wave0[x,y] = ((np.sin(y*np.pi/(ydiv))/2 + np.sin(x*np.pi/(xdiv))/2)**2)*(.5*np.sin(rtim/phw+ y/5+x/4)+.5)*255  #49 and 14 are circular
            #print((.5*np.sin(rtim/phw+ y/5)+.5))
            #ar_wave1[x,y] = ((np.sin(y*np.pi/(49))/2 + np.sin(x*np.pi/(14))/2)**2)*(.5*np.sin(rtim/phw + phw/3)+.5)*255
            #ar_wave2[x,y] = ((np.sin(y*np.pi/(49))/2 + np.sin(x*np.pi/(14))/2)**2)*(.5*np.sin(rtim/phw + 2*phw/3)+.5)*255
    coo[0] = (.5*np.sin(rtim/30)+.5)**.5
    coo[1] = (.5*np.sin(rtim/30+30/3)+.5)**.5
    coo[2] = (.5*np.sin(rtim/30+2*30/3)+.5)**.5
    
    p[0,:] = coo[0]*ar_wave0.flatten() + arby.flatten()
    p[1,:] = coo[1]*ar_wave0.flatten() + arby.flatten()
    p[2,:] = coo[2]*ar_wave0.flatten() + arby.flatten()
    ppm = np.mean(p[:,:])
    
    
    xdiv = (.5*np.sin(rtim/100)+.5)*20 + 14-14/2
    ydiv = (.5*np.sin(rtim/100)+.5)*49 + 49-49/2
        
    p = gaussian_filter1d(p, sigma=.2)
    return p

arby = np.zeros((15,50))
rr = rn.randint(2,13)
ry = rn.randint(2,47)
#xxs = np.array([rr, rr+1, rr-1]).astype(int)
#yys = np.array([ry, ry, ry]).astype(int)
xxs = np.linspace(0,14,15).astype(int)
yys = np.zeros((1,15)).astype(int)
yys2 = np.zeros((1,15)).astype(int)+49
yys3 = np.zeros((1,15)).astype(int)+24
rtim4 = 0
def visualize_pong(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim4, coo, xdiv, ydiv, arby, abc, dcr, xxs, yys, yys2, yys3, oods
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    abc+=1
    rtim+=1
    
    if np.mean(y[2*len(y)//3::])>2 and abc>2 or abc>20: #on at 5 
        abc=0
        
        if dcr == 0:
            arby[xxs,yys] = 0
            arby[xxs,yys2] = 0
            arby[xxs,yys3] = 0
            #xxs += 1
            yys += 1
            yys2-=1
            yys3-=1
        elif dcr == 1:
            arby[xxs,yys] = 0
            arby[xxs,yys2] = 0
            arby[xxs,yys3] = 0
            #xxs -= 1
            yys -= 1
            yys2+=1
            yys3+=1
        if np.max(yys)>=49: #np.max(xxs)>= 12 or 
            dcr = 1
        elif np.min(yys)<=0: #np.min(xxs)<= 2 or 
            dcr = 0
            rtim4+=1
            #rr = rn.randint(2,13)
            #ry = rn.randint(2,47)
            #xxs = np.array([rr]).astype(int)
            #yys = np.array([ry]).astype(int)
        arby[xxs,yys] = 255
        arby[xxs,yys2] = 255
        arby[xxs,yys3] = 255
    coo[0] = (.5*np.sin(rtim/30)+.5)**.5
    coo[1] = (.5*np.sin(rtim/30+30/3)+.5)**.5
    coo[2] = (.5*np.sin(rtim/30+2*30/3)+.5)**.5
    
    p[0,:] = coo[0]*arby.flatten()
    p[1,:] = coo[1]*arby.flatten()
    p[2,:] = coo[2]*arby.flatten()
    if rtim4>2:
        p[0,oods] = p[1,oods]
        p[1,oods] = p[2,oods]
        p[2,oods] = p[0,oods]
    if rtim4>4:
        rtim4 = 0
    
    p[0,:] = gaussian_filter1d(p[0,:], sigma=1)**1.5
    p[1,:] = gaussian_filter1d(p[1,:], sigma=1)**1.5
    p[2,:] = gaussian_filter1d(p[2,:], sigma=1)**1.5
    
    
    return p

evs2 = np.linspace(0,14,8).astype(int)
ods2 = np.linspace(1,13,7).astype(int)
phw2 = 25
rtim2 = 0
phw_gap =0
exc = 8
def visualize_radial_wave2(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, evs2, ods2, phw2,rtim2, phw_gap, exc
   
    rtim +=1
    rtim2 +=1
    for y in ary:
        
        ar_wave0[evs2,y] = ((.5*np.sin(y*np.pi/49+rtim/10)+.5)**exc)*(.5*(np.sin(rtim/phw        )+1))*255
        ar_wave0[ods2,y] = ((.5*np.sin(y*np.pi/49+rtim/10)+.5)**exc)*(.5*(np.sin(rtim/phw + np.pi)+1))*255
        
        ar_wave1[evs2,y] = ((.5*np.sin(y*np.pi/49+rtim/10)+.5)**exc)*(.5*(np.sin(rtim/phw2 + phw2/3+phw_gap)+1))*255
        ar_wave1[ods2,y] = ((.5*np.sin(y*np.pi/49+rtim/10)+.5)**exc)*(.5*(np.sin(rtim/phw2 + phw2/3 + np.pi)+1))*255
        
        ar_wave2[evs2,y] = ((.5*np.sin(y*np.pi/49+rtim/10)+.5)**exc)*(.5*(np.sin(rtim/phw + 2*phw/3)+1))*255
        ar_wave2[ods2,y] = ((.5*np.sin(y*np.pi/49+rtim/10)+.5)**exc)*(.5*(np.sin(rtim/phw2 + 2*phw2/3 + np.pi+phw_gap)+1))*255

        
    p[0,:] = ar_wave0.flatten()
    p[1,:] = ar_wave1.flatten()
    p[2,:] = ar_wave2.flatten()
    if rtim2 > 50 and np.mean(p[:,:])<10:
        exc-=1
        rtim2 = 0
    #p = gaussian_filter1d(p, sigma=2)
    return p
arx = np.linspace(0,14,15).astype(int)
ary = np.linspace(0,49,50).astype(int)
ar_wave0 = np.ones((15,50))
ar_wave1 = np.ones((15,50))
ar_wave2 = np.ones((15,50))

def visualize_radial_wave3(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, phum
    
    rtim +=1
    rtim3+=1
    #for y in ary:
        #for x in arx:
    tt = np.transpose(((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1))
    ar_wave0[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ ary/7)+1)*255  #49 and 14 are circular +arx/4 + np.sin(arx*np.pi/(xdiv))/2
    #ar_wave1[:,:] = ((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1)*255
    #ar_wave0[arx,:] *= np.transpose(((np.cos(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/5)+1)*255)
    #ar_wave0[arx,arx] = ((np.cos(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/5)+1)*255  #49 and 14 are circular +arx/4 + np.sin(arx*np.pi/(xdiv))/2
    #ar_wave1[arx,ary] = ((np.sin(ary*np.pi/(ydiv))/2 + np.sin(arx*np.pi/(xdiv))/2)**2)*(.5*np.sin(rtim/phw- ary/5-arx/4)+.5)*255  #49 and 14 are circular
            #ar_wave1[x,y] = ((np.sin(y*np.pi/(ydiv))/2 + np.sin(x*np.pi/(xdiv))/2)**2)*(.5*np.sin(rtim/phw-y/5-x/2)+.5)*255  #49 and 14 are circular
            #print((.5*np.sin(rtim/phw+ y/5)+.5))
            #ar_wave1[x,y] = ((np.sin(y*np.pi/(49))/2 + np.sin(x*np.pi/(14))/2)**2)*(.5*np.sin(rtim/phw + phw/3)+.5)*255
            #ar_wave2[x,y] = ((np.sin(y*np.pi/(49))/2 + np.sin(x*np.pi/(14))/2)**2)*(.5*np.sin(rtim/phw + 2*phw/3)+.5)*255
    coo[0] = (.5*np.sin(rtim/phw)+.5)**.5
    coo[1] = (.5*np.sin(rtim/phw/3+phw/3)+.5)**.5
    coo[2] = (.5*np.sin(rtim/phw/3+2*phw/3)+.5)**.5
    
    p[0,:] = coo[0]*ar_wave0.flatten()#+ar_wave1.flatten())
    p[1,:] = coo[1]*ar_wave0.flatten()#+ar_wave1.flatten())
    p[2,:] = coo[2]*ar_wave0.flatten()#+ar_wave1.flatten())
    
    
    ppm = np.mean(p[:,:])
    
    
    xdiv = (.5*np.sin(rtim/50)+.5)*20 + 14-14/2
    ydiv = (.5*np.sin(rtim/50)+.5)*49 + 49-49/2
        
    p = gaussian_filter1d(p, sigma=2)*.5
    
    return p
def visualize_umbrella_wave(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, phum
    
    rtim +=1
    rtim3+=1

    tt = np.transpose(((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1))
    ar_wave0[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ ary/7)+1)*255  #49 and 14 are circular +arx/4 + np.sin(arx*np.pi/(xdiv))/2

    coo[0] = (.5*np.sin(rtim/phw)+.5)**.5
    coo[1] = (.5*np.sin(rtim/phw/3+phw/3)+.5)**.5
    coo[2] = (.5*np.sin(rtim/phw/3+2*phw/3)+.5)**.5
    
    p[0,:] = coo[0]*ar_wave0.flatten()#+ar_wave1.flatten())
    p[1,:] = coo[1]*ar_wave0.flatten()#+ar_wave1.flatten())
    p[2,:] = coo[2]*ar_wave0.flatten()#+ar_wave1.flatten())
    
    
    ppm = np.mean(p[:,:])
    
    
    xdiv = (.5*np.sin(rtim/50)+.5)*20 + 14-14/2
    ydiv = (.5*np.sin(rtim/50)+.5)*49 + 49-49/2
        
    p = gaussian_filter1d(p, sigma=2)*.5
    
    p[0,umbrella] = .5*(np.sin(rtim/25+phum[0])+1)*255 #[umbl]
    p[1,umbrella] = .5*(np.sin(rtim/25+phum[1])+1)*255
    p[2,umbrella] = .5*(np.sin(rtim/25+phum[2])+1)*255
    
    return p
def visualize_radial_wave4(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv
    
    rtim +=1
    rtim3+=1
    #for y in ary:
        #for x in arx:
    tt = np.transpose(((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1))
    print(tt)
    ar_wave0[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ ary/7)+1)*255
    ar_wave1[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ 2*rtim/400*ary**2/7/3)+1)*255
    ar_wave2[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ 2*rtim/400*ary**2/7)+1)*255

    
    p[0,:] = ar_wave0.flatten()#+ar_wave1.flatten())
    p[1,:] = ar_wave1.flatten()#+ar_wave1.flatten())
    p[2,:] = ar_wave2.flatten()#+ar_wave1.flatten())
    
    
    ppm = np.mean(p[:,:])
    
    
    xdiv = (.5*np.sin(rtim/50)+.5)*20 + 14-14/2
    ydiv = (.5*np.sin(rtim/50)+.5)*49 + 49-49/2
        
    p = gaussian_filter1d(p, sigma=2)
    return p
cnt3 = 0
oods = np.linspace(1,749,749//4).astype(int)
evs = np.linspace(0,748,749//4).astype(int)
cy = 0
def visualize_breathe(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, cy, oods, evs

    
    rtim +=1
    rtim3+=1
    for y in ary:
        ar_wave0[arx,y] = ((np.sin(y*np.pi/(49))/2 + np.sin(arx*np.pi/(14))/4))*(.5*np.sin(rtim/25)+.5)*255
    
    if np.mean(p[:,:])<1 and rtim3 >15:
        rtim3 = 0
        cy+=1
        coo[0] = (.5*np.sin(rtim/30)+.5)**.5
        coo[1] = (.5*np.sin(rtim/30+30/3)+.5)**.5
        coo[2] = (.5*np.sin(rtim/30+2*30/3)+.5)**.5
    
        
    p[0,:] = coo[0]*ar_wave0.flatten()
    p[1,:] = coo[1]*ar_wave0.flatten()
    p[2,:] = coo[2]*ar_wave0.flatten()
    ppm = np.mean(p[:,:])
    if cy>1:
        p[0,oods] = p[1,oods]
        p[1,oods] = p[2,oods]
        if cy>3:
            p[1,evs] = p[2,evs]
            p[2,evs] = p[0,evs]
            if cy>7:
                p[1,oods] = p[2,evs]
                p[2,oods] = p[1,evs]  
        #p[0,oods] = p[1,oods]
        
    p = gaussian_filter1d(p, sigma=.2)
    return p
ard = 1
cyc=0
bthe = 0
thresh_bthe = 0.5
def visualize_breathe2(y):
    global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.
    ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    
    rtim +=1
    rtim3+=1
    bthe+=1
    #if ty>thresh_bthe and bthe>10:
        #cyc = 0
        #if bthe<30:
            #thresh*=1.1
            #print("Threshold Change, tic")
            #print(thresh)
        #elif bthe>60:
            #thresh*=.9
            #print("Threshold Change, tic")
            #print(thresh)
    if ty>thresh_bthe:
        cyc = 0
    
    if cyc==0:
        for y in ary:
            ar_wave0[arx,y] = ((np.sin(y*np.pi/(49))/2 + np.sin(arx*np.pi/(14))/4))*(.5*np.sin(rtim/10+y/5+arx/5)+.5)*255
    else:
        for y in ary:
            ar_wave0[arx,y] = ((np.sin(y*np.pi/(49))/2 + np.sin(arx*np.pi/(14))/4))*(.5*np.sin(rtim/10+y/5-arx/5)+.5)*255
    if rtim3 >30 and ty>thresh_bthe:
        rtim3 = 0
        bthe=0
        cy+=1
        cyc = 1
        
    for x in arx:
        coo[0] = x*(.5*np.sin(rtim/22)+.5)**.5
        coo[1] = x*(.5*np.sin(rtim/22+22/3)+.5)**.5
        coo[2] = x*(.5*np.sin(rtim/22+2*22/3)+.5)**.5
        
    p[0,:] = coo[0]*ar_wave0.flatten()
    p[1,:] = coo[1]*ar_wave0.flatten()
    p[2,:] = coo[2]*ar_wave0.flatten()
    ppm = np.mean(p[:,:])
    if cy>1:
        p[0,oods] = p[1,oods]
        p[1,oods] = p[2,oods]
        if cy>3:
            p[1,evs] = p[2,evs]
            p[2,evs] = p[0,evs]
            if cy>5:
                p[1,oods] = p[2,evs]
                p[2,oods] = p[1,evs]
                if cy>7:
                    p[0,oods] = p[2,evs]
                    p[2,oods] = p[0,evs]
                    if cy>9:
                        cy = 0

        
    p = gaussian_filter1d(p, sigma=.2)
    
    return p
def visualize_spunk(y):
    global p, cnt2, pix, cnt3, sl
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    
    m1 = np.mean(y[:4])
    m2 = np.mean(y[5:8])
    m3 = np.mean(y[9:12])
    #print(m2)
    cnt2 = cnt2+1
    cnt3+=1
    sl+=1
    print(m2)
    if m1 > 10 and cnt2>10 or cnt2>50:
        ppp = rn.randint(0,pix)
        p[0,ppp] = int(.5*(np.sin(sl/22+0)+1)*255)
        p[1,ppp] = int(.5*(np.sin(sl/22+22/3)+1)*255)
        p[2,ppp] = int(.5*(np.sin(sl/22+22*2/3)+1)*255)
        tu = rn.randint(1,3)
        p[:, tu:] = p[:, :-tu]
        cnt2 = 0
        
    if m2 > 10 and cnt3>10 or cnt3>50:
        ppp = rn.randint(0,pix)
        p[0,ppp] = int(.5*(np.sin(sl/22+0)+1)*255)
        p[1,ppp] = int(.5*(np.sin(sl/22+22/3)+1)*255)
        p[2,ppp] = int(.5*(np.sin(sl/22+22*2/3)+1)*255)
        tu = rn.randint(1,3)
        p[:, tu:] = p[:, :-tu]
        cnt3 = 0
    if cnt2>10 and cnt3>10:
        p = gaussian_filter1d(p, sigma=0.35)
    

    return np.concatenate((p, p[:, ::-1]), axis=1)   
    
rowl = np.linspace(0,49,50).astype(int)
rowl2 = np.linspace(0,24,25).astype(int)
sl = 0
dire = 0
rowl_thresh = 20
rowl_time = 25
sl_rowl = 0
def visualize_rowl(y):
    global p, rowl, cnt, sl_rowl, dire, rowl_thresh, rowl_time, SS
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    cnt +=1
    sl_rowl+=1
    m2 = np.mean(y[28:])
    print(SS)
    p[0,rowl] = int(.5*(np.sin(sl_rowl/rowl_time+0)+1)*255)
    p[1,rowl] = int(.5*(np.sin(sl_rowl/rowl_time+rowl_time/3)+1)*255)
    p[2,rowl] = int(.5*(np.sin(sl_rowl/rowl_time+2*rowl_time/3)+1)*255)   
    if np.min(rowl) == 0:
        dire = 1
    elif np.max(rowl) >= SS-50:
        dire = 0
    if m2 >rowl_thresh and cnt >15 and dire == 1:
        rowl+=50
        cnt = 0
        p[:,:]= 0
    if m2 >1.5*rowl_thresh and cnt >15 and dire == 0:
        rowl-=50
        cnt = 0
        p[:,:]= 0    
    if m2 >.5*rowl_thresh and dire == 0:
        p[0,rowl-50] = int(.5*(np.sin(sl_rowl/rowl_time+2*rowl_time/3)+1)*255)
        p[1,rowl-50] = int(.5*(np.sin(sl_rowl/rowl_time+rowl_time/3)+1)*255)
        p[2,rowl-50] = int(.5*(np.sin(sl_rowl/rowl_time + 0)+1)*255)
    if cnt >30:
        rowl_thresh *=.75
    

    return np.concatenate((p, p[:, ::-1]), axis=1)

rowl2 = np.linspace(0,24,25).astype(int)
sl = 0
dire = 0
phl = 25
def visualize_rowl2(y):
    global p, rowl2, cnt, sl, dire, phl
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    cnt +=1
    sl+=1
    m2 = np.mean(y[28:])
    
    if np.mean(y[28:]) < 1:
        p[:, 1:] = p[:, :-1]
        return np.concatenate((p, p[:, ::-1]), axis=1)
    p[0,rowl2] = int(.5*(np.sin(sl/phl +  0     )+1)*255)
    p[1,rowl2] = int(.5*(np.sin(sl/phl +  phl/3 )+1)*255)
    p[2,rowl2] = int(.5*(np.sin(sl/phl +2*phl/3) +1)*255)
    if np.min(rowl2) == 0:
        dire = 1
    elif np.max(rowl2) == 249:
        dire = 0
    if m2 >10 and cnt >15 and dire == 1:
        rowl2+=25
        cnt = 0
        p[:,:]= 0
    if m2 >10 and cnt >15 and dire == 0:
        rowl2-=25
        cnt = 0
        p[:,:]= 0    
    if m2 >20 and dire == 0:
        p[0,rowl2-25] = int(.5*(np.sin(sl/phl+2*phl/3)+1)*255)
        p[1,rowl2-25] = int(.5*(np.sin(sl/phl+phl/3)+1)*255)
        p[2,rowl2-25] = int(.5*(np.sin(sl/phl + 0)+1)*255)
        
    if m2 >20 and np.max(rowl2+25)<250:

        p[0,rowl2+25] = int(.5*(np.sin(sl/phl+2*phl/3)+1)*255)
        p[1,rowl2+25] = int(.5*(np.sin(sl/phl+phl/3)+1)*255)
        p[2,rowl2+25] = int(.5*(np.sin(sl/phl + 0)+1)*255)
    

    return np.concatenate((p, p[:, ::-1]), axis=1)

rowl3 = np.linspace(0,14,15).astype(int)
rowl4 = np.linspace(0,14,15).astype(int)
def visualize_rowl3(y):
    global p, rowl3, cnt, sl, dire, phl, pix
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    cnt +=1
    sl+=1
    m2 = np.mean(y[28:])
    
    if np.mean(y[28:]) < 1:
        p[:, 1:] = p[:, :-1]
        return np.concatenate((p, p[:, ::-1]), axis=1)
    
    p[0,rowl3] = int(.5*(np.sin(sl/phl +  0     )+1)*255)
    p[1,rowl3] = int(.5*(np.sin(sl/phl +  phl/3 )+1)*255)
    p[2,rowl3] = int(.5*(np.sin(sl/phl +2*phl/3) +1)*255)
    if np.min(rowl3)-25 <= 0:
        dire = 1
    elif np.max(rowl3)+25 >= 750//2:
        dire = 0
    if m2 >10 and cnt >15 and dire == 1:
        rowl3+=25
        cnt = 0
        p[:,rowl3-25]= 0
    if m2 >10 and cnt >15 and dire == 0:
        rowl3-=25
        cnt = 0
        p[:,rowl3+25]= 0    
    if m2 >20 and dire == 0:
        p[0,rowl3-25] = int(.5*(np.sin(sl/phl+2*phl/3)+1)*255)
        p[1,rowl3-25] = int(.5*(np.sin(sl/phl+phl/3)+1)*255)
        p[2,rowl3-25] = int(.5*(np.sin(sl/phl + 0)+1)*255)
        
    if m2 >20 and np.max(rowl3+25)<250:

        p[0,rowl3+25] = int(.5*(np.sin(sl/phl+2*phl/3)+1)*255)
        p[1,rowl3+25] = int(.5*(np.sin(sl/phl+phl/3)+1)*255)
        p[2,rowl3+25] = int(.5*(np.sin(sl/phl + 0)+1)*255)
    
    p2 = 0*p
    p2[0,:] = p[1,:]
    p2[1,:] = p[2,:]
    p2[2,:] = p[0,:]

    return np.concatenate((p, p2[:, ::-1]), axis=1)
nn = 0
ss3 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
tr1 = 0
qw = 0
cnt = 0
t1 = 0
cntp2 = 0
cntp3 = 0
sweetsc = 0
def visualize_sweetscroll(y):
    global p, nn, ss3, tr1, qw, kz, mp, t1, cnt, t2, dT, cntp2, cntp3, sweetsc
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = 2*int(np.max(y[:len(y) // 3]))
    g = 2*int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = 2*int(np.max(y[2 * len(y) // 3:]))
    # Scrolling effect window
    
    #p = gaussian_filter1d(p, sigma=0.2)
    #if sweetsc == 0:
        #nn=0
        #kz=0
        #ss= np.array([0, 0, 0, 0, 0, 0, 0, 0])
        #qw = 0
        #sweetsc += 1
        #print(sweetsc)
    nn += 1
    kz+=1
    p[0, ss3] = g
    p[1, ss3] = r
    p[2, ss3] = b
    tr = np.mean(y)
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    if tr > 5:
        tr1 +=1
        if tr1>100 and nn>50:
            #p[:,:] = 0
            ss3[qw] = rn.randint(0,350)
            nn = 0
            qw += 1
            #print(qw)
            if qw>6:
                qw = 0
    # Update the LED strip
    #print(kz)
    #p = gaussian_filter1d(p, sigma=0.2)
    #p[:,:] = np.floor(p[:,:])
        #symmetry change up based on loop count
    cntp2+=1
    
    #Cool reshaping function that just doesnt work well here, probably elsewhere tho
    if cntp2>50 and cntp3==0:
        cntp3+=1
        
        
        #p2 = np.reshape(p[0,:],(25,15))
        #p3 = np.reshape(p[1,:],(25,15))
        #p4 = np.reshape(p[2,:],(25,15))
    
        #p2 = np.transpose(p2)
        #p3 = np.transpose(p3)
        #p4 = np.transpose(p4)
    
        #p2 = p2.flatten()
        #p3 = p3.flatten()
        #p4 = p4.flatten()
        #p[0,:] = p2
        #p[1,:] = p3
       # p[2,:] = p4
        #cntp2 = 0
    if cntp2>100:
        cntp2 = 0
        cntp3 = 0
    if kz <mp:
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    elif kz > mp-1 and kz < 2*mp:
        return np.concatenate((p[:, ::-1], p), axis=1) #reversed symmetry
    else:
        s[0] += 1
        if s[0] > mp:
           kz = 0
        return np.concatenate((p, p         ), axis=1) #no symmetry

#coll2 = np.array([23, 24, 25, 49, 50, 51, 73, 74, 75, 123, 124, 125, 173, 174, 175, 224, 225]) #column of the net
#coll2 = np.array([1, 2, 3])
#coll2 = np.array([172, 0, 125, 220, 100])
coll2 = np.linspace(23,SS,rn.randint(7,13)).astype(int)
jit = 0
fwd = 1
sl = 0
ccn = 0
fwd2 = 1
qq2 = 0
qq = 0
hg = 0
ffi = 0.3
thresh7 = 10
def visualize_slide(y):
    """Effect that expands from the center with increasing sound energy"""
    global p, slide, coll2, jit, fwd, sl, ccn, fwd2, coll3, hg, qq, qq2, ffi, thresh7,SS
    y2 = y**2
    gain.update(y2)
    y2 /= gain.value
    y2 *= 255.0
    m2 = np.mean(y2[28:])
    
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    sl+=1
    ccn+=1
    
    arq = int(.5*(np.sin(qq/50)+1)*255)
    ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    qq2+=1
    qq+=1
    
    if m2>thresh7 or qq>60:
        hg+=1
        if qq>10:
            if qq<15:
                thresh7=1.1*thresh7
                print("Threshold Change, slide")
                print(thresh7)
            elif qq>40:
                thresh7*=.9
                print("Threshold Change, slide")
                print(thresh7)
            if np.max(coll2) >348:
                fwd = 0
            if np.min(coll2) < 2:
                fwd = 1
                
            if fwd == 1:
                p[:,:] = 0
                coll2 += 1
            elif fwd == 0:
                p[:,:] = 0
                coll2 -= 1
            #print(coll2)
            hg =0 
            qq = 0
    if m2>2*thresh7:
        p[:,:] = 0
        coll2 = np.linspace(3,SS,rn.randint(25,55)).astype(int)
        ffi = rn.randint(3, 5)/10
    p[0,coll2] = int(.5*(np.sin(sl/25+0)+1)*255)
    p[1,coll2] = int(.5*(np.sin(sl/25+25/3)+1)*255)
    p[2,coll2] = int(.5*(np.sin(sl/25+2*25/3)+1)*255)
    
    p[0, :] = gaussian_filter1d(p[0, :], sigma=ffi)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=ffi)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=ffi)
    
    #Reorder colors for half the net
    p2 = 0*p
    p2[0,:] = p[1,:]
    p2[1,:] = p[2,:]
    p2[2,:] = p[0,:]
    #p[0,:len(p[0,:])//2] = p[1,len(p[1,:])//2::]
    #p[1,:len(p[0,:])//2] = p[2,len(p[2,:])//2::]
    #p[2,:len(p[0,:])//2] = p[0,len(p[0,:])//2::]
    
    return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin

nn = 0
#ss = np.array([0, 0, 0, 0, 0])
tr1 = 0
qw = 0
sym = 0
x = 0
tim = 200
ss=ss3
def visualize_scroll(y):
    """Effect that originates in the center and scrolls outwards"""
    global p, nn, ss, tr1, qw, kz, sym, x, tim
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = 2*int(np.max(y[:len(y) // 3]))
    g = 2*int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = 2*int(np.max(y[2 * len(y) // 3:]))
    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)
    nn += 1
    kz+=1
    p[0, ss] = g
    p[1, ss] = r
    p[2, ss] = b
    tr = np.mean(y[28:])
    if tr > 20:
        ss[qw] = rn.randint(0,350)
        nn = 0
        qw += 1
        if qw>6:
            qw = 0
            ss = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            if sym == 0:
                sym = 1
            elif sym == 1:
                sym = 0
    # Update the LED strip
    x = x+1
    
    p = gaussian_filter1d(p, sigma=0.1)
    if tr>100:
        p2 = 0*p
        p2[0,:] = p[1,:]
        p2[1,:] = p[2,:]
        p2[2,:] = p[0,:]
        p = gaussian_filter1d(p, sigma=0.3)
        return np.concatenate((p, p2[:, ::-1]), axis=1)
    
        #symmetry change up based on loop count
    if sym == 0:
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    else: 
        return np.concatenate((p[:, ::-1], p), axis=1) #reversed symmetry

def visualize_simplescroll(y):
    """Effect that originates in the center and scrolls outwards"""
    global p, pix
    y = y**2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0
    r = 2*int(np.max(y[:len(y) // 3]))
    g = 2*int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = 2*int(np.max(y[2 * len(y) // 3:]))
    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    #p *= 0.98
    #p = gaussian_filter1d(p, sigma=0.2)
    # Create new color originating at the center
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    p[0, pix] = r
    p[1, pix] = g
    p[2, pix] = b
    return np.concatenate((p[:, ::-1], p), axis=1)
c1 = 0
c2 = 0
red = rn.randint(100,255)
gr = rn.randint(100,255)
bl = rn.randint(100,255)
red2 = rn.randint(0,255)
gr2 = rn.randint(0,255)
bl2 = rn.randint(0,255)
u2 = rn.randint(0,120)
w2 = rn.randint(1,3)
it = 0
k3 = 1
it2 = 0
v1 = 0
v2 = 0
v3 = 0
v4 = 0
en1 = 0
coll = np.array([24, 25, 26, 124, 125, 74, 75, 174]) #column of the net
#coll = np.array([24, 25, 26, 122, 123, 124, 125, 74, 75, 76, 77, 174]) #column of the net
tip = np.array([-1, 1, 1, -1, 1, -1, 1, -1]) #rement so that column moves together
#tip = np.array([-1,  1,  1, 1, -1, 1, -1, 1, -1, 1, -1 ,1, -1]) #increment so that column moves together

it3 = 0
o1 = 149
o2 = 125
o3 = 240
nn = 0
trip2 = 0
up = np.array([172, 0, 125, 220, 100])
s = np.array([0,0,0])
y_prev = [0]
rty = 50
pix = (config.N_PIXELS / 2) - 1
odds = np.linspace(1,374,374//2).astype(int)
evens = np.linspace(1,372,374//2).astype(int)

cnt1 = 0
phum = np.array([0,25/3,2*25/3])
trig1 = 0
def visualize_tetris(y):
    global p, rty, cnt1, odds, kz, phum, cnt2, trig1,evens,pix
    cnt1 +=1
    cnt2+=1
    kz+=1
    if trig1 == 0:
        p[0,odds] = .5*int(.5*(np.sin(cnt2/25+phum[0])+1)*255)
        p[1,odds] = .5*int(.5*(np.sin(cnt2/25+phum[1])+1)*255)
        p[2,odds] = .5*int(.5*(np.sin(cnt2/25+phum[2])+1)*255)
    if cnt1>200 and cnt1<400:
        p[:,:] = 0
        p[0,evens] = .5*int(.5*(np.sin(cnt2/25+phum[0])+1)*255)
        p[1,evens] = .5*int(.5*(np.sin(cnt2/25+phum[1])+1)*255)
        p[2,evens] = .5*int(.5*(np.sin(cnt2/25+phum[2])+1)*255)
        odds = np.linspace(0,pix,rn.randint(17,178)).astype(int)
    if cnt1>=400:
        p[:, 1:] = p[:, :-1]
        if cnt1>500:
            cnt1 = 0
            evens = np.linspace(0,pix,rn.randint(17,178)).astype(int)
        
    p2 = 0*p
    p2[0,:] = p[1,:]
    p2[1,:] = p[2,:]
    p2[2,:] = p[0,:]
 
    if trig1 == 0:
        return np.concatenate((p, p2         ), axis=1) #no symmetry
    
    #symmetry change up based on loop count
    if kz <200:
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    elif kz > 199 and kz < 400:
        return np.concatenate((p[:, ::-1], p), axis=1) #reversed symmetry
    else:
        s[0] += 1
        if s[0] > 200:
            kz = 0
        return np.concatenate((p, p         ), axis=1) #no symmetry        
it = 0
trig1 = 0
def visualize_slow_wave(y):
    global p, cnt1,cnt2,phum, kz, it,trig1
    kz+=1
    cnt1+=1
    cnt2+=1
    num = int(.5*(np.sin(cnt1/50)+1)*374)
    print(num)
    p[0,num] = .5*int(.5*(np.sin(cnt2/25+phum[0])+1)*255)
    p[1,num] = .5*int(.5*(np.sin(cnt2/25+phum[1])+1)*255)
    p[2,num] = .5*int(.5*(np.sin(cnt2/25+phum[2])+1)*255)
    p = gaussian_filter1d(p, sigma=.35)
    p2 = 0*p
    p2[0,:] = p[1,:]
    p2[1,:] = p[2,:]
    p2[2,:] = p[0,:]
    #symmetry change up based on loop count
    if num == 373:
        it+=1
        if it>4:
            trig1 = 1
            it = 0
    if num == 0:
        it+=1
        if it>4:
            trig1 = 0
            it = 0
            p[:,:] = 0
            
    if trig1 == 0:
        return np.concatenate((p, p2[:, ::-1]), axis=1) #typical symmetry about origin
    elif trig1==1:
        return np.concatenate((p[:, ::-1], p), axis=1) #reversed symmetry
    else:
        s[0] += 1
        if s[0] > 400:
            kz = 0
            p[:,:] = 0
        return np.concatenate((p, p         ), axis=1) #no symmetry  
def visualize_energy(y):
    """Effect that expands from the center with increasing sound energy"""
    global p, pnorm, k3, cinc, kk, kz, c1, u, w, red, bl, gr, c2, u2, w2, red2, bl2, gr2, \
           y_prev, it, t1, it2, point, v1, v2, v3, v4, en1, coll, tip, it3, trip, o1, o2, o3, trip2, nn, up1, s
    
    
    kz +=1 #LOOP COUNTER
    #print(type(kz))
    y = np.copy(y)
    y2 = y**2
    gain.update(y)
    y /= gain.value
    diff = y-y_prev
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    #print(y)
    # Map color channels according to energy in the different freq bands
    scale = 0.7
    r = int(np.mean(y[:len(y) // 3]**scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**scale))
    
    #p[:,:] = 0
    
    #triggers - are you triggered yet?
        #taking the mean of the frequency vs power bins, number of bins = len(y)
    trig1 = np.mean(y[int(len(y)/2):]) #Lower half
    trig2 = np.mean(y[:len(y)]) #higher half
    trig3 = np.mean(y[:int(len(y)/4)]) #lowest 1/4
    trig4 = np.mean(y[int(len(y)/4):]) #highest 1/4 
    
    #thresholds - we should normalize these eventually (or does the mic already do that)
    thres1 = 20
    thres2 = 150
    thres3 = 50
    
    y2 /= gain.value
    y2 *= 255.0
    r = int(np.max(y2[:len(y2) // 3]))
    g = int(np.max(y2[len(y2) // 3: 2 * len(y2) // 3]))
    b = int(np.max(y2[2 * len(y2) // 3:]))
    
   
    #Inward outward wave
    if trig1 > thres1 or en1>0: #Trigger on the lower half or if its already going
        if en1<150: #en1 is just a counter
            p[:, 1:] = p[:, :-1] #Scrolling effect window
            #p *= 0.98 
            p = gaussian_filter1d(p, sigma=0.2)
            p[0, 0] = g
            p[1, 0] = r
            p[2, 0] = b
            en1+=1
        if en1>149:
            p[:,:] = 0
            p[0,coll] = o1
            p[1,coll] = o2
            p[2,coll] = o3
            it3 +=1
            
            if trig3>thres3 or it3>10:
            
                coll = coll + tip
                it3 = 0
                if coll[0] == -25:
                    tip = -tip
                    trip   = 1
                    trip2 += 1
                    o1 = rn.randint(100,250)
                    o2 = rn.randint(100,250)
                    o3 = rn.randint(100,250)
                if coll[0] == 24 and trip == 1:
                    tip = -tip
                    trip = 0
                    trip2 +=1
                    o1 = rn.randint(1,250)
                    o2 = rn.randint(1,250)
                    o3 = rn.randint(1,250)
                if trip2>2:
                    en1 = 0
    # End Inward Outward wave
    # START  in2out train
    if np.mean(y)>30 or it>0:
        p[0,k3:k3+7] = np.mean(y[0:int(len(y)/3)])/np.mean(y)*255#rn.randint(200,255)
        p[1,k3:k3+7] = np.mean(y[int(len(y)/3):int(2*len(y)/3)])/np.mean(y)*255#rn.randint(100,255)
        p[2,k3:k3+7] = np.mean(y[int(2*len(y)/3):])/np.mean(y)*255#rn.randint(100,255)
        it += 1
        if it>50:
            k3 += 8
            it = 1
    if k3>int((config.N_PIXELS / 2) - 1):
        k3=0
        it = 0
    #End In2out train
    # Semi Standard Energy #
    if trig2>thres2:#140: #Trigger on the upper half 
        p[0, 0:r] = rn.randint(150,255) 
        p[0, r:] = 0
        p[1, 0:g] = rn.randint(150,255) 
        p[1, g:] = 0
        p[2, 0:b] = rn.randint(150,255)
        p[2, b:] = 0
        p_filt.update(p)
        p = np.round(p_filt.value)
        en1 = 0
    #End Semi Standard Energy
    
    #Attempted beat keeping - kinda works, but not usefully
    #beat = np.mean(y)
    #if beat>20 and it==0:
        #it = it + 1
        #t1 = time.time()
    #if beat>20 and it ==1:
        #it = 0
        #t2 = time.time()
        #dT = t2 - t1
    
    #Upward rain, up = np.array([172, 0, 125, 220, 100])
    if trig3 > 50 or trig4 > 50: #Trigger easily
        if up[1]>30:
            up[0] -=50
            up[1] = 0
            up[2:4] = rn.randint(150,250)

        else:
            up[1] +=1
        if up[0]<25:
            up[0] = 174
        
        p[0,up[0]-2:up[0]+2] = up[2]*255
        p[1,up[0]-2:up[0]+2] = up[3]*255
        p[2,up[0]-2:up[0]+2] = up[4]*255
    

    y_norm = y/np.max(y)
   
    if np.mean(y[0:8]>20) or it2 == 0:
        if it2 == 0:
            point = rn.randint(125,170)
            v1 = np.mean(y_norm[:int(len(y_norm)/3)])*255
            v2 = np.mean(y_norm[int(len(y_norm)/3):int(len(y_norm)*2/3)])*255
            v3 = np.mean(y_norm[int(len(y_norm)*2/3):])*255
            it2 = 1
            v4 = rn.randint(3,7)
        else:
            it2+=1
            if it2>50:
                it2 = 0
        p[0,point:point+v4] = v1
        p[1,point:point+v4] = v2
        p[2,point:point+v4] = v3
    
    # START  in2out train
    if np.mean(y)>30 or it>0:
        p[0,k3:k3+7] = np.mean(y[0:int(len(y)/3)])/np.mean(y)*255#rn.randint(200,255)
        p[1,k3:k3+7] = np.mean(y[int(len(y)/3):int(2*len(y)/3)])/np.mean(y)*255#rn.randint(100,255)
        p[2,k3:k3+7] = np.mean(y[int(2*len(y)/3):])/np.mean(y)*255#rn.randint(100,255)
        it += 1
        if it>50:
            k3 += 8
            it = 1
    if k3>int((config.N_PIXELS / 2) - 1):
        k3=0
        it = 0
    #End In2out train
   
    # High note block tracker
    if trig2 > 50: 
        if c1<1: #Define color and shape
            u = rn.randint(0,120)
            w = rn.randint(1,3)
            red = rn.randint(0,255)
            gr = rn.randint(0,255)
            bl = rn.randint(0,255)
            nn=rn.randint(3,9) #Used to be 5 only
        p[0,u:u+nn] = red
        p[1,u:u+nn] = gr
        p[2,u:u+nn] = bl
        c1+=1
        if c1>50: #when to reset
            c1=0
            
    #Low block tracker        
    if trig1 > 20:
        if c2<1:
            u2 = rn.randint(0,120)
            w2 = rn.randint(1,3)
            red2 = rn.randint(0,255)
            gr2 = rn.randint(0,255)
            bl2 = rn.randint(0,255)
        p[0,u2] = 255
        p[1,u2] = 255 
        p[2,u2] = 255 
        nn2=3
        p[0,u2:u2+nn2] = red2
        p[1,u2:u2+nn2] = gr2
        p[2,u2:u2+nn2] = bl2
        c2+=1
        
        if c2>50:
            c2=0

    
    #symmetry change up based on loop count
    if kz <200:
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    elif kz > 199 and kz < 400:
        return np.concatenate((p[:, ::-1], p), axis=1) #reversed symmetry
    else:
        s[0] += 1
        if s[0] > 200:
            kz = 0
        return np.concatenate((p, p         ), axis=1) #no symmetry
    
    #return np.concatenate((p, p[:, ::-1]), axis=1)
    #return np.concatenate((p[:, ::-1], p), axis=1)
s = np.array([0, 0, 0])
a = np.zeros((1,32))


m1 = 0
m2 = 0
def visualize_energy2(y):
    global p, pnorm, k, cinc, kk, kz, y_prev, i, c1, u, w, red, gr, bl, nn, s, a, m1, m2
    kz += 1
    
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    #y *= float((config.N_PIXELS // 2) - 1)
    trig1 = np.mean(y[int(len(y)/2):]) #Lower half
    trig2 = np.mean(y[:len(y)]) #higher half
    trig3 = np.mean(y[:int(len(y)/4)]) #lowest 1/4
    trig4 = np.mean(y[int(len(y)/4):]) #highest 1/4 
    trig = np.array([trig1, trig2, trig3, trig4])
    num = float((config.N_PIXELS // 2) - 1)
    #print(y)
    
    # Map color channels according to energy in the different freq bands
    scale = 0.7
    r = int(num*np.mean(y[:len(y) // 3]**scale))
    g = int(num*np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(num*np.mean(y[2 * len(y) // 3:]**scale))

    if np.sqrt(np.mean(y[0:32])) >.1 or c1>0: 
        if c1<1: #Define color and shape
            u = rn.randint(0,120)
            w = rn.randint(1,3)
            red = rn.randint(0,255)
            gr = rn.randint(0,255)
            bl = rn.randint(0,255)
            nn=rn.randint(3,9) #Used to be 5 only
        p[0,u:u+nn] = red
        p[1,u:u+nn] = gr
        p[2,u:u+nn] = bl
        c1+=1
        if c1>50: #when to reset
            c1=0
    else:
        p[:,:] =0
    p[:, 2:] = p[:, :-2]

    #p_filt.update(p)

    return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin



_prev_spectrum = np.tile(0.01, config.N_PIXELS // 2)
qq = 0

#a = np.array([0])
ar  = np.array([21, 23, 70, 71, 72, 73,  74,  75,  76,  77,  78, 79])
colm = np.linspace(0,config.N_PIXELS // 2-1, 50).astype(int)
qq2=0
thresh = 0.2
def visualize_tic(y):
    global p, qq, a, ar, colm, qq2, hg, thresh
    
    y = y**2
    gain.update(y)
    qq +=1
    y /= gain.value
    arq = int(.5*(np.sin(qq/50)+1)*255)
    ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    qq2+=1
    
    if ty>thresh or qq>50:
        hg+=1
        if qq>7:
            if qq<15:
                thresh*=1.1
                print("Threshold Change, tic")
                print(thresh)
            elif qq>25:
                thresh*=.9
                print("Threshold Change, tic")
                print(thresh)
            p[:,:] = 0
            colm = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(25,75)).astype(int)
            hg =0 
            qq = 0
        
    p[0,colm] = int(.5*(np.sin(qq2/25+0)+1)*255)
    p[1,colm] = int(.5*(np.sin(qq2/25+25/3)+1)*255)
    p[2,colm] = int(.5*(np.sin(qq2/25+25*2/3)+1)*255)
    
    p[0, :] = gaussian_filter1d(p[0, :], sigma=.4)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=.4)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=.4)
    return np.concatenate((p, p[:, ::-1]), axis=1)
def visualize_tic2(y):
    global p, qq, a, ar, colm, qq2, hg, thresh
    
    y = y**2
    gain.update(y)
    qq +=1
    y /= gain.value
    arq = int(.5*(np.sin(qq/50)+1)*255)
    ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    qq2+=1
    
    if ty>thresh or qq>40:
        hg+=1
        if qq>10:
            if qq<15:
                thresh*=1.1
                print("Threshold Change, tic")
                print(thresh)
            elif qq>25:
                thresh*=.89
                print("Threshold Change, tic")
                print(thresh)
            p[:,:] = 0
            colm = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(25,125)).astype(int)
            hg =0 
            qq = 0
        
    p[0,colm] = int(.5*(np.sin(qq2/50+0)+1)*255)
    p[1,colm] = int(.5*(np.sin(qq2/50+50/3)+1)*255)
    p[2,colm] = int(.5*(np.sin(qq2/50+50*2/3)+1)*255)
    
    p[0, :] = gaussian_filter1d(p[0, :], sigma=.1)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=.1)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=.1)
    return np.concatenate((p, p[:, ::-1]), axis=1)
qq = 249
cr=0
hg = 0
hg2 = 0
thresh_inc = 0.1
pix = config.N_PIXELS // 2 - 1
fwdd = 1
gau = .6
crg = [2,1,0]
colm2 = np.linspace(0,config.N_PIXELS // 2-1, 2).astype(int)
hg3=0
def visualize_tic3(y):
    global p, qq, a, ar, colm, qq2, hg3, thresh, fwdd, colm2, gau, crg
    
    y = y**2
    gain.update(y)
    qq +=1
    y /= gain.value
    arq = int(.5*(np.sin(qq/50)+1)*255)
    ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    qq2+=1
    
    if ty>thresh or qq>75:
       
        if qq>25:
            if qq<40:
                thresh*=1.1
                print("Threshold Change, tic")
                print(thresh)
            elif qq>50:
                thresh*=.9
                print("Threshold Change, tic")
                print(thresh)
            p[:,:] = 0
           
            colm = np.linspace(0,config.N_PIXELS // 2-1, 3+hg3).astype(int)
            colm2 = np.linspace(0,config.N_PIXELS // 2-1, 2+hg3).astype(int)
            print(hg)
            if fwdd ==1:
                hg3+=1
            elif fwdd == 0:
                hg3-=1
            if hg > 10:
                fwdd = 0
                crg = [0,2,1]
            if hg3 == 0:
                fwdd = 1
                gau+=1
                crg = [1,2,0]
            qq = 0
        
    p[0,colm] = int(.5*(np.sin(qq2/25+0)+1)*255)
    p[1,colm] = int(.5*(np.sin(qq2/25+25/3)+1)*255)
    p[2,colm] = int(.5*(np.sin(qq2/25+25*2/3)+1)*255)
    
    p[crg[0],colm2] = int(.5*(np.sin(qq2/25+0)+1)*255)
    p[crg[1],colm2] = int(.5*(np.sin(qq2/25+25/3)+1)*255)
    p[crg[2],colm2] = int(.5*(np.sin(qq2/25+25*2/3)+1)*255)
    
    p[0, :] = gaussian_filter1d(p[0, :], sigma=gau)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=gau)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=gau)
    return np.concatenate((p, p[:, ::-1]), axis=1)
def visualize_inc(y):
    global p, qq, cr, hg, hg2, thresh_inc, pix
    
    y = y**2
    gain.update(y)
    y /= gain.value
    beat = 0
    fg = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    
    if fg>thresh_inc:
        beat = 1
        p[0,rn.randint(0,pix)] = rn.randint(150,255)
        p[1,rn.randint(0,pix)] = rn.randint(150,255)
        p[2,rn.randint(0,pix)] = rn.randint(150,255)
        cr+=1
        hg = 0
        hg2 +=1
    else:
        hg+=1
        hg2 = 0
        if hg>30:
            thresh_inc=thresh_inc*.75
            print("Threshold down, inc")
            print(thresh_inc)
    if hg2-hg>30:
        thresh_inc=thresh_inc*1.25
        print("Threshold up, tic")
        print(thresh_inc)
    if cr>100 and beat == 1:
        p[:,:] = 0
        cr = 0
        

    p[0, :] = gaussian_filter1d(p[0, :], sigma=.25)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=.25)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=.25)
    return np.concatenate((p[:, ::-1], p), axis=1)
fwddd = 1
ghh = 0
gh2 =0
jk = 0
def visualize_inc2(y):
    global p, qq, cr, hg, hg2, thresh_inc, pix, fwddd, ghh, gh2, jk
    
    y = y**2
    gain.update(y)
    y /= gain.value
    beat = 0
    fg = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    
    if fg>thresh_inc and fwddd==1:
        beat = 1
        px1 = rn.randint(0,pix)
        px2 = rn.randint(0,pix)
        px3 = rn.randint(0,pix)
        
        p[0,px1] = rn.randint(100,200)
        p[1,px1] = rn.randint(100,200)
        p[2,px1] = rn.randint(100,200)
        
        p[0,px2] = rn.randint(50,150)
        p[1,px2] = rn.randint(100,150)
        p[2,px2] = rn.randint(150,250)
        
        p[0,px3] = rn.randint(0,100)
        p[1,px3] = rn.randint(0,100)
        p[2,px3] = rn.randint(0,100)
        
        cr+=1
        hg = 0
        hg2 +=1
    else:
        hg+=1
        hg2 = 0
        if hg>30:
            thresh_inc=thresh_inc*.75
            print("Threshold down, inc")
            print(thresh_inc)
    if hg2-hg>30:
        thresh_inc=thresh_inc*1.25
        print("Threshold up, tic")
        print(thresh_inc)
    if cr>100 and beat == 1:
        fwddd = 0
        cr = 0
    if fwddd == 0:
        p[:,rn.randint(0,pix)] = 0
        p[:,rn.randint(0,pix)] = 0
        p[:,rn.randint(0,pix)] = 0
        p[:,rn.randint(0,pix)] = 0
        ghh+=1
        if ghh>75:
            ghh=0
            fwddd=1
            gh2+=1
    if gh2>1:
        p[:,1:] = p[:,:-1]
        jk+=1
        if jk>75:
            gh2=0
            jk = 0

    p[0, :] = gaussian_filter1d(p[0, :], sigma=.25)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=.25)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=.25)
    return np.concatenate((p[:, ::-1], p), axis=1)     
p_prev = 0
#a = np.zeros((1,350))
mn = np.zeros((1,350))
c = np.zeros((1,350))
qe1 = 25
qe2 = 26
qe3 = 27
qew1 = np.linspace(0,pix, qe1).astype(int)
qew2 = np.linspace(0,pix, qe2).astype(int)
qew3 = np.linspace(0,pix, qe3).astype(int)
sthresh = .2
def visualize_spectrum(y):
    #print("tic")
    global p, qq, cr, hg, hg2, sthresh, qq2, colm, qew1, qew2, qew3, qe1, qe2, qe3
    y = y**2
    gain.update(y)
    qq +=1
    y /= gain.value
    arq = int(.5*(np.sin(qq/50)+1)*255)
    ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    qq2+=1
    per = np.mean(y[0:8])/np.max(y[0:8])
    print(per)
    per = 255*(per)**.25 

    if ty>sthresh:
        hg+=1
        if qq>20:
            if qq<25:
                sthresh*=1.125
                print("Threshold up, spectrum")
                print(sthresh)
            elif qq>50:
                sthresh*=.875
                print("Threshold down, spectrum")
                print(sthresh)
            p[:,:] = 0
            qew1 = np.linspace(0,pix, qe1).astype(int)
            qew2 = np.linspace(0,pix, qe2).astype(int)
            qew3 = np.linspace(0,pix, qe3).astype(int)
            qe1+=1
            qe2+=1
            qe3+=1
            colm = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(25,75)).astype(int)
            hg =0 
            qq = 0
        
    p[0,qew1] = per
    p[1,qew2] = per
    p[2,qew3] = per
    p[0, :] = gaussian_filter1d(p[0, :], sigma=.35)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=.35)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=.35)
    return np.concatenate((p[:, ::-1], p), axis=1)
    
    #return p
c5 = 0
ar  = np.array([21, 23, 70, 71, 72, 73,  74,  75,  76,  77,  78, 79])
ar2 = np.array([15, 17, 19, 65, 66, 67,  68,  69,  80,  81,  82, 83, 84])
yup = 1
x=0
du = 0
kz2 = 0


#C = np.arcsin(255)
#print(C)
cntb =0
cntb2 = 21
base_e = np.array([0, 100, 200, 300, 400, 500, 600, 700]).astype(int)
base_o = np.array([99, 199, 299, 399, 499, 599, 699]).astype(int)
up_e = np.array([50, 150, 250, 350, 450, 550, 650, 750]).astype(int)
up_o = np.array([49, 149, 249, 349, 449, 549, 649, 749]).astype(int)
bb = np.linspace(0,6,7).astype(int)
upp = 1
eth = .2
def visualize_energy_base(y):
    """Effect that expands from the center with increasing sound energy"""
    global p, cntb, cntb2, base_e, base_o, up_e, up_o, bb, upp, eth
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    cntb+=1
    cntb2+=1
    etr = np.mean(y)/np.max(y)

    rrr = 1*int(np.mean(y[:len(y) // 3]**scale))//3
    ggg = 1*int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))//3
    bbb = 1*int(np.mean(y[2 * len(y) // 3:]**scale))//3
    
    #ed2 = base_o-rrr
    # Assign color to different frequency regions
    
    if cntb2>5 and upp == 1 and etr>eth:
        #p[:,:] = 0   
        for i in bb:
            p[0, base_e[i]:base_e[i]+rrr] = int(.5*(np.sin(cntb/25+0+base_e[i]/10)+1)*255)
            p[1, base_e[i]:base_e[i]+ggg] = int(.5*(np.sin(cntb/25+25/3+2*base_e[i]/10)+1)*255)
            p[2, base_e[i]:base_e[i]+bbb] = int(.5*(np.sin(cntb/25+25*2/3+3*base_e[i]/10)+1)*255)
            
            p[0, up_e[i]+rrr:up_e[i]+50] = int(.5*(np.sin(cntb/25+0+up_e[i]/10)+1)*255)
            p[1, up_e[i]+ggg:up_e[i]+50] = int(.5*(np.sin(cntb/25+25/3+2*up_e[i]/10)+1)*255)
            p[2, up_e[i]+bbb:up_e[i]+50] = int(.5*(np.sin(cntb/25+25*2/3+3*up_e[i]/10)+1)*255)
            
            p[0, base_e[i]+rrr:base_e[i]+50] = 0
            p[1, base_e[i]+ggg:base_e[i]+50] = 0
            p[2, base_e[i]+bbb:base_e[i]+50] = 0
            
            p[0, up_e[i]:up_e[i]+rrr] = 0
            p[1, up_e[i]:up_e[i]+ggg] = 0
            p[2, up_e[i]:up_e[i]+bbb] = 0
        cntb2 = 0
    

    p[0, :] = gaussian_filter1d(p[0, :], sigma=3)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=3)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=3)
    
    #p[0,pix//2:pix] = p[1,0:pix//2]
    #p[1,pix//2:pix] = p[2,0:pix//2]
    #p[2,pix//2:pix] = p[0,0:pix//2]
    return p

def visualize_energy_classic(y):
    """Effect that expands from the center with increasing sound energy"""
    global p
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    rrr = 1*int(np.mean(y[:len(y) // 3]**scale))
    ggg = 1*int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    bbb = 1*int(np.mean(y[2 * len(y) // 3:]**scale))
    # Assign color to different frequency regions
    p[0, :rrr] = 255.0
    p[0, rrr:] = 0.0
    p[1, :ggg] = 255.0
    p[1, ggg:] = 0.0
    p[2, :bbb] = 255.0
    p[2, bbb:] = 0.0
    
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=2)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=2)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=2)
    # Set the new pixel value
    return np.concatenate((p[:, ::-1], p), axis=1)
thresh = .2
hg = 0
qq2 = 0
colm = np.linspace(0,config.N_PIXELS // 2-1, 50).astype(int)
sz_on = 0
def visualize_energy_gaps(y):
    """Effect that expands from the center with increasing sound energy"""
    global p, qq2, qq, colm, thresh, hg, sz_on, spot, sz
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    rrr = int(np.mean(y[:len(y) // 3]**scale))

    ggg = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    bbb = int(np.mean(y[2 * len(y) // 3:]**scale))
    # Assign color to different frequency regions
    p[0, :rrr] = 255.0
    p[0, rrr:] = 0.0
    p[1, :ggg] = 255.0
    p[1, ggg:] = 0.0
    p[2, :bbb] = 255.0
    p[2, bbb:] = 0.0
    #GAPS
    ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
    qq2+=1
    qq+=1
    if ty>thresh:
        hg+=1
        if qq>45:
            if qq<50:
                thresh*=1.125
                print("Threshold Change, tic")
                print(thresh)     
            elif qq>100:
                thresh*=.875
                print("Threshold Change, tic")
                print(thresh)
                sz_on = 0
            p[:,:] = 0
            colm = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(10,65)).astype(int)
            hg =0 
            qq = 0
            spot = rn.randint(0,249-9)
            sz = rn.randint(3,9)
            sz_on = 1
            
            p[2,spot:spot+sz]= int(.5*(np.sin(qq2/50+0)+1)*255)
            p[1,spot:spot+sz]= int(.5*(np.sin(qq2/50+50/3)+1)*255)
            p[0,spot:spot+sz]= int(.5*(np.sin(qq2/50+2*50/3)+1)*255) 
    
    if sz_on == 1:
        p[2,spot:spot+sz]= int(.5*(np.sin(qq2/50+0)+1)*255)
        p[1,spot:spot+sz]= int(.5*(np.sin(qq2/50+50/3)+1)*255)
        p[0,spot:spot+sz]= int(.5*(np.sin(qq2/50+2*50/3)+1)*255)
    
    p[0,colm] = int(.5*(np.sin(qq2/50+0)+1)*255)
    p[1,colm] = int(.5*(np.sin(qq2/50+50/3)+1)*255)
    p[2,colm] = int(.5*(np.sin(qq2/50+50*2/3)+1)*255)

   
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=.1)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=.1)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=.1)
    # Set the new pixel value
    return np.concatenate((p[:, ::-1], p), axis=1)

ewb = 25 #Speeeeed of our bump
ph2 = 20
def visualize_bump(y):
    global p, kz, S, c5, ar, yup, x, ar2, du, kz2, ph2, ewb, beat
    y2 = y**2
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    y2 /=gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    y2 *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    r = int(np.mean(y[:len(y) // 3]**scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**scale))
    # Assign color to different frequency regions
    st = kz
    kz+=1
    #Attempted beat keeping - kinda works, but not usefully
    #beat = np.mean(y)
    #if beat>20 and it==0:
        #it = it + 1
        #t1 = time.time()
    #if beat>20 and it ==1:
        #it = 0
        #t2 = time.time()
        #dT = t2 - t1

    thres = 300

    y /= np.max(y)
   
    ar3 = [11, 13, 60, 61, 62, 63,  64,  85,  86,  87,  88, 89] 
    ar4 = [5,  7,  9,  55, 56, 57,  58,  59,  90,  91,  92, 93, 94]
    ar5 = [1,  3,  50, 51, 52, 53,  54,  95,  96,  97,  98, 99]
    ar6 = [0,  2,  4,  45, 46, 47, 48, 49,  100, 101, 102, 103, 104]
    ar7 = [6,  8, 40, 41, 42, 43, 44, 105, 106, 107, 108, 109]
    ar8 = [10, 12, 14, 35, 36, 37, 38, 39, 110, 111, 112, 113, 114]
    pl = config.N_PIXELS // 2
    #if kz >20: 
        #p[:, :] = 0
        #if np.max(ar)>pl // 2 or np.max(ar2)>pl: 
            #ar2 -= 1
            #yup = -1
        #if np.min(ar)<-pl or np.min(ar2)<-pl:
            #yup = rn.randint(1,2)
        #ar  = ar + yup
        #ar2 = ar2 + yup
        #ar3 +=3
        #tog = 1
        #kz = 0
        #du = du+1
    #
    arq = int(.5*(np.sin(x/ewb)+1)*pl)
    du = 5
    p[0, arq:arq+du] = .5*(np.sin(np.pi*x/20+ph2)+1) *255
    p[1, arq:arq+du] = .5*(np.sin(np.pi*x/20+ph2/3)+1) *255
    p[2, arq:arq+du] = .5*(np.sin(np.pi*x/20+2*ph2/3)+1) *255

    x = x+1           
    kz2+=1
    
    #p_filt.update(p)
    #p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    m2 = np.mean(y2[28:])
    if m2 >10:
        ph2 = rn.randint(5,25)
    #if arq+du >= pl-2 or m2 >50:
        #arq = 0
    p[:, 1:] = p[:, :-1]
        #p[:,:] = 0
    #p[0, :] = gaussian_filter1d(p[0, :], sigma=.3)
    #p[1, :] = gaussian_filter1d(p[1, :], sigma=.3)
    #p[2, :] = gaussian_filter1d(p[2, :], sigma=.3)
        #x = 0
        #dud = 1
        #ph = [rn.randint(5,15), rn.randint(10,30)]
    #Change up the mapping symmetry
    #if kz2 < 200:
    return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    #if kz2 == 200:
        #du = 1 #reset the number of light blocks in the chain
        
        #p[:,:] = 0
        #s[0] += 1 #start counting these loops
    #if s[0] > 200: #reset  
        #kz2 = 0
        #return np.concatenate((p, p ), axis=1) #no symmetry
            
            
            
ar  = np.array([21, 23, 70, 71, 72, 73,  74,  75,  76,  77,  78, 79])
ar2 = np.array([15, 17, 19, 65, 66, 67,  68,  69,  80,  81,  82, 83, 84])
yup = 1
x=0
du = 1
ph = np.array([5, 20])
def visualize_wave(y):
    global p, kz, ar, ar2, du, kz2, ph, x, yup, k3, it, coll, o1, o2, o3, it3, tip, trip, trip2       
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    r = 1*int(np.mean(y[:len(y) // 3]**scale))
    g = 1*int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = 1*int(np.mean(y[2 * len(y) // 3:]**scale))
    # Assign color to different frequency regions
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=1)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=1)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=1)
    pl = config.N_PIXELS // 2
    trig3 = np.mean(y[:int(len(y)/4)]) #lowest 1/4
    thres3 = 50
    if kz >1: 
        
        if np.max(ar)>pl or np.max(ar2)>pl: 
            ar2 -= 1
            yup = -1
        if np.min(ar)<-pl or np.min(ar2)<-pl:
            yup = rn.randint(1,2)
        ar  = ar + yup
        ar2 = ar2 + yup
        #ar3 +=3
        #tog = 1
        kz = 0
        du = du+1
    #
    arq = int(.5*(np.sin(x/20)+1)*pl)
    #rint(np.mean(y[:int(len(y)/2)]))
    mm = np.mean(y[:int(len(y)/2)])/100
    #if arq+du == pl:
       #p[:, :] = 0
        #if du == 1:
            #du = du+1
    tim = 50 
    p[0, arq:arq+6] = (np.sin(np.pi*x/tim)+1)       *255
    p[1, arq:arq+6] = (np.sin(np.pi*x/tim+ph[0])+1) *255
    p[2, arq:arq+6] = (np.sin(np.pi*x/tim+ph[1])+1) *255 #mm*.5*
    
    x = x+1
    kz2+=1
    #step wave
    if np.mean(y)>30 or it>0:
        p[0,k3:k3+7] = np.mean(y[0:int(len(y)/3)])/np.mean(y)*255#rn.randint(200,255)
        p[1,k3:k3+7] = np.mean(y[int(len(y)/3):int(2*len(y)/3)])/np.mean(y)*255#rn.randint(100,255)
        p[2,k3:k3+7] = np.mean(y[int(2*len(y)/3):])/np.mean(y)*255#rn.randint(100,255)
        it += 1
        if it>50:
            k3 += 8
            it = 1
    if k3>int((config.N_PIXELS / 2) - 1):
        k3=0
        it = 0
    # end step wave
    #if mm>1:
        #p[:,:] = 0
    if mm<.2:
        #p[:,:] = 0
        p[0,coll] = o1
        p[1,coll] = o2
        p[2,coll] = o3
        it3 +=1
            
        if trig3>thres3 or it3>10:
            
            coll = coll + tip
            it3 = 0
            if coll[0] == -25:
                tip = -tip
                trip   = 1
                trip2 += 1
                o1 = rn.randint(100,250)
                o2 = rn.randint(100,250)
                o3 = rn.randint(100,250)
            if coll[0] == 24 and trip == 1:
                tip = -tip
                trip = 0
                trip2 +=1
                o1 = rn.randint(1,250)
                o2 = rn.randint(1,250)
                o3 = rn.randint(1,250)
    #Change up the mapping symmetry
    if kz2 < 100:
    #if s[0] == 0:
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    else:
        if kz2 == 100:
            du = 1 #reset the number of light blocks in the chain
            ph = [rn.randint(1,25), rn.randint(1,25)] #reset the color phase offset to something random
        s[0] += 1 #start counting these loops
        if s[0] > 100 and mm>0.75: #reset  
            kz2 = 0
            s[0] = 0
        return np.concatenate((p, p ), axis=1) #no symmetry
    return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin

def visualize_trans_wave(y):
    global p, kz, ar, ar2, du, kz2, ph, x, yup, k3, it, coll, o1, o2, o3, it3, tip, trip, trip2       
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
    # Scale by the width of the LED strip
    y *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    r = 1*int(np.mean(y[:len(y) // 3]**scale))
    g = 1*int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = 1*int(np.mean(y[2 * len(y) // 3:]**scale))
    # Assign color to different frequency regions
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=1)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=1)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=1)
    pl = config.N_PIXELS // 2
    trig3 = np.mean(y[:int(len(y)/4)]) #lowest 1/4
    thres3 = 50
    if kz >1: 
        
        if np.max(ar)>pl or np.max(ar2)>pl: 
            ar2 -= 1
            yup = -1
        if np.min(ar)<-pl or np.min(ar2)<-pl:
            yup = rn.randint(1,2)
        ar  = ar + yup
        ar2 = ar2 + yup
        #ar3 +=3
        #tog = 1
        kz = 0
        du = du+1
    #
    arq = int(.5*(np.sin(x/20)+1)*pl)
   
    mm = np.mean(y[:int(len(y)/2)])/100

    tim = 50 
    p[0, arq:arq+6] = (np.sin(np.pi*x/tim)+1)       *255
    p[1, arq:arq+6] = (np.sin(np.pi*x/tim+ph[0])+1) *255
    p[2, arq:arq+6] = (np.sin(np.pi*x/tim+ph[1])+1) *255 #mm*.5*
    
    x = x+1
    kz2+=1
    #step wave
    if np.mean(y)>30 or it>0:
        p[0,k3:k3+7] = np.mean(y[0:int(len(y)/3)])/np.mean(y)*255#rn.randint(200,255)
        p[1,k3:k3+7] = np.mean(y[int(len(y)/3):int(2*len(y)/3)])/np.mean(y)*255#rn.randint(100,255)
        p[2,k3:k3+7] = np.mean(y[int(2*len(y)/3):])/np.mean(y)*255#rn.randint(100,255)
        it += 1
        if it>50:
            k3 += 8
            it = 1
    if k3>int((config.N_PIXELS / 2) - 1):
        k3=0
        it = 0
    # end step wave
    #if mm>1:
        #p[:,:] = 0
    if mm<.2:
        #p[:,:] = 0
        p[0,coll] = o1
        p[1,coll] = o2
        p[2,coll] = o3
        it3 +=1
            
        if trig3>thres3 or it3>10:
            
            coll = coll + tip
            it3 = 0
            if coll[0] == -25:
                tip = -tip
                trip   = 1
                trip2 += 1
                o1 = rn.randint(100,250)
                o2 = rn.randint(100,250)
                o3 = rn.randint(100,250)
            if coll[0] == 24 and trip == 1:
                tip = -tip
                trip = 0
                trip2 +=1
                o1 = rn.randint(1,250)
                o2 = rn.randint(1,250)
                o3 = rn.randint(1,250)
    #Change up the mapping symmetry
    p2 = np.reshape(p[0,:],(25,15))
    p3 = np.reshape(p[1,:],(25,15))
    p4 = np.reshape(p[2,:],(25,15))
    
    p2 = np.transpose(p2)
    p3 = np.transpose(p3)
    p4 = np.transpose(p4)
    
    p2 = p2.flatten()
    p3 = p3.flatten()
    p4 = p4.flatten()
    p[0,:] = p2
    p[1,:] = p3
    p[2,:] = p4
    
    return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    
def visualize_spunk2(y):
    global p, cnt2, pix, cnt3, sl, adr,cnt4
    y = y**2
    gain.update(y)
    y /= gain.value
    y *= 255.0
    m1 = np.mean(y[:4])
    m2 = np.mean(y[5:8])
    m3 = np.mean(y[9:12])
    cnt2+=1
    cnt3+=1
    cnt4+=1
    sl+=1
    #Side Bars
    nn = int(np.mean(y[:len(y) // 3]**.9))
    nn2 = int(np.mean(y[len(y) // 3:2*len(y)//3]**.9))
    nn3 = int(np.mean(y[len(y) // 2::]**.9))
    if nn>50:
        nn = 50
    if nn2>50:
        nn2 = 50
    if nn3>25:
        nn3 = 25
    nn4 = (nn+nn2+nn3)//3+1
    grr = np.linspace(0, nn, nn+1).astype(int)
    p[0,:nn4] = nn/nn4*255
    p[1,:nn4] = nn2/nn4*255
    p[2,:nn4] = nn3/nn4*255
    if cnt4>7:
        cnt4 = 0
        p[:,nn4:nn4+50] = 0
    #p[1,:nn2] = 255
    #p[1,nn2:nn2+50] = 0
    #p[2,:nn3] = 255
    #p[2,nn3:nn3+50] = 0
    
    p[:,:nn4 ] = gaussian_filter1d(p[:,:nn4 ], sigma=2)
    #p[1,:nn4] = gaussian_filter1d(p[1,:nn2], sigma=2)
    #p[2,:nn3] = gaussian_filter1d(p[2,:nn3], sigma=2)
    
    #p[:,0:int(m1)] = 255
    print(m1)
    if m1>5 or cnt2>15: 
        ppp = rn.randint(50,pix-50)
        ppp2 = rn.randint(1,9)
        co = rn.randint(25,40)
        aaa = [ppp,ppp+1,ppp+50, ppp+51]
        p[0,aaa] = int(.5*(np.sin(sl/co+0)+1)*255)
        p[1,aaa] = int(.5*(np.sin(sl/co+co/3)+1)*255)
        p[2,aaa] = int(.5*(np.sin(sl/co+co*2/3)+1)*255)
        
        cnt2 = 0
        
    if cnt3 >250:
        p[:,1:] = p[:,:-1]
        if cnt3 >350:
            p[:,:] = 0
            cnt3 = 0
    
    return np.concatenate((p, p[:, ::-1]), axis=1)

def vizualize_idk():
    y = np.copy(y)
    gain.update(y)
    y /= gain.value
        

    

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
    y = audio_samples / 2.0**15
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
        N = len(y_data)
        N_zeros = 2**int(np.ceil(np.log2(N))) - N
        # Pad with zeros until the next power of two
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
        # Construct a Mel filterbank from the FFT data
        mel = np.atleast_2d(YS).T * dsp.mel_y.T
        # Scale data to values more suitable for visualization
        # mel = np.sum(mel, axis=0)
        mel = np.sum(mel, axis=0)
        mel = mel**2.0
        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        
         
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
elif sys.argv[1] == "energy":
        visualization_type = visualize_energy
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
elif sys.argv[1] == "tic":
        visualization_type = visualize_tic
elif sys.argv[1] == "tic2":
        visualization_type = visualize_tic2
elif sys.argv[1] == "tic3":
        visualization_type = visualize_tic3
elif sys.argv[1] == "wave":
        visualization_type = visualize_wave
elif sys.argv[1] == "sweetscroll":
        visualization_type = visualize_sweetscroll
elif sys.argv[1] == "energy_classic":
        visualization_type = visualize_energy_classic
elif sys.argv[1] == "energy_gaps":
        visualization_type = visualize_energy_gaps
elif sys.argv[1] == "energy_base":
        visualization_type = visualize_energy_base
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
elif sys.argv[1] == "tetris":
        visualization_type = visualize_tetris
elif sys.argv[1] == "radial_wave":
        visualization_type = visualize_radial_wave
elif sys.argv[1] == "radial_wave2":
        visualization_type = visualize_radial_wave2
elif sys.argv[1] == "radial_wave3":
        visualization_type = visualize_radial_wave3
elif sys.argv[1] == "umbrella_wave":
        visualization_type = visualize_umbrella_wave
elif sys.argv[1] == "radial_wave4":
        visualization_type = visualize_radial_wave4
elif sys.argv[1] == "trans_wave":
        visualization_type = visualize_trans_wave
elif sys.argv[1] == "breathe":
        visualization_type = visualize_breathe
elif sys.argv[1] == "breathe2":
        visualization_type = visualize_breathe2
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
        
else: #sys.argv[1] == "tetris":
        visualization_type = visualize_slow_wave
visualization_effect = visualization_type
"""Visualization effect to display on the LED strip"""


if __name__ == '__main__':

    # Initialize LEDs
    led.update()
    # Start listening to live audio stream
    microphone.start_stream(microphone_update)
  

