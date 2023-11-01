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
p        = np.tile(1.0, (3, config.N_PIXELS//2))
p_filt2  = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.1, alpha_rise=0.99)
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
kz = 0

arby = np.zeros((config.N_PIXELS//50,50))
rr = rn.randint(2,13)
ry = rn.randint(2,47)
#xxs = np.array([rr, rr+1, rr-1]).astype(int)
#yys = np.array([ry, ry, ry]).astype(int)
xxs = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
yys = np.zeros((1,config.N_PIXELS//50)).astype(int)
yys2 = np.zeros((1,config.N_PIXELS//50)).astype(int)+49
yys3 = np.zeros((1,config.N_PIXELS//50)).astype(int)+24

it = 0
trig1 = 0
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
pix = int((config.N_PIXELS / 2) - 1)
odds = np.linspace(1,pix,pix//2).astype(int)
evens = np.linspace(0,pix-1,pix//2).astype(int)

cnt1 = 0
phum = np.array([0,25/3,2*25/3])
trig1 = 0
cnt3 = 0
clr = pallette.pal(0)
rtim4 = 0
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
ewb = 25 #Speeed of our bump
ph2 = 20
kz2 = 0
x = 0
qq = 0
qq2 = 0
hg = 0
cnt2 = 0
sl = 0
cnt4 = 0
per = np.array([0,0,0])

#Tetris
cnt1t = 0
cnt3t = 0
tethresh = 5

class spunky:
    
    def spunk(y):
        global p, cnt2, pix, cnt3, sl
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        
        m1 = np.mean(y[:4])
        m2 = np.mean(y[5:8])
        m3 = np.mean(y[9:12])

        cnt2 = cnt2+1
        cnt3+=1
        sl+=1

        if m1 > 10 and cnt2>10 or cnt2>50:
            ppp = np.random.randint(len(p[0,:]),size=15)
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

    def tetris(y):
        global p, rty, cnt1t, odds, phum, evens, pix, cnt3t, clr, tethresh
        y2 = y**2
        gain.update(y2)
        y2 /= gain.value
        y2 *= 255.0
        m2 = np.mean(y2[28:])
        
        cnt1t +=1
        cnt3t+=1
        div = len(p[0,:]) // len(clr[:,0]) // 2
        
        
        
        
        if cnt3t>5:
            p[:,evens] = 0
            for i in np.linspace(0,len(clr[:,0])-1,len(clr[:,0])).astype(int):
                p[0,odds[(i)*div:(i+1)*div]] = clr[i,0]
                p[1,odds[(i)*div:(i+1)*div]] = clr[i,1]
                p[2,odds[(i)*div:(i+1)*div]] = clr[i,2]
            evens = np.linspace(0,pix,rn.randint(17,178)).astype(int)
            cnt3t = 0
        if cnt1t>20 and m2<tethresh:# and cnt3>5:
            p[:,odds] = 0
            odds = np.linspace(0,pix,rn.randint(17,178)).astype(int)
            for i in np.linspace(0,len(clr[:,0])-1,len(clr[:,0])).astype(int):
                p[0,evens[(i)*div:(i+1)*div]] = clr[i,0]
                p[1,evens[(i)*div:(i+1)*div]] = clr[i,1]
                p[2,evens[(i)*div:(i+1)*div]] = clr[i,2]
            cnt3t = 0
        if cnt3t>25:
            tethresh *=.95
            print("Tetris Thresh Down")
            
        if cnt1t>=20 and m2<tethresh:
            p[:, 1:] = p[:, :-1]
            if cnt1t>35:
                cnt1t = 0
                clr = pallette.pal(0)
                evens = np.linspace(0,pix,rn.randint(17,178)).astype(int)
            
        return np.concatenate((p, p[:, ::-1]), axis=1) #mirrored color symmetry
        
 
    def slow_wave(y):
        global p, cnt1, cnt2, phum, kz, it, trig1
        kz+=1
        cnt1+=1
        cnt2+=1
        num = int(.5*(np.sin(cnt1/50)+1)*374)
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
     
    def spectrum(y):
        global p, gain, qq, cr, hg, hg2, sthresh, qq2, colm, qew1, qew2, qew3, qe1, qe2, qe3, per
        y = y**2
        gain.update(y)
        qq +=1
        y /= gain.value
        arq = int(.5*(np.sin(qq/50)+1)*255)
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        qq2+=1
        per0 = np.mean(y[0:8])/np.max(y[0:8])
        per1 = np.mean(y[8:16])/np.max(y[8:16])
        per2 = np.mean(y[16:24])/np.max(y[16:24])
        
        if ty>sthresh:
            hg+=1
            if hg>3:
                per = (255*np.array([per0,per1,per2]))+50
                hg = 0
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
               
                qq = 0
        for i in np.array([0,1,2]).astype(int):    
            p[i,qew1] = per[i]
            p[i,qew2] = per[i]
            p[i,qew3] = per[i]
        p[0, :] = gaussian_filter1d(p[0, :], sigma=.35)
        p[1, :] = gaussian_filter1d(p[1, :], sigma=.35)
        p[2, :] = gaussian_filter1d(p[2, :], sigma=.35)
        return np.concatenate((p[:, ::-1], p), axis=1)

    def bump(y):
        global p, gain, kz, S, c5, ar, yup, x, ar2, du, kz2, ph2, ewb, beat
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
    
    def spunk2(y):
        global p, p_filt2, cnt2, pix, cnt3, sl, adr, cnt4
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
        p[:,:nn4 ] = gaussian_filter1d(p[:,:nn4 ], sigma=2)
        
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
                #p[:,:] = 0
                cnt3 = 0
        
        #p_filt2.update(p)
        return np.concatenate((p, p[:, ::-1]), axis=1)   
