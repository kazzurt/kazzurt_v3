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
pix      = config.N_PIXELS // 2 - 1

p        = np.tile(1.0, (3, config.N_PIXELS))

cnt2 = np.array([0,0,0])
#This is the umbrella pixel collection
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
rtim = 0
rtim3 = 0
arx = np.linspace(0,len(p[0,:])//50-1,len(p[0,:])//50).astype(int)
ary = np.linspace(0,49,50).astype(int)
ar_wave0 = np.ones((len(p[0,:])//50,50))
ar_wave1 = np.ones((len(p[0,:])//50,50))
ar_wave2 = np.ones((len(p[0,:])//50,50))
phw = 10
rtim3 =0
coo = np.array([1,1,1]).astype(float)
xdiv = 14
ydiv = 49
abc = 0
dcr = 0


arby = np.zeros((config.N_PIXELS//50,50))
rr = rn.randint(2,13)
ry = rn.randint(2,47)
#xxs = np.array([rr, rr+1, rr-1]).astype(int)
#yys = np.array([ry, ry, ry]).astype(int)
xxs = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
yys = np.zeros((1,config.N_PIXELS//50)).astype(int)
yys2 = np.zeros((1,config.N_PIXELS//50)).astype(int)+49
yys3 = np.zeros((1,config.N_PIXELS//50)).astype(int)+24

class umb:
    
    def umbrella(y):
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
        
                Por  = np.reshape(p[0],(len(p[0,:])//50,50))
                Por1 = np.reshape(p[1],(len(p[0,:])//50,50))
                Por2 = np.reshape(p[2],(len(p[0,:])//50,50))
        
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
    
    def umb_wave(y):
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
