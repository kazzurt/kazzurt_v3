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

p      = np.tile(1.0, (3, config.N_PIXELS // 2))
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
cnt = 0
class incs:
#     def __init__(self, incvar1, incvar2):
#         self.mainloop = 
    
    def inc(y):
        global p, qq, cr, hg, hg2, thresh_inc, pix, cnt
        
        cnt+=1
        fg = 1
        #print(cnt)
        upp = int(3*np.sin(cnt/15)+4)
        dow = int(7*np.cos(cnt/15)+8)
        if fg>thresh_inc:
            
            p[0,np.random.randint(len(p[0,:]),size=upp)] = rn.randint(150,255)
            p[1,np.random.randint(len(p[0,:]),size=upp)] = rn.randint(150,255)
            p[2,np.random.randint(len(p[0,:]),size=upp)] = rn.randint(150,255)
            cr+=1
            hg = 0
            hg2 +=1
        print(cnt)
        if cnt>50:
            noff = np.random.randint(len(p[0,:]//2),size=dow)
            p[0,noff] = 0
            p[1,noff] = 0
            p[2,noff] = 0
    
        p[0, :] = gaussian_filter1d(p[0, :], sigma=.4)
        p[1, :] = gaussian_filter1d(p[1, :], sigma=.4)
        p[2, :] = gaussian_filter1d(p[2, :], sigma=.4)
        return np.concatenate((p[:, ::-1], p), axis=1)

    def inc2(y):
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
    
    def incpal(y):
        global p, qq, cr, hg, hg2, thresh_inc, pix, fwddd, ghh, gh2, jk, colo, co
        #still need to add the color pallete stuff
        
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