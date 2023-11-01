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
import cmdfun
p      = np.tile(1.0, (3, config.N_PIXELS))
gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix    = config.N_PIXELS // 2 - 1
ar     = np.array([21, 23, 70, 71, 72, 73,  74,  75,  76,  77,  78, 79])
colm   = np.linspace(0,config.N_PIXELS-1, 50).astype(int)
colm2  = np.linspace(0,config.N_PIXELS // 2-1, 2).astype(int)

qq     = 0
qq     = 249
qq2    = 0
thresh = 0.3

cr     = 0
hg     = 0
hg2    = 0
hg3    = 0
fwdd   = 1
gau    = .6
crg    = [2,1,0]

co     = 0
colo2 = pallette.pal(0)
color = pallette.pal(0)
colm2 = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(10,25)).astype(int)
cl = 1
nu = 50 #starting number of tics for tic 2
fwd = 0 #starting direction for tic 2
tic_yn = 0
class ticsfull:
    def tic1(y):
        global p, qq, a, ar, colm, qq2, hg, thresh
       
        y = y**2
        gain.update(y)
        qq +=1
        y /= gain.value
        arq = int(.5*(np.sin(qq/50)+1)*255)
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        
        
        if ty>thresh and qq>5 or qq>25:
            hg+=1
            if qq<=7:
                thresh*=1.1
                print("Threshold Change, tic")
                print(thresh)
            elif qq>15:
                thresh*=.9
                print("Threshold Change, tic")
                print(thresh)
            p[:,:] = 0
            qq2+=1
            mu = (25*np.sin(qq2*np.pi/100)+50)
            colm = np.linspace(0,config.N_PIXELS - 1, np.floor(mu).astype(int)).astype(int)
            hg =0 
            qq = 0
            
        p[0,colm] = int(.5*(np.sin(qq2/25+0)+1)*255)
        p[1,colm] = int(.5*(np.sin(qq2/25+25/3)+1)*255)
        p[2,colm] = int(.5*(np.sin(qq2/25+25*2/3)+1)*255)
        
        p[0, :] = gaussian_filter1d(p[0, :], sigma=.4)
        p[1, :] = gaussian_filter1d(p[1, :], sigma=.4)
        p[2, :] = gaussian_filter1d(p[2, :], sigma=.4)
        return p
    
    def tic2(y):
        global p, qq, a, ar, colm, qq2, thresh, color, colm2, cl, nu, fwd, hg, tic_yn
        
        y = y**2
        gain.update(y)
        qq +=1
        y /= gain.value
        arq = int(.5*(np.sin(qq/50)+1)*255)
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        qq2+=1
        

        
        if ty>thresh and qq2>40:
            color = pallette.pal(0)
            qq2 = 0
           
        if ty>thresh or qq>40:
            if qq>10:
                #Automatic moving thresholds
                tic_yn = 1
                hg+=1
                if qq<15 and hg>3:
                    thresh*=1.1
                    hg = 0
                    print('Threshold Up, tic2 (auto): %5.3f.' % thresh)
                elif qq>25:
                    thresh*=.9
                    hg = 0
                    print('Threshold Up, tic2 (auto): %5.3f.' % thresh)
                    print(thresh)
                
        if tic_yn == 1:
            tic_yn = 0
            p[:,:] = 0
            cl += 1
            for x in np.linspace(0,len(color[0,:])-1,len(color[0,:])).astype(int):
                colm = np.linspace(0,config.N_PIXELS - 1, nu+x).astype(int)
                p[0,colm] = color[0,x]
                p[1,colm] = color[1,x]
                p[2,colm] = color[2,x]
            
            if nu>=75:
                fwd = 1
            elif nu <=25:
                fwd = 0
                
            if fwd == 0:
                nu+=5
            elif fwd == 1:
                nu-=5                
            qq = 0
        
        sig = .25*np.sin(cl*np.pi/2)+.3
        p[0, :] = gaussian_filter1d(p[0, :], sigma=sig)
        p[1, :] = gaussian_filter1d(p[1, :], sigma=sig)
        p[2, :] = gaussian_filter1d(p[2, :], sigma=sig)
        
        return p
    
    def ticpal(y, lpad):
        global p, qq, a, ar, colm, qq2, hg, thresh, colo2, co
        coms = cmdfun.pygrun() 
        y = y**2
        gain.update(y)
        qq +=1
        y /= gain.value
        arq = int(.5*(np.sin(qq/50)+1)*255)
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        qq2+=1
        
        #Manual threshold up/down
        if coms[273] == 1: #Arrow Up
            coms[273] = 0
            thresh*=1.1
            print('Threshold Up, ticpal: %5.3f.' % thresh)
        if coms[274] == 1: #Arrow Down
            coms[274] = 0
            thresh*=.9
            print('Threshold Down, ticpal: %5.3f.' % thresh)
        
        if ty>thresh or qq>30  or coms[275] == 1 or lpad[1,61] == 1: #Right arrow
            if coms[275] == 1 or lpad[1,61] == 1:
                lpad[1,61] = 0
                p[:,:] = 0
                colm = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(25,125)).astype(int)
                qq = 0
                colo2 = pallette.pal(0)
                coms[275] = 0
            elif qq>10 and coms[275] == 0:
                hg+=1
                if qq<15 and hg>3:
                    hg = 0
                    thresh*=1.1
                    print('Threshold Up, ticpal (auto): %5.3f.' % thresh)
                elif qq>25 and hg>3:
                    hg = 0
                    thresh*=.89
                    print('Threshold Down, ticpal (auto): %5.3f.' % thresh)
                p[:,:] = 0
                colm = np.linspace(0,config.N_PIXELS // 2-1, rn.randint(25,125)).astype(int)
                qq = 0
                colo2 = pallette.pal(0)
                
            
            
        CL = int(len(colo2[:,0]))
        com = len(colm)
        
        for x in np.linspace(0,CL-1, CL).astype(int): #Going through list of color pallettes
            p[0,colm[(x-1):com//x]] = colo2[x,0]
            p[1,colm[(x-1):com//x]] = colo2[x,1]
            p[2,colm[(x-1):com//x]] = colo2[x,2]
            
        if qq>10:
            p[0, :] = gaussian_filter1d(p[0, :], sigma=.5)
            p[1, :] = gaussian_filter1d(p[1, :], sigma=.5)
            p[2, :] = gaussian_filter1d(p[2, :], sigma=.5)
        return np.concatenate((p, p[:, ::-1]), axis=1)