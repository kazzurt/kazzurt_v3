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
import kzbutfun

gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix    = config.N_PIXELS // 2 - 1
rowl1  = np.linspace(0,49,50).astype(int)
rowl2  = np.linspace(0,24,25).astype(int)
rowl3  = np.linspace(0,24,25).astype(int)
rowl4  = np.linspace(0,24,25).astype(int)

sl     = 0
dire   = 0
rowl_thresh = 20
rowl_time = 25
sl_rowl = 0

sl = 0
dire = 0
phl = 25
cnt = 0
phl = 50
p = np.tile(1.0, (3, config.N_PIXELS // 2))
pv = np.tile(1.0, (3, config.N_PIXELS // 2))
xx = config.N_PIXELS//100 #strands of 50, 1000 pixels is 20 strands
yy = 50 

ara1 = np.ones((xx,yy))
ara2 = ara1
ara3 = ara1
ary = np.linspace(0, 49, 50).astype(int)
arx = np.linspace(0, xx-1, xx).astype(int)
right = 0
left = 0
vr3 = 0
class rowl:
    
    def vrowl(y):
        global p, gain, rowl1, cnt, sl_rowl, dire, rowl_thresh, rowl_time, pix, right, left
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        cnt +=1
        sl_rowl+=1
        m2 = np.mean(y[28:])
#         buts = kzbutfun.kzlaunchpad()
#         if buts is not None:
#             if buts[0] == 74 or buts[0] == 75:
#                 if buts[0] == 75 and np.min(rowl1)-25 > 0:
#                     dire = 1
#                 elif buts[0] == 74 and np.max(rowl1)+25 < config.N_PIXELS//2-26:
#                     dire = 0
#                 m2  = rowl_thresh+1
#                 cnt = 6
#             elif buts[0] == 71:
#                 right = 1
#                 left = 0
#             elif buts[0] == 70:
#                 left = 0
#                 right = 0
            
        p[0,rowl1] = int(.5*(np.sin(sl_rowl/rowl_time+0)+1)*255)
        p[1,rowl1] = int(.5*(np.sin(sl_rowl/rowl_time+rowl_time/3)+1)*255)
        p[2,rowl1] = int(.5*(np.sin(sl_rowl/rowl_time+2*rowl_time/3)+1)*255)
        
        if right == 1 or left == 1:
            cnt = 6
            m2 = rowl_thresh+1
        
        if np.min(rowl1) < 0:
            dire = 1
            if left == 1:
                right = 1
                left = 0
        elif np.max(rowl1) >= pix-25:
            dire = 0
            if right == 1:
                left = 1
                right = 0
        
        if m2 >rowl_thresh and cnt >5 and dire == 1:
            rowl1+=50
            cnt = 0
            #p[:,:]= 0
        if m2 >rowl_thresh and cnt >5 and dire == 0:
            p[:,rowl1] = 0
            rowl1-=50
            cnt = 0

        if cnt >30:
            rowl_thresh *=.75
        return np.concatenate((p, p[:, ::-1]), axis=1)


    def vrowl2(y):
        global pv, gain, rowl2, cnt, sl, dire, phl, ara1, ara2, ara3, arx, ary
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        cnt +=1
        sl+=1
        m2 = np.mean(y[28:])
        
        if np.mean(y[28:]) < 1:
            pv[:, 1:] = pv[:, :-1]
            return np.concatenate((pv, pv[:, ::-1]), axis=1)
        pv[0,rowl2] = int(.5*(np.sin(sl/phl +  0     )+1)*255)
        pv[1,rowl2] = int(.5*(np.sin(sl/phl +  phl/3 )+1)*255)
        pv[2,rowl2] = int(.5*(np.sin(sl/phl +2*phl/3) +1)*255)
        if np.min(rowl2) <= 0:
            dire = 1
        elif np.max(rowl2) >= config.N_PIXELS//2-1:
            dire = 0
        if m2 >1 and cnt >15 and dire == 1:
            rowl2+=25
            cnt = 0
            #if np.mean(p[:,:])>150:
                #p[:,:]= 0 
        if m2 >1 and cnt >15 and dire == 0:
            rowl2-=25
            cnt = 0
            #if np.mean(p[:,:])>150:
                #p[:,:]= 0    
        if dire == 0: #m2 >5 and 
            pv[0,rowl2-25] = int(.5*(np.sin(sl/phl+2*phl/3)+1)*255)
            pv[1,rowl2-25] = int(.5*(np.sin(sl/phl+phl/3)+1)*255)
            pv[2,rowl2-25] = int(.5*(np.sin(sl/phl + 0)+1)*255)
            
        if np.max(rowl2+25)<500: #m2 >5 and

            pv[0,rowl2+25] = int(.5*(np.sin(sl/phl+2*phl/3)+1)*255)
            pv[1,rowl2+25] = int(.5*(np.sin(sl/phl+phl/3)+1)*255)
            pv[2,rowl2+25] = int(.5*(np.sin(sl/phl + 0)+1)*255)
        
        for x in arx:
            ara1[x,ary] = .5*np.sin(sl/20 + x + ary)+.75

        pv *= ara1.flatten()    
        pv[0,:] = gaussian_filter1d(pv[0,:], sigma=.5)
        pv[1,:] = gaussian_filter1d(pv[1,:], sigma=.5)
        pv[2,:] = gaussian_filter1d(pv[2,:], sigma=.5)
        #pv[:,len(p[0,:])//2::] = np.fliplr(p[:,:len(pv[0,:])//2] )
        return np.concatenate((pv, pv[:, ::-1]), axis=1)


    def vrowl3(y): 
        global p, gain, rowl3, cnt, sl, dire, phl, pix, vr3
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        cnt +=1
        m2 = np.mean(y[28:])
        if cnt>15:
            m2 = 6 #force next
        vr3 +=1
           
        if np.mean(y[28:]) < 1 or vr3>8 and dire ==0:
            vr3 = 0
            p[:, 1:] = p[:, :-1]
            p2 = 0*p
            p2[0,:] = p[1,:]
            p2[1,:] = p[2,:]
            p2[2,:] = p[0,:]
            #return np.concatenate((p, p[:, ::-1]), axis=1)
        if np.mean(y[28:]) < 1 or vr3>8 and dire ==1:
            vr3 = 0
            p[:, :-1] = p[:, 1:] 
            p2 = 0*p
            p2[0,:] = p[1,:]
            p2[1,:] = p[2,:]
            p2[2,:] = p[0,:]
            #return np.concatenate((p, np.fliplr(p)), axis=1)
        
        p[0,rowl3] = int(.5*(np.sin(sl/phl +  0     )+1)*255)
        p[1,rowl3] = int(.5*(np.sin(sl/phl +  phl/3 )+1)*255)
        p[2,rowl3] = int(.5*(np.sin(sl/phl +2*phl/3) +1)*255)
        
        if np.min(rowl3)-25 <= 0:
            dire = 1
        elif np.max(rowl3)+25 >= config.N_PIXELS//2-26:
            dire = 0
        if m2 >5 and cnt >10 and dire == 1:
            rowl3+=25
            cnt = 0
            p[:,rowl3-25]= 0
            p[:, :-1] = p[:, 1:]
        if m2 >5 and cnt >10 and dire == 0:
            rowl3-=25
            cnt = 0
            p[:,rowl3+25]= 0
            p[:, :-1] = p[:, 1:]
        if dire == 0:
            sl+=1
            p[0,rowl3-25] = int(.5*(np.sin(sl/phl+2*phl/3)+1)*255)
            p[1,rowl3-25] = int(.5*(np.sin(sl/phl+phl/3)+1)*255)
            p[2,rowl3-25] = int(.5*(np.sin(sl/phl + 0)+1)*255)
            
        if dire == 1:
            sl+=1
            p[0,rowl3+25] = int(.5*(np.sin(sl/phl+2*phl/3)+1)*255)
            p[1,rowl3+25] = int(.5*(np.sin(sl/phl+phl/3)+1)*255)
            p[2,rowl3+25] = int(.5*(np.sin(sl/phl + 0)+1)*255)
    
       

        return p #np.concatenate((p, np.fliplr(p)), axis=1)
