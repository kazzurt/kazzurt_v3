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
import viz_mf

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix      = config.N_PIXELS // 2 - 1
arx      = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
ary      = np.linspace(0,49,50).astype(int)
ar_wave0 = np.ones((config.N_PIXELS//50,50))
ar_wave1 = np.ones((config.N_PIXELS//50,50))
ar_wave2 = np.ones((config.N_PIXELS//50,50))
evs2     = np.linspace(0,config.N_PIXELS//50-2,config.N_PIXELS//50//2).astype(int)
ods2     = np.linspace(1,config.N_PIXELS//50-1,config.N_PIXELS//50//2).astype(int)
phum     = np.array([0,25/3,2*25/3])
arby     = np.zeros((config.N_PIXELS//50,50))
cnt1     = 0
trig1    = 0
phw      = 10
phw2     = 5
coo      = np.array([1,1,1]).astype(float)
xdiv     = 14
ydiv     = 49
abc      = 0
dcr      = 0
coc      = 0
coc1     = 0
coc2     = 1

rtim     = 0
rtim2    = 0
rtim3    = 0

phw_gap  = 0
exc      = 8

f        = 0
co       = 1
co2      = pallette.pal(0)
p        = np.tile(1.0, (3, config.N_PIXELS))

class radial_wave:
    def rwave(y):
        global p, rtim, pix, arx, ary, phw, rtim3, coo, xdiv, ydiv
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
    
    def rwave2(y):
        global p, rtim, pix, arx, ary, phw, evs2, ods2, phw2, rtim2, phw_gap, exc
       
        rtim   +=1
        rtim2  +=1
        timing = 10
        ydi = 10
        for y in ary:
            #these sin waves need to be renormalized to fit the net better
            ar_wave0[evs2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw        )+1))*255
            ar_wave0[ods2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw + np.pi)+1))*255
            
            ar_wave1[evs2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw2 + phw2/3+phw_gap)+1))*255
            ar_wave1[ods2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw2 + phw2/3 + np.pi)+1))*255
            
            ar_wave2[evs2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw + 2*phw/3)+1))*255
            ar_wave2[ods2,y] = ((.5*np.sin(y*np.pi/ydi+rtim/timing)+.5)**exc)*(.5*(np.sin(rtim/phw2 + 2*phw2/3 + np.pi+phw_gap)+1))*255
            
        p[0,:] = viz_mf.flatMat(ar_wave0)
        p[1,:] = viz_mf.flatMat(ar_wave1)
        p[2,:] = viz_mf.flatMat(ar_wave2)

        if rtim2 > 50 and np.mean(p[:,:])<10:
            exc-=1
            rtim2 = 0
        return p

    def rwave3(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, phum
        rtim +=1
        rtim3+=1
        
        tt = np.transpose(((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1))
        ar_wave0[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ ary/7)+1)*255  #49 and 14 are circular +arx/4 + np.sin(arx*np.pi/(xdiv))/2
        
        coo[0] = (.5*np.sin(rtim/phw)+.5)**.5
        coo[1] = (.5*np.sin(rtim/phw/3+phw/3)+.5)**.5
        coo[2] = (.5*np.sin(rtim/phw/3+2*phw/3)+.5)**.5
        
        p[0,:] = coo[0]*ar_wave0.flatten()
        p[1,:] = coo[1]*ar_wave0.flatten()
        p[2,:] = coo[2]*ar_wave0.flatten()
        
        ppm = np.mean(p[:,:])
        xdiv = (.5*np.sin(rtim/50)+.5)*20 + 14-14/2
        ydiv = (.5*np.sin(rtim/50)+.5)*49 + 49-49/2 
        p = gaussian_filter1d(p, sigma=2)
        return p
    
    def rwave4(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv
        
        rtim +=1
        rtim3+=1

        tt = np.transpose(((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1))
        
        ar_wave0[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ ary/7)+1)*255
        ar_wave1[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ 2*rtim/400*ary**2/7/3)+1)*255
        ar_wave2[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ 2*rtim/400*ary**2/7)+1)*255
        p[0,:] = ar_wave0.flatten()
        p[1,:] = ar_wave1.flatten()
        p[2,:] = ar_wave2.flatten()

        ppm = np.mean(p[:,:])
        xdiv = (.5*np.sin(rtim/50)+.5)*20 + 14-14/2
        ydiv = (.5*np.sin(rtim/50)+.5)*49 + 49-49/2
            
        p = gaussian_filter1d(p, sigma=2)
        return p
    
    def rwave5(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv
        
        rtim +=np.sin(rtim3/100)+1
        rtim3+=1
        
        tt = np.transpose(((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1))
        
        ar_wave0[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw + ary/12)+1)*255
        ar_wave1[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw + 2*rtim/200*ary**2/13/3)+1)*255
        ar_wave2[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw + 4*rtim/200*ary**2/13/3)+1)*255
        
        p[0,:] = ar_wave0.flatten()
        p[1,:] = ar_wave1.flatten()
        p[2,:] = ar_wave2.flatten()

        ppm = np.mean(p[:,:])
        xdiv = (.5*np.sin(rtim/50)+.5)*100 + 2-14/2
        ydiv = (.5*np.sin(rtim/50)+.5)*49 + 7-49/2    
        p = gaussian_filter1d(p, sigma=2)
        return p
    
    def rwave6(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv
        
        rtim +=2
        rtim3+=2
        tt = np.transpose(((np.sin(arx*np.pi/(xdiv))/2 )**2)*(.5*np.sin(rtim/phw+ arx/7)+1))
        
        ar_wave0[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ ary/12)+1)*255
        ar_wave1[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ 2*rtim/200*ary**2/13/3)+1)*255
        ar_wave2[:,:] = ((np.sin(ary*np.pi/(ydiv))/2 ))*(.5*np.sin(rtim/phw+ 2*rtim/200*ary**2/13)+1)*255
        p[0,:] = ar_wave0.flatten()
        p[1,:] = ar_wave1.flatten()
        p[2,:] = ar_wave2.flatten()

        ppm = np.mean(p[:,:])
        xdiv = (.5*np.sin(rtim/50)+.5)*100 + 2-14/2
        ydiv = (.5*np.sin(rtim/50)+.5)*49 + 7-49/2    
        p = gaussian_filter1d(p, sigma=2)
        return p
    
    def rpal(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, evs2, ods2, phw2, rtim2, phw_gap, exc, co2, coc, coc1, coc2, f
        rtim +=1
        rtim2 +=1
        fac = 5
        phw = 5
        if rtim2<75:
            for y in ary:
            
                ar_wave0[evs2,y] = co2[coc1,0]*(.5*(np.sin(rtim/phw + y/fac       )+1))
                ar_wave0[ods2,y] = co2[coc2,0]*(.5*(np.sin(rtim/phw + y/fac + np.pi)+1))
            
                ar_wave1[evs2,y] = co2[coc1,1]*(.5*(np.sin(rtim/phw2 + y/fac + phw2/3+phw_gap)+1))
                ar_wave1[ods2,y] = co2[coc2,1]*(.5*(np.sin(rtim/phw2 + y/fac + phw2/3 + np.pi)+1))
            
                ar_wave2[evs2,y] = co2[coc1,2]*(.5*(np.sin(rtim/phw  + y/fac + 2*phw/3)+1))
                ar_wave2[ods2,y] = co2[coc2,2]*(.5*(np.sin(rtim/phw2 + y/fac + 2*phw2/3 + np.pi+phw_gap)+1))

            
            p[0,:] = ar_wave0.flatten()
            p[1,:] = ar_wave1.flatten()
            p[2,:] = ar_wave2.flatten()
        if rtim2 > 100:
            exc-=1
            num = np.random.randint(len(p[0,:]),size=100)
            if f == 0:
                p*=.95
            elif f == 1:
                num = np.random.randint(len(p[0,:]),size=100)
                p[:,num] = 0
                
            if rtim2>150:
                rtim2 = 0
                coc+=1
                co2 = pallette.pal(coc)
                coc1 = rn.randint(0,len(co2[:,0]))
                coc2 = rn.randint(0,len(co2[:,0]))
                if f == 0:
                    f = 1
                elif f == 1:
                    f = 0
                if coc >16:
                    coc = 0
        return p
