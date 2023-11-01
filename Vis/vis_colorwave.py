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
import cmdfun

import kzbutfun

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix      = config.N_PIXELS // 2 - 1
p        = np.tile(1.0, (3, config.N_PIXELS))
coo      = np.array([1,1,1]).astype(float)


rtim        = 0
rtim3       = 0
cnt3        = 0
cy          = 0
ard         = 1
cyc         = 0
bthe        = 0
thresh_bthe = 0.5
timeCount   = 1
countUp     = True

arx         = np.linspace(0,39,40).astype(int)
ary         = np.linspace(0,24,25).astype(int)
red_ar      = np.zeros((40,25))
gre_ar      = np.zeros((40,25))
blu_ar      = np.zeros((40,25))
red_ar2     = np.zeros((40,25))
gre_ar2     = np.zeros((40,25))
blu_ar2     = np.zeros((40,25))
inten1      = np.zeros((40,25))
inten2      = np.zeros((40,25))

bdir        = 1
nuu         = 50 #defines speed for kuwave2 (higher is slower)
mat_map     = 1
sparkle     = 0
rtim4       = 0
rtim5       = 0
y_off       = 14
x_off       = 2
sparkle2    = 0
sparkle3    = 0

right       = 1
left        = 0
rig         = 0
lig         = 0
numx        = 1
numy        = 0
phas        = np.pi
sec         = 4
ydi         = 10
xdi         = 5
thresh      = .4
swit        = 1
#Colorwave1
rtim11      = 0
rtim31      = 0
coo11       = np.array([1,1,1]).astype(float)
cy11        = 0
y_off11     = 14
x_off11     = 2

#Colorwave6
rtim36 = 0

class colorwave:
    
    def colorwave1(y):
        global p, rtim11, pix, arx, ary, rtim31, coo11, cy11, bdir, red_ar, gre_ar, blu_ar, y_off11, x_off11
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
        
        rtim11+=1
        rtim31+=1
        
        num = 5*np.sin(rtim11/4)+6
        
        #If our x and y division sine wave is at its peak, let's increment cy, which determines color remapping below
        #rtim3 prevents going into this function a bunch of times while num sits close to its maximum
        #bdir controls whether we're stepping cy upwards or downwards
        if num > 10 and rtim31 > 15:
            rtim31 = 0
            if bdir == 1: 
                cy11+=1
            elif bdir == -1:
                cy11-=1
                if cy11<=0:
                    bdir = 1
                    #sparkle = 0
                    rtim4 = 0
        
        #copied from breathe2
        #This is the best wave function, dependent on loop number (rtim), x direction (i in arx) and y direction (ary)
        #Would be nice if both directions could be a matrix operation, but idk how to do that. Choosing smaller direction for the for loop
        xf = x_off11 + num
        yf = y_off11 - num
        
        for i in arx:
            red_ar[i,ary] =  (.5*np.sin(rtim11/4 + ary/yf + i/xf            )+.5)*255
            gre_ar[i,ary] =  (.5*np.sin(rtim11/4 + ary/yf + i/xf + 2*np.pi/3)+.5)*255
            blu_ar[i,ary] =  (.5*np.sin(rtim11/4 + ary/yf + i/xf + 4*np.pi/3)+.5)*255
        if cy11>1:
            red_ar = np.fliplr(red_ar)
            if cy11>2:
                gre_ar = np.fliplr(gre_ar)
                if cy11>3:
                    blu_ar = np.fliplr(blu_ar)
                    if cy11>4:
                        bdir=-1 #this will make us start stepping cy backwards
                        
        p[0,:] = coo11[0]*viz_mf.flatMatHardMode(red_ar)
        p[1,:] = coo11[1]*viz_mf.flatMatHardMode(gre_ar)
        p[2,:] = coo11[2]*viz_mf.flatMatHardMode(blu_ar)
        
        return p
    
    def colorwave2(y,lpad):
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               right, left, rig, lig, numx, numy
        
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim3+=1
        rtim5+=1
        bthe+=1

        xf = x_off
        yf = y_off
        
            
            
        if cyc==0:
            for i in arx:
                red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)*255
                gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)*255
                blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)*255
        if cy>1:
            red_ar = np.fliplr(red_ar)
            if cy>2:
                gre_ar = np.fliplr(gre_ar)
                if cy>3:
                    blu_ar = np.fliplr(blu_ar)
                    if cy>4:
                        bdir=-1 #this will make us start stepping cy backwards
                        #sparkle = 1
                        
        p[0,:] = coo[0]*viz_mf.flatMatHardMode(red_ar)
        p[1,:] = coo[1]*viz_mf.flatMatHardMode(gre_ar)
        p[2,:] = coo[2]*viz_mf.flatMatHardMode(blu_ar)
   
        if lpad[1,56] == 1:    #Right
            coms[275] = 1
            lpad[1,56] = 0
            
        elif lpad[1,54] == 1: #Left
            coms[276] = 1
            lpad[1,54] = 0
            
        elif lpad[1,51] == 1: #down
            coms[274] = 1
            lpad[1,52] = 0
            
        elif lpad[1,59] == 1:    #Up
            coms[273] = 1
            lpad[1,60] = 0
            
        elif lpad[1,55] == 1:    #middle (stop)
            right = 0
            left = 0
              
        if coms[275] == 1 or right ==1: #right arrow
            if coms[275] == 1 and right == 1:
                rig += 1
            lig = 0
            right = 1
            left = 0
            rtim -= (2+rig)
            coms[275] = 0
            
        if coms[276] == 1 or left == 1: #left arrow
            if coms[276] == 1 and left == 1:
                lig += 1
            rig = 0
            left = 1
            right = 0
            rtim += 2 + lig
            coms[276] = 0
            
        if coms[274] == 1: #down arrow
            y_off *= .5
            print(y_off)
            coms[274] = 0
            
        if coms[273] == 1: #up arrow
            y_off *= 2
            print(y_off)
            coms[273] = 0
        return p
    
    def colorwave3(y,lpad): #function 50
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               coms, right, left, rig, lig, numx, numy
        
        y      = y**2
        gain.update(y)
        y     /= gain.value
        y     *= 255.
        ty     = np.mean(y[len(y)//2::])
        rtim  += 1
        rtim3 += 1
        rtim5 += 1
        bthe  += 1
        num = 5*np.sin(rtim/4)+6
        xf     = x_off
        yf     = y_off
        
        for i in arx:
            if i % 2 == 0:
                red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)**2*255
                gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)**2*255
                blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)**2*255
            else:
                red_ar[i,ary] =  (.5*np.sin(-rtim/4 + ary/yf + i/xf            )+.5)**2*255
                gre_ar[i,ary] =  (.5*np.sin(-rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)**2*255
                blu_ar[i,ary] =  (.5*np.sin(-rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)**2*255
#         for i in range(0,len(red_ar[0,:])):
#             if i % 2 == 1:
#                 red_ar[i,:] = np.flip(red_ar[i,:])
#                 blu_ar[i,:] = np.flip(blu_ar[i,:])
#                 gre_ar[i,:] = np.flip(gre_ar[i,:])
        coms = cmdfun.pygrun()
        if coms[275] == 1 or right ==1 or lpad[1,56] == 1: #right arrow
            if coms[275] == 1 and right == 1:
                rig += 1
            lig = 0
            right = 1
            left = 0
            rtim -= (2+rig)
            coms[275] = 0
            
        if coms[276] == 1 or left == 1 or lpad[1,54] == 1: #left arrow
            if coms[276] == 1 and left == 1:
                lig += 1
            rig = 0
            left = 1
            right = 0
            rtim += 2 + lig
            coms[276] = 0
        
        if lpad[1,53] == 1:  #RESET
            lpad[1,53] = 0
            phas = np.pi
            y_off = 14
            x_off = 2
            sec  = 4
            ydi  = 10
            xdi  = 5
            
        if coms[274] == 1 or lpad[1,54] == 1: #down arrow
            y_off *= .5
            coms[274] = 0
            
        if coms[273] == 1 or lpad[1,59] == 1: #up arrow
            y_off *= 2
            coms[273] = 0        
        p[0,:] = coo[0]*viz_mf.flatMatHardMode(red_ar)
        p[1,:] = coo[1]*viz_mf.flatMatHardMode(gre_ar)
        p[2,:] = coo[2]*viz_mf.flatMatHardMode(blu_ar)

        return p
    
    def colorwave4(y,lpad): #function 51
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               coms, right, left, rig, lig, numx, numy, phas, red_ar2, blu_ar2, gre_ar2, sec
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim3+=1
        rtim5+=1
        bthe+=1

        xf = x_off
        yf = y_off
        
        
        for i in arx:
            red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)
            gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3+phas)+.5)
            blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3+2*phas)+.5)
        
        if lpad[1,47] == 1:
            sec *=2
            lpad[1,47] = 0
            print(sec)
        elif lpad[1,48] == 1:
            sec *=.5
            print(sec)
            lpad[1,47] = 0
        
        for i in arx:
            red_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 2*np.pi/3+phas)+.5)
            gre_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 4*np.pi/3+2*phas)+.5)
            blu_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf )+.5)  
                        
        p[0,:] = (viz_mf.flatMatHardMode(red_ar)+viz_mf.flatMatHardMode(red_ar2)/2) * 255
        p[1,:] = (viz_mf.flatMatHardMode(gre_ar)+viz_mf.flatMatHardMode(gre_ar2)/2) * 255
        p[2,:] = (viz_mf.flatMatHardMode(blu_ar)+viz_mf.flatMatHardMode(blu_ar2)/2) * 255
        coms = cmdfun.pygrun()
        
        if lpad[1,56] == 1:    #Right
            coms[275] = 1
            lpad[1,56] = 0
            
        elif lpad[1,54] == 1: #Left
            coms[276] = 1
            lpad[1,54] = 0
            
        elif lpad[1,51] == 1: #down
            coms[274] = 1
            lpad[1,52] = 0
            
        elif lpad[1,59] == 1:    #Up
            coms[273] = 1
            lpad[1,60] = 0
            
        elif lpad[1,55] == 1:    #middle (stop)
            right = 0
            left = 0
        
        if lpad[1,49] == 1:
            lpad[1,49] = 0
            phas +=.5
            print(phas)
        elif lpad[1,48] == 1:
            lpad[1,48] = 0
            phas-=.5
        
        if lpad[1,53] == 1:  #RESET
            lpad[1,53] = 0
            phas = np.pi
            y_off       = 14
            x_off       = 2
            sec  = 4
            ydi  = 10
            xdi  = 5
            
        if coms[275] == 1 or right ==1: #right arrow
            if coms[275] == 1 and right == 1:
                rig += 1
            lig = 0
            right = 1
            left = 0
            rtim -= (2+rig)
            coms[275] = 0
            
        if coms[276] == 1 or left == 1: #left arrow
            if coms[276] == 1 and left == 1:
                lig += 1
            rig = 0
            left = 1
            right = 0
            rtim += 2 + lig
            coms[276] = 0
            
        if coms[274] == 1: #down arrow
            y_off *= .5
            print(y_off)
            coms[274] = 0
            
        if coms[273] == 1: #up arrow
            y_off *= 2
            print(y_off)
            coms[273] = 0
        return p
    def colorwave5(y,lpad):  #function 52
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               coms, right, left, rig, lig, numx, numy, phas, red_ar2, blu_ar2, gre_ar2, sec, inten1, inten2, ydi, xdi
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim3+=1
        rtim5+=1
        bthe+=1

        xf = x_off
        yf = y_off
        
        if lpad[1,61] == 1:
            ydi *=.75
            print(ydi)
        if lpad[1,60] == 1:
            xdi *= .75
            print(xdi)
            
        for i in arx:
            inten1[i,ary]  = ((.5*np.sin(np.pi*ary/ydi - 25*np.pi)+.5)/2 + (.5*np.sin(np.pi*i/xdi - 10*np.pi)+.5)/2)*(.5*np.sin(rtim5*2*np.pi/25)+.5)
            inten2[i,ary]  = ((.5*np.sin(np.pi*ary/ydi)+.5)/2 + (.5*np.sin(np.pi*i/xdi)+.5)/2)*(.5*np.sin(rtim5*2*np.pi/25-np.pi)+.5)
            
#             red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)
#             gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3+phas)+.5)
#             blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3+2*phas)+.5)
            red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi            )+.5)
            gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi + 2*np.pi/3+phas)+.5)
            blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi + 4*np.pi/3+2*phas)+.5)        
        if lpad[1,47] == 1:
            sec *=2
            lpad[1,47] = 0
            print(sec)
        elif lpad[1,48] == 1:
            sec *=.5
            print(sec)
            lpad[1,47] = 0
        
        for i in arx:
#             red_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 2*np.pi/3+phas)+.5)
#             gre_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 4*np.pi/3+2*phas)+.5)
#             blu_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf )+.5)  
            red_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/ydi + i/xdi + 2*np.pi/3+phas)+.5)
            gre_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/ydi + i/xdi + 4*np.pi/3+2*phas)+.5)
            blu_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/ydi + i/xdi )+.5)
            
        p[0,:] = (viz_mf.flatMatHardMode(red_ar)+viz_mf.flatMatHardMode(red_ar2)/2) * 255
        p[1,:] = (viz_mf.flatMatHardMode(gre_ar)+viz_mf.flatMatHardMode(gre_ar2)/2) * 255
        p[2,:] = (viz_mf.flatMatHardMode(blu_ar)+viz_mf.flatMatHardMode(blu_ar2)/2) * 255
        p     *=  viz_mf.flatMatHardMode(inten1)/2 + viz_mf.flatMatHardMode(inten2)/2
        coms = cmdfun.pygrun()
        
        if lpad[1,56] == 1:    #Right
            coms[275] = 1
            lpad[1,56] = 0
            
        elif lpad[1,54] == 1: #Left
            coms[276] = 1
            lpad[1,54] = 0
            
        elif lpad[1,51] == 1: #down
            coms[274] = 1
            lpad[1,52] = 0
            
        elif lpad[1,59] == 1:    #Up
            coms[273] = 1
            lpad[1,60] = 0
            
        elif lpad[1,55] == 1:    #middle (stop)
            right = 0
            left = 0
            
        if lpad[1,53] == 1:
            lpad[1,53] = 0
            phas = np.pi
            sec  = 4
            ydi  = 10
            xdi  = 5
            y_off = 14
            x_off = 2
            
        if lpad[1,49] == 1:
            lpad[1,49] = 0
            phas +=.5
            
        elif lpad[1,48] == 1:
            lpad[1,48] = 0
            phas-=.5
        
        if coms[275] == 1 or right ==1: #right arrow
            if coms[275] == 1 and right == 1:
                rig += 1
            lig = 0
            right = 1
            left = 0
            rtim -= (2+rig)
            coms[275] = 0
            
        if coms[276] == 1 or left == 1: #left arrow
            if coms[276] == 1 and left == 1:
                lig += 1
            rig = 0
            left = 1
            right = 0
            rtim += 2 + lig
            coms[276] = 0
            
        if coms[274] == 1: #down arrow
            y_off *= .5
            print(y_off)
            coms[274] = 0
            
        if coms[273] == 1: #up arrow
            y_off *= 2
            print(y_off)
            coms[273] = 0
        return p
    def colorwave6(y):
        global p, rtim, pix, arx, ary, rtim36, coo, xdiv, ydiv, cy, oods, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, sparkle, rtim4, C, rtim5, y_off, x_off, \
               sparkle2, sparkle3, coms
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
        
        rtim+=1
        rtim36+=1
        rtim5+=1
        bthe+=1
        
        num = 5*np.sin(rtim/3)+2
        
        #If our x and y division sine wave is at its peak, let's increment cy, which determines color remapping below
        #rtim3 prevents going into this function a bunch of times while num sits close to its maximum
        #bdir controls whether we're stepping cy upwards or downwards
        if num > 10 and rtim36 > 15:
            rtim36 = 0
            if bdir == 1: 
                cy+=1
            elif bdir == -1:
                cy-=1
                if cy<=0:
                    bdir = 1
                    #sparkle = 0
                    rtim4 = 0
        
        #copied from breathe2
        #This is the best wave function, dependent on loop number (rtim), x direction (i in arx) and y direction (ary)
        #Would be nice if both directions could be a matrix operation, but idk how to do that. Choosing smaller direction for the for loop
        xf = x_off+num
        yf = y_off-num
        if cyc==0:
            for i in arx:
                red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)*255
                gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)*255
                blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)*255
        if cy>1:
            red_ar = np.fliplr(red_ar)
            if cy>2:
                gre_ar = np.fliplr(gre_ar)
                if cy>3:
                    blu_ar = np.fliplr(blu_ar)
                    if cy>4:
                        bdir=-1 #this will make us start stepping cy backwards
                        #sparkle = 1
                        
        p[0,:] = coo[0]*viz_mf.flatMatHardMode(red_ar)
        p[1,:] = coo[1]*viz_mf.flatMatHardMode(gre_ar)
        p[2,:] = coo[2]*viz_mf.flatMatHardMode(blu_ar)
        
        return p
    def colorwave22(y):
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, oods, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
              coms, right, left, rig, lig, numx, numy, tcy, thresh, swit, qq
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim3+=1
        rtim5+=1
        bthe+=1
       
        xf = x_off
        yf = y_off
        
        rtim3 += 1

        tcy = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        if tcy > thresh and rtim3 > 5:
           
            if rtim3<=7:
                thresh*=1.05
                print("Threshold Change, colormove22")
                print(thresh)
            elif rtim3>15:
                thresh*=.95
                print("Threshold Change, colormove22")
                print(thresh) 
            rtim3 = 0
            swit *= -1
        if rtim3 >=25:
            rtim3 = 0
            thresh*=.95
            print("Threshold Change, colormove22")
            print(thresh)
            swit *= -1
        
        rtim += swit
        if cyc==0:
            for i in arx:
                red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)*255
                gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)*255
                blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)*255
        if cy>1:
            red_ar = np.fliplr(red_ar)
            if cy>2:
                gre_ar = np.fliplr(gre_ar)
                if cy>3:
                    blu_ar = np.fliplr(blu_ar)
                    if cy>4:
                        bdir=-1 #this will make us start stepping cy backwards
                        #sparkle = 1
                        
        p[0,:] = coo[0]*viz_mf.flatMatHardMode(red_ar)
        p[1,:] = coo[1]*viz_mf.flatMatHardMode(gre_ar)
        p[2,:] = coo[2]*viz_mf.flatMatHardMode(blu_ar)

        return p
    
    def colorwave23(y): #function 50
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, oods, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               coms, right, left, rig, lig, numx, numy
        
        y      = y**2
        gain.update(y)
        y     /= gain.value
        y     *= 255.
        ty     = np.mean(y[len(y)//2::])
        rtim  += 1
        rtim3 += 1
        rtim5 += 1
        bthe  += 1
        num = 5*np.sin(rtim/4)+6
        xf     = x_off
        yf     = y_off
        
        for i in arx:
            if i % 2 == 0:
                red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)**2*255
                gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)**2*255
                blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)**2*255
            else:
                red_ar[i,ary] =  (.5*np.sin(-rtim/4 + ary/yf + i/xf            )+.5)**2*255
                gre_ar[i,ary] =  (.5*np.sin(-rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)**2*255
                blu_ar[i,ary] =  (.5*np.sin(-rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)**2*255
#         for i in range(0,len(red_ar[0,:])):
#             if i % 2 == 1:
#                 red_ar[i,:] = np.flip(red_ar[i,:])
#                 blu_ar[i,:] = np.flip(blu_ar[i,:])
#                 gre_ar[i,:] = np.flip(gre_ar[i,:])
         
        p[0,:] = coo[0]*viz_mf.flatMatHardMode(red_ar)
        p[1,:] = coo[1]*viz_mf.flatMatHardMode(gre_ar)
        p[2,:] = coo[2]*viz_mf.flatMatHardMode(blu_ar)

        return p
    
    def colorwave24(y): #function 51
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, oods, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               coms, right, left, rig, lig, numx, numy, phas, red_ar2, blu_ar2, gre_ar2, sec
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim3+=1
        rtim5+=1
        bthe+=1
        rtim +=1
        xf = x_off
        yf = y_off
        
        for i in arx:
            red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)
            gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3+phas)+.5)
            blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3+2*phas)+.5)
        
        for i in arx:
            red_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 2*np.pi/3+phas)+.5)
            gre_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf + 4*np.pi/3+2*phas)+.5)
            blu_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/yf + i/xf )+.5)  
                        
        p[0,:] = (viz_mf.flatMatHardMode(red_ar)+viz_mf.flatMatHardMode(red_ar2)/2) * 255
        p[1,:] = (viz_mf.flatMatHardMode(gre_ar)+viz_mf.flatMatHardMode(gre_ar2)/2) * 255
        p[2,:] = (viz_mf.flatMatHardMode(blu_ar)+viz_mf.flatMatHardMode(blu_ar2)/2) * 255
        
        return p
    
    def colorwave25(y):  #function 52
        global p, rtim, pix, arx, ary, rtim3, coo, xdiv, ydiv, cy, oods, ard, cyc, bthe, thresh_bthe, bdir, red_ar, gre_ar, blu_ar, rtim4, C, rtim5, y_off, x_off, \
               right, left, rig, lig, numx, numy, phas, red_ar2, blu_ar2, gre_ar2, sec, inten1, inten2, ydi, xdi
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[len(y)//2::])
       
        rtim3+=1
        rtim5+=1
        bthe+=1

        xf = x_off
        yf = y_off
            
        for i in arx:
            inten1[i,ary]  = ((.5*np.sin(np.pi*ary/ydi - 25*np.pi)+.5)/2 + (.5*np.sin(np.pi*i/xdi - 10*np.pi)+.5)/2)*(.5*np.sin(rtim5*2*np.pi/25)+.5)
            inten2[i,ary]  = ((.5*np.sin(np.pi*ary/ydi)+.5)/2 + (.5*np.sin(np.pi*i/xdi)+.5)/2)*(.5*np.sin(rtim5*2*np.pi/25-np.pi)+.5)
            red_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi            )+.5)
            gre_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi + 2*np.pi/3+phas)+.5)
            blu_ar[i,ary] =  (.5*np.sin(rtim/4 + ary/ydi + i/xdi + 4*np.pi/3+2*phas)+.5)        

        
        for i in arx:
            red_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/ydi + i/xdi + 2*np.pi/3+phas)+.5)
            gre_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/ydi + i/xdi + 4*np.pi/3+2*phas)+.5)
            blu_ar2[i,ary] =  (.5*np.sin(rtim/sec + ary/ydi + i/xdi )+.5)
            
        p[0,:] = (viz_mf.flatMatHardMode(red_ar)+viz_mf.flatMatHardMode(red_ar2)/2) * 255
        p[1,:] = (viz_mf.flatMatHardMode(gre_ar)+viz_mf.flatMatHardMode(gre_ar2)/2) * 255
        p[2,:] = (viz_mf.flatMatHardMode(blu_ar)+viz_mf.flatMatHardMode(blu_ar2)/2) * 255
        p     *=  viz_mf.flatMatHardMode(inten1)/2 + viz_mf.flatMatHardMode(inten2)/2
        
        return p