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
import quadratize

gain     = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
p_filt   = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS)), alpha_decay=0.1, alpha_rise=0.99)

pix      = config.N_PIXELS // 2 - 1
co2      = pallette.pal(0)
p        = np.tile(1.0, (3, config.N_PIXELS))
coo      = np.array([1,1,1]).astype(float)
oods     = np.linspace(1,config.N_PIXELS-1,config.N_PIXELS//2).astype(int)
evs      = np.linspace(0,config.N_PIXELS-2,(config.N_PIXELS-1)//2).astype(int)

rtim   = 0
rtim3  = 0
cnt3   = 0
cy     = 0
ard    = 1
cyc    = 0
bthe   = 0
thresh_bthe = 0.5

coo2 = coo

timeCount = 1
countUp   = True
# arx       = np.linspace(0,config.N_PIXELS//50-1,config.N_PIXELS//50).astype(int)
# ary       = np.linspace(0,49,50).astype(int)
arx         = np.linspace(0,config.ARX-1,config.ARX).astype(int)
ary         = np.linspace(0,config.ARY-1,config.ARY).astype(int)
coo3      = np.ones((config.ARX,config.ARY))
coo4      = np.ones((config.ARX,config.ARY))
coo5      = np.ones((config.ARX,config.ARY))
coo6      = np.ones((config.ARX,config.ARY))
coo7      = np.ones((config.ARX,config.ARY))
coo8      = np.ones((config.ARX,config.ARY))
ar_wave0  = np.ones((config.ARX,config.ARY))
ar_wave1  = np.ones((config.ARX,config.ARY))
ar_wave2  = np.ones((config.ARX,config.ARY))
bdir      = 1
nuu       = 50 #defines speed for kuwave2 (higher is slower)
mat_map   = 1
xn        = 14
yn        = 49
upcnt     = 0

class breathing:
        
    def breathe(y):
        global p, p_filt, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, cy, oods, evs, xn, yn, upcnt
   
       
        rtim +=1
        rtim3+=1
        
        br = 5*np.sin(rtim/10)+10        
        for x in arx:
            ar_wave0[x,ary] = ((np.sin(ary*np.pi/(4*br))/2 + np.sin(x*np.pi/(br))/4))*(.5*np.sin(rtim/10+2*x+ary+(br**.4)/(br-cy))+.5)*255 
            
        if rtim3 >25:
            rtim3 = 0
            cy+=1
        coo[0] = (.5*np.sin(rtim/20)+.5)**.5
        coo[1] = (.5*np.sin(rtim/20+30/3)+.5)**.5
        coo[2] = (.5*np.sin(rtim/20+2*30/3)+.5)**.5
        
        p[0,:] = coo[0]*quadratize.flatMatQuads(ar_wave0)
        p[1,:] = coo[1]*quadratize.flatMatQuads(ar_wave0)
        p[2,:] = coo[2]*quadratize.flatMatQuads(ar_wave0)
        ppm    = np.mean(p[:,:])
        
        #if cy>1:
            #oods+=np.ones(len(oods))
            #p[0,oods] = p[1,oods]
            #p[1,oods] = p[2,oods]
#             if cy>3:
#                 p[1,evs] = p[2,evs]
#                 p[2,evs] = p[0,evs]
#                 if cy>7:
#                     p[1,evs] = p[2,evs]
#                     p[2,oods] = p[1,oods]
#                     if cy>10:
#                         cy = 0
        p = gaussian_filter1d(p, sigma=.2)
#         if lpad[1,56] == 1:
#             upcnt+=1
#             if upcnt>=20:
#                 lpad[1,56] = 0
#                 upcnt      = 0
#             p*= 1.5 * (.5*np.sin(upcnt/20*(np.pi/4) + np.pi/4)+.5)
#         elif lpad[1,54] == 1:
#             lpad[1,54]   = 0
#             p           *=.75
        return p

    def breathe2(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, bdir
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        
        #rtim +=.05
        #rtim3+=.05
        bthe+=1

        num = 5*np.sin(rtim/8)+5
        
        #If our x and y division sine wave is at its peak, let's increment cy, which determines color remapping below
        #rtim3 prevents going into this function a bunch of times while num sits close to its maximum
        #bdir controls whether we're stepping cy upwards or downwards
        if num > 9 and rtim3 > 20:
            rtim3 = 0
            if bdir == 1: 
                cy+=1
            elif bdir == -1:
                cy-=1
                if cy<=0:
                    bdir = 1
        
        #This is the best wave function, dependent on loop number (rtim), x direction (i in arx) and y direction (ary)
        #Would be nice if both directions could be a matrix operation, but idk how to do that. Choosing smaller direction for the for loop
        if cyc==0:
            for i in arx:
                ar_wave0[i,ary] =  (.5*np.sin(rtim/7 + ary/(11-num) + i/(.5+num))+.5)*255
     
        #Lets make colors perfectly phased using 2pi/3 and 4pi/3, num2 in denom controls how quickly the change
        num2 = 25*np.sin(rtim/25)+50 #this is a sine between 25 and 75 with a relatively slow speed
        ctim = (2*np.pi*rtim)/25
        coo[0] = (.5*np.sin(ctim + 0    )+.5 )
        coo[1] = (.5*np.sin(ctim + 2*np.pi/3 )+.5)
        coo[2] = (.5*np.sin(ctim + 4*np.pi/3 )+.5)

        p[0,:] = coo[0]*viz_mf.flatMat(ar_wave0)
        p[1,:] = coo[1]*viz_mf.flatMat(ar_wave0)
        p[2,:] = coo[2]*viz_mf.flatMat(ar_wave0)

        if cy>1:
            #remap odd colors
            p[0,oods] = p[1,oods] 
            p[1,oods] = p[2,oods]
            if cy>3:
                #remap even colors differently
                p[1,evs] = p[2,evs]
                p[2,evs] = p[0,evs]
                if cy>5:
                    #remap odds again, but differently
                    p[1,oods] = p[2,oods]
                    p[2,oods] = p[1,oods]
                    if cy>7:
                        #remap evens again, but differently
                        p[0,evs] = p[2,evs]
                        p[2,evs] = p[0,evs]
                        if cy>9:
                            bdir=-1 #this will make us start stepping cy backwards
        
        return p

    def becca_breathe(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        
        rtim +=1
        rtim3+=1
        bthe+=1
        
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
        for x in arx:
            coo2[0] = x*(.5*np.sin(rtim/10)+.5)**.5/5
            coo2[1] = x*(.5*np.sin(rtim/10+10/3)+.5)**.5/5
            coo2[2] = x*(.5*np.sin(rtim/27+2*27/3)+.5)**.5/5
        p[0,:] = coo2[0]*ar_wave0.flatten()
        p[1,:] = coo2[1]*ar_wave0.flatten()
        p[2,:] = coo2[2]*ar_wave0.flatten()
        ppm = np.mean(p[:,:])
        if cy>1:
            p[0,oods] = p[1,oods]
            p[1,oods] = p[2,oods]
            if cy>3:
                p[1,evs] = p[2,evs]
                p[2,evs] = p[0,evs]
                if cy>5:
                    p[1,oods] = p[2,oods]
                    p[2,evs] = p[1,evs]
                    if cy>7:
                        p[0,oods] = p[2,oods]
                        p[2,evs] = p[0,evs]
                        if cy>9:
                            cy = 0
        p = gaussian_filter1d(p, sigma=.2)
        return p


    def steve_breathe(y):
        global p, rtim, pix, arx, ary, ar_wave, ar_wave1, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        cyc = 0
        
        rtim +=1
        rtim3+=1
        bthe+=1
        if timeCount == 20:
            countUp = False
        
        if timeCount == 1:
            countUp = True
        
        if countUp:
            timeCount += 1
        else:
            timeCount -= 1
        
        for x in arx:
            ar_wave0[x,ary] = ((np.sin(ary*np.pi/((np.sin(rtim/10)+2)*5))/2
                                + np.sin(x*np.pi/((np.sin(rtim/10)+2)*25))/2))*(np.sin(rtim/10+ary/(2*np.sin(ary)+5)+x/3)+.75)*350#*(255//(timeCount/8))
            ar_wave1[x,ary] = ((np.sin((ary-5)*np.pi/((np.sin(rtim/10)+2)*5))/2
                                + np.sin(x*np.pi/((np.sin(rtim/10)+2)*25))/2))*(np.sin(rtim/10+(ary-5)/(2*np.sin((ary-5))+5)+(x+5)/3)+.75)*350#*(255//(timeCount/8))
            # ar_wave0[x,ary] = ((np.sin(ary*np.pi/(49))/2 + np.sin(x*np.pi/(14))/4))*(.5*np.sin(rtim/10+ary/5+x/5)+.5)*255
            coo3[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x)+.75)*(.5*np.sin(rtim/(4*np.pi)            )+.5)**.5
            coo4[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x+np.pi/2)+.75)*(.5*np.sin(rtim/(4*np.pi) + 2*np.pi/3)+.5)**.5
            coo5[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x+np.pi)+.75)*(.5*np.sin(rtim/(4*np.pi) + 4*np.pi/3)+.5)**.5
             
        if rtim3 >30 and ty>thresh_bthe:
            rtim3 = 0
            bthe=0
            cy+=1

        coo2[0] = (.5*np.sin(rtim/(2*np.pi)            )+.5)#**.5 #(.5*np.sin(arx)+.5)*
        coo2[1] = (.5*np.sin(rtim/(2*np.pi) + 2*np.pi/3)+.5)#**.5
        coo2[2] = (.5*np.sin(rtim/(2*np.pi) + 4*np.pi/3)+.5)#**.5
        
        a1 = ar_wave0*coo3
        a2 = ar_wave0*coo4
        a3 = ar_wave0*coo5
        a4 = ar_wave1*coo3
        a5 = ar_wave1*coo4
        a6 = ar_wave1*coo5
#         p[0,:] = a1.flatten()
#         p[1,:] = a2.flatten()
#         p[2,:] = a3.flatten()
        p[0,:] = viz_mf.flatMat(a1) + viz_mf.flatMat(a4)
        p[1,:] = viz_mf.flatMat(a2) + viz_mf.flatMat(a5)
        p[2,:] = viz_mf.flatMat(a3) + viz_mf.flatMat(a6)        
        ppm = np.mean(p[:,:])
        if cy>1:
            p[0,oods] = p[1,oods]
            p[1,oods] = p[2,oods]
            if cy>3:
                p[1,evs] = p[2,evs]
                p[2,evs] = p[0,evs]
                if cy>5:
                    p[1,oods] = p[2,oods]
                    p[2,evs] = p[1,evs]
                    if cy>7:
                        p[0,oods] = p[2,oods]
                        p[2,evs] = p[0,evs]
                        if cy>9:
                            cy = 0
        p = gaussian_filter1d(p, sigma=.5)
        return p
    
    def kuwave(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, mat_map
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        cyc = 0
        
        rtim +=1
        rtim3+=1
        bthe+=1
        
        #Dont change the xsp and ysp here. They're great. 
        xsp = 10*(np.sin(rtim/10)) + 2*np.sin(rtim/40+rtim/40*np.pi/2)+10
        ysp = 5*(np.sin(rtim/10)) + 2*np.sin(rtim/40)+8
        for x in arx:
            ar_wave0[x,ary] = (np.sin(rtim/25+x/xsp + ary/ysp))*255
            
            coo3[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x)+.75)*(.5*np.sin(rtim/(4*np.pi)            )+.5)**.5
            coo4[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x+2*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi) + 2*np.pi/3)+.5)**.5
            coo5[x,ary] = (.25*np.sin(ary)+.75)*(.25*np.sin(x+4*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi) + 4*np.pi/3)+.5)**.5
             
        coo2[0] = (.5*np.sin(rtim/(2*np.pi)            )+.5)**.5 
        coo2[1] = (.5*np.sin(rtim/(2*np.pi) + 2*np.pi/3)+.5)**.5
        coo2[2] = (.5*np.sin(rtim/(2*np.pi) + 4*np.pi/3)+.5)**.5
        
        a1 = ar_wave0*coo3
        a2 = ar_wave0*coo4
        a3 = ar_wave0*coo5
        
        #Both flattening functions lookin good. I'd prob choose flatmat as standard
        if mat_map == 1:
            p[0,:] = viz_mf.flatMat(a1)
            p[1,:] = viz_mf.flatMat(a2)
            p[2,:] = viz_mf.flatMat(a3)
        elif mat_map == 0:
            p[0,:] = a1.flatten()
            p[1,:] = a2.flatten()
            p[2,:] = a3.flatten()
        
        gau = .5*np.sin(rtim/30)+1
        p = gaussian_filter1d(p, sigma=gau)
        return p


    def kuwave2(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, nuu, mat_map, \
        coo6, coo7, coo8
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        cyc = 0
        
        rtim +=1
        rtim3+=1
        bthe+=1

        xsp  = 5
        ysp  = 5
        nuu = (.5*np.sin(rtim/10)+.5)*2 + 5


        
        #fade out after a certain amount of loops. I dont like it anymore
#         if rtim>400:
#             #p*=.98
#             p = gaussian_filter1d(p, sigma=3)
#             if rtim>500:
#                 rtim = 0
#             return p
        
        #The Essence of Perfection
        for x in arx:
            ar_wave0[x,ary] =(np.sin(rtim/nuu + x/xsp + ary/ysp))*255
            
            coo3[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x)+.75)*(.5*np.sin(rtim/(4*np.pi) /nuu           )+.5)**.5
            coo4[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x+2*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi)/nuu + 2*np.pi/3)+.5)**.5
            coo5[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x+4*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi)/nuu + 4*np.pi/3)+.5)**.5
#         if lpad[1,56] == 1:
#             print(lpad[1,56])
#             coo6[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(x)+.75)*(.5*np.sin(rtim/2/(4*np.pi) /nuu           )+.5)**.5
#             coo7[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(x+2*np.pi/3)+.75)*(.5*np.sin(rtim/2/(4*np.pi)/nuu + 2*np.pi/3)+.5)**.5
#             coo8[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(x+4*np.pi/3)+.75)*(.5*np.sin(rtim/2/(4*np.pi)/nuu + 4*np.pi/3)+.5)**.5
#         else:
#             coo6 = np.zeros((config.N_PIXELS//50,50))
#             coo7 = np.zeros((config.N_PIXELS//50,50))
#             coo8 = np.zeros((config.N_PIXELS//50,50))

        #if rtim<100:    
        a1 = ar_wave0*coo3
        a2 = ar_wave0*coo4
        a3 = ar_wave0*coo5
        a4 = ar_wave0*coo6
        a5 = ar_wave0*coo7
        a6 = ar_wave0*coo8
       
        if mat_map == 1:
            #flipping left right so that waves come from top left corner, since other wave functions go to top right corner. 
            p[2,:] = viz_mf.flatMat(np.fliplr(a1))+viz_mf.flatMat(a4)
            p[1,:] = viz_mf.flatMat(np.fliplr(a2))+viz_mf.flatMat(a5)
            p[0,:] = viz_mf.flatMat(np.fliplr(a3))+viz_mf.flatMat(a6)

        p = gaussian_filter1d(p, sigma=2)
        #p=np.fliplr(p)
        return p
    
    def palwave(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, co2, mat_map
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        cyc = 0
        
        rtim +=1
        rtim3+=1
        bthe+=1
        
        xsp   = 5*(np.sin(rtim/10)) + 20
        ysp   = 5*(np.sin(rtim/10)) + 10
        dfg   = len(arx)//len(co2[:,0])
        color = np.linspace(0,len(co2[:,0])-1,len(co2[:,0])).astype(int)
        

        for x in arx:
            ar_wave0[x,ary] = (np.sin(rtim/25+x/xsp + ary/ysp))
            ar_wave1[x,ary] = (np.sin(rtim/25+x/xsp + ary/ysp))
            ar_wave2[x,ary] = (np.sin(rtim/25+x/xsp + ary/ysp))
        
        for i in color:
            ar_wave0[arx[i*dfg:(i+1)*dfg],:] *= co2[i,0]
            ar_wave1[arx[i*dfg:(i+1)*dfg],:] *= co2[i,1]
            ar_wave2[arx[i*dfg:(i+1)*dfg],:] *= co2[i,2]
           
            if i==len(color)-1:
                ar_wave0[arx[(i+1)*dfg::],:] *= co2[i,0]
                ar_wave1[arx[(i+1)*dfg::],:] *= co2[i,1]
                ar_wave2[arx[(i+1)*dfg::],:] *= co2[i,2]
        
        #Eh this one looks better with original flattening         
        mat_map = 0
        if mat_map == 1:
            p[0,:] = viz_mf.flatMat(ar_wave0)
            p[1,:] = viz_mf.flatMat(ar_wave1)
            p[2,:] = viz_mf.flatMat(ar_wave2)
        elif mat_map == 0:    
            p[0,:] = ar_wave0.flatten()
            p[1,:] = ar_wave1.flatten()
            p[2,:] = ar_wave2.flatten()
        
        gau = .5*np.sin(rtim/30)+1
        p = gaussian_filter1d(p, sigma=gau)
        if rtim3>75:
            co2 = pallette.pal(0) #pull another random colour off the pallet
            rtim3 = 0
        return p
    def stag(y):
        global p, rtim, pix, arx, ary, ar_wave, phw, rtim3, coo2, xdiv, ydiv, cy, oods, evs, ard, cyc, bthe, thresh_bthe, timeCount, countUp, coo3, coo4, coo5, nuu
        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.
        ty = np.mean(y[:len(y)//2])/np.max(y[:len(y)//2])
        ty2 = np.mean(y[2*len(y)//3:len(y)])
        cyc = 0
        #print(ty2)
        #NO thats dumb
        #if ty2>5:
        rtim = (.5*np.sin(rtim3/50)+.5)*100 + 100
        rtim3+=1
        bthe+=1
        

        #xsp = (5*np.sin(rtim/50)+25)*(np.sin(rtim/10)) + 25#+ (10*np.sin(rtim/100)+15)
        #ysp = (5*np.sin(rtim/50)+25)*(np.sin(rtim/10)) + 25#+ (10*np.sin(rtim/100)+25)
        xsp  = (.5*np.sin(rtim3/20)+.5)*5+15
        ysp  = (.5*np.sin(rtim3/20)+.5)*5+15
        nuu = (.5*np.sin(rtim/50)+.5)*2 + 5
        #print(nuu)
        #nuu = 5
        #The Essence of Perfection
        for x in arx:
            ar_wave0[x,ary] = (np.sin(rtim/nuu+x/xsp + ary/ysp))*255
            
            coo3[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x)+.75)*(.5*np.sin(rtim/(4*np.pi) /nuu           )+.5)**.5
            coo4[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x+2*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi)/nuu + 2*np.pi/3)+.5)**.5
            coo5[x,ary] = (.5*np.sin(ary)+.75)*(.25*np.sin(-x+4*np.pi/3)+.75)*(.5*np.sin(rtim/(4*np.pi)/nuu + 4*np.pi/3)+.5)**.5
        #if rtim<100:    
        a1 = ar_wave0*coo3
        a2 = ar_wave0*coo4
        a3 = ar_wave0*coo5
           
#         p[2,:] = a1.flatten()
#         p[1,:] = a2.flatten()
#         p[0,:] = a3.flatten()
        p[2,:] = viz_mf.flatMat(a1)
        p[1,:] = viz_mf.flatMat(a2)
        p[0,:] = viz_mf.flatMat(a3)
        p = gaussian_filter1d(p, sigma=2)
        p=np.fliplr(p)
        return p
