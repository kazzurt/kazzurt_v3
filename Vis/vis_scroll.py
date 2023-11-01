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
import viz_mf
import kzbutfun
p      = np.tile(1.0, (3, config.N_PIXELS // 2))
pc     = np.tile(1.0, (3, config.N_PIXELS // 2))
pscroll= np.tile(1.0, (3, config.N_PIXELS // 2))
gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)
pix    = config.N_PIXELS // 2 - 1
nn     = 0

tr1    = 0
qw     = 0
sym    = 0
x      = 0
tim    = 200

scr    = 0
nn     = 0
ss3    = np.array([0, 0, 0, 0, 0, 0, 0, 0])
ss4 = ss3
ss     = ss3
s      = 0

tr1       = 0
qw        = 0
cnt       = 0
t1        = 0
cntp2     = 0
cntp3     = 0
sweetsc   = 0
kz        = 0
mp        = 200
rtim      = 0
y_off     = 14
x_off     = 2
red_ar    = np.zeros((20,25))
gre_ar    = np.zeros((20,25))
blu_ar    = np.zeros((20,25))
sig       = .2
arra1      = np.zeros((25,20))
arra2      = np.zeros((25,20))
arra3      = np.zeros((25,20))
h1 = np.linspace(0,499,10).astype(int)
cnt = 0
class scroll:

    def scrolly(y):
        """Effect that originates in the center and scrolls outwards"""
        global p, nn, ss, tr1, qw, kz, sym, x, tim
        coms = cmdfun.pygrun()   #Pulls all the command values
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
        
        if coms[47] == 1:
            p[rn.randint(0,3),rn.randint(0,len(p[0,:]))] = 255
            coms[47] = 0
        
        if tr > 20:
            ss[qw] = rn.randint(0,249)
            nn = 0
            qw += 1
            if qw>6:
                qw = 0
                ss = np.array([0, 0, 0, 0, 0, 0, 0, 0])
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

    def simplescroll(y):
        """Effect that originates in the center and scrolls outwards"""
        global p, scr, coms, rtim, red_ar, blu_ar, gre_ar, sig
        y = y**2.0
        gain.update(y)
        y /= gain.value
        y *= 255.0
        r = 2*int(np.max(y[:len(y) // 3]))
        g = 2*int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
        b = 2*int(np.max(y[2 * len(y) // 3:]))
        # Scrolling effect window
        scr+=1
        
        # Create new color originating at the center
        p[0, 0] = r
        p[1, 0] = g
        p[2, 0] = b
        
        p[0, config.N_PIXELS//2-1] = r
        p[1, config.N_PIXELS//2-1] = g
        p[2, config.N_PIXELS//2-1] = b
        p[:, 1:] = p[:, :-1]
        return np.concatenate((p[:, ::-1], p), axis=1)

    
    def sweetscroll(y):
        global p, nn, ss3, tr1, qw, kz, mp, t1, cnt, t2, dT, cntp2, cntp3, sweetsc, s, cnt, ss4
        cnt+=1
        
        r = 255/2*np.sin(cnt/20)+255/2
        g = 255/2*np.sin(cnt/20+2*np.pi/3)+255/2
        b = 255/2*np.sin(cnt/20+4*np.pi/3)+255/2

        nn += 1
        kz+=1
        
        p[0, ss3] = g
        p[1, ss3] = r
        p[2, ss3] = b
        if qw > 5:
            p[0, ss4] = 0
            p[1, ss4] = 0
            p[2, ss4] = 0
        #tr = np.mean(y)
        p[:, 1:] = p[:, :-1]
        
        if nn>20:
            ss3[qw] = rn.randint(0,config.N_PIXELS//2-1)
            if qw>5:
                ss4[qw] = rn.randint(0,config.N_PIXELS//2-1)
            #p[0, ss3[0]] = 0
            #p[1, ss3[0]] = 0
            #p[2, ss3[0]] = 0
            nn = 0
            qw += 1   
            if qw>=8:
                qw = 0
                ss3[0:3] = 0
                ss4[:] = 0
        p = gaussian_filter1d(p, sigma=0.5)

        
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
        
    def sweetscroll2(y):
        global p, nn, ss3, tr1, qw, kz, mp, t1, cnt, t2, dT, cntp2, cntp3, sweetsc, s, cnt, ss4
        cnt+=1
        
        r = 255/2*np.sin(cnt/20)+255/2
        g = 255/2*np.sin(cnt/20+4*(.5*np.sin(cnt/10)+1)*np.pi/3)+255/2
        b = 255/2*np.sin(cnt/20+2*(.5*np.sin(cnt/15)+1)*np.pi/3)+255/2

        nn += 1
        kz+=1
        
        p[0, ss3] = g
        p[1, ss3] = r
        p[2, ss3] = b
        if qw > 5:
            p[0, ss4] = 0
            p[1, ss4] = 0
            p[2, ss4] = 0
        #tr = np.mean(y)
        p[:, 1:] = p[:, :-1]
        
        if nn>20:
            ss3[qw] = rn.randint(0,config.N_PIXELS//2-1)
            if qw>5:
                ss4[qw] = rn.randint(0,config.N_PIXELS//2-1)
            #p[0, ss3[0]] = 0
            #p[1, ss3[0]] = 0
            #p[2, ss3[0]] = 0
            nn = 0
            qw += 1   
            if qw>=8:
                qw = 0
                ss3[0:3] = 0
                ss4[:] = 0
        p = gaussian_filter1d(p, sigma=0.5)

        
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    def scroll4(y):
        global p, nn, ss3, tr1, qw, kz, mp, t1, cnt, t2, dT, cntp2, cntp3, sweetsc, s, h1
        

        nn += 1
        kz+=1
        h2 = int((1300/4-1)*np.cos(kz/100)+(1300/4-1))
        p[0, h2] = (.25*np.sin(kz/25)+.75)*255
        p[1, h2] = (.25*np.sin(kz/25+np.pi/3)+.75)*255
        p[2, h2] = (.25*np.sin(kz/25+np.pi/3*2)+.75)*255
        
        #tr = np.mean(y)
        p[:, 1:] = p[:, :-1]
        
     
        p = gaussian_filter1d(p, sigma=1)

        
        return np.concatenate((p, p[:, ::-1]), axis=1) #typical symmetry about origin
    
    def launchscroll(y, lpad):
        global p, scr, coms, rtim, red_ar, blu_ar, gre_ar, sig, arra1, arra2, arra3

        # Scrolling effect window
        scr+=1
        coms = cmdfun.pygrun()
        fred = viz_mf.flatMatHardMode(red_ar)
        fblu = viz_mf.flatMatHardMode(blu_ar)
        fgre = viz_mf.flatMatHardMode(gre_ar)
        if np.sum(lpad[1,36:52])>=1:
            for ind in range(0,16):
                if lpad[1,ind+36] == 1:
                    arra1[ind,rn.randint(5,15)] = fred[ind]/.9
                    arra2[ind,rn.randint(5,15)] = fblu[ind]/.9
                    arra3[ind,rn.randint(5,15)] = fgre[ind]/.9
                    
        p[0,:] += viz_mf.flatMatHardMode(arra1)
        p[1,:] += viz_mf.flatMatHardMode(arra2)
        p[2,:] += viz_mf.flatMatHardMode(arra3)
#             print(arra)
#             for j in range(0,3):
#                 for i in range(0,len(arra[0,:])):
#                     p[j,i*31:int((lpad[1,i+36]/127)*31)] = arra[j,i] #:(i+1)*31
#         else:
#             arra = np.zeros((3,16))
                    
         
#         if coms[47] == 1:
# #             num = rn.randint(5,len(p[0,:]))
# #             num2 = num-5
# #             num3 = np.linspace(num,num2,5).astype(int)
#             num3 = np.random.randint(len(p[0,:]),size=3)
#             #print(num3)
#             for j in range(0,len(num3)):
#                 num4 = np.linspace(num3[j], num3[j]-5,5).astype(int)
#                 p[0,num4] *= fred[num4]
#                 p[1,num4] *= fblu[num4]
#                 p[2,num4] *= fgre[num4]
#                 
#             coms[47] = 0
        #p *= 0.98
#         if coms[13] == 1:
#             coms[13] = 0
#             sig+=.1
#         if coms[303] == 1:
#             coms[303] = 0
#             sig-=.1
#         if coms[312] == 1:
#             coms[312] = 0
#             p[:,rn.randint(0,len(p))] = 0
            
        

        num = 5*np.sin(rtim/4)+6
        xf = x_off+num
        yf = y_off-num
        ary = np.linspace(0,19,20).astype(int)
        rtim+=1
        for i in range(0,20):
                red_ar[ary,i] =  (.5*np.sin(rtim/4 + ary/yf + i/xf            )+.5)*255
                gre_ar[ary,i] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 2*np.pi/3)+.5)*255
                blu_ar[ary,i] =  (.5*np.sin(rtim/4 + ary/yf + i/xf + 4*np.pi/3)+.5)*255
        
        
#         p[0, 1:] = p[0, :-1]*viz_mf.flatMatHardMode(red_ar[:,:)
#         p[1, 1:] = p[1, :-1]*viz_mf.flatMatHardMode(gre_ar)
#         p[2, 1:] = p[2, :-1]*viz_mf.flatMatHardMode(blu_ar)
        p[:, 1:] = p[:, :-1]
        p*=.9
        p = gaussian_filter1d(p, sigma=.5) 
        #p[:, config.N_PIXELS//2-1-scr] = [r,g,b] #WHY does this jitter so much
        
        return np.concatenate((p[:, ::-1], p), axis=1)