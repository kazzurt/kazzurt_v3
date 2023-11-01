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
import kzbutfun
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)
p      = np.tile(1.0, (3, config.N_PIXELS ))#// 2))
gain   = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)

ends = np.linspace(0,950,20).astype(int)
#mids = np.linspace(25,975,20).astype(int)
mids = ends+49
mids      = np.tile(1.0, (3, len(ends )))
mids[0,:] = np.linspace(25,975,20).astype(int)
mids[1,:] = ends+35

store = np.zeros((5,3,1000))
n1 = np.linspace(0,999,50).astype(int)
n = 0
c = 0
c1 = 0
rtim = 0
arr = np.zeros((40,25))
r1 = 17
r2 = 22
c1 = 12
c2 = 15
d = np.array([1,1,1,1])
xdir = np.linspace(0,39,40).astype(int)
ydir = np.linspace(0,19,20).astype(int)
for i in range(0,40):
    for j in range(0,20):
        arr[i,j] = (np.abs((i-20)*(j-10)))
arr /= np.max(arr)
arr2 = arr
arr3 = arr
cnt = 0
class drop:

    def drop(y):
        global p, ends, mids, n, c, c1, n1,rtim, arr, arr2, arr3, r1, r2, c1, c2, d, xdir, ydir, p_filt, prev,cnt,store

        y = y**2
        gain.update(y)
        y /= gain.value
        y *= 255.0
        rtim+=1
        g = int(np.max(y[:len(y) // 3]))**.25
        b = int(np.max(y[len(y) // 2: 2 * len(y) // 2]))**.25
        r = int(np.max(y[2 * len(y) // 3:]))**.25
        c+=1
        trig = np.mean((r+g+b)/3)
        trig/=100
        nim = 25*np.sin(rtim/5)+2
        #xd = 5*np.sin(rtim/nim)+20
        #yd = 5*np.sin(rtim/nim+np.pi/2) + 10
        xd = 20
        yd = 12
        for i in range(0,40):
            for j in range(0,25):
                #arr[i,j] = np.sin((rtim*((i-xd)/4*(j-yd)/4))/100 + 10)*200
                arr[i,j] = 255* np.sin(rtim /15 + (i-xd)*(j-yd))
                arr2[i,j] = 255* np.sin(rtim /15 + (i-xd)*(j-yd)+np.pi/3)
        n1 = viz_mf.flatMatHardMode(arr)
        n2 = viz_mf.flatMatHardMode(np.fliplr(arr))
        n3 = viz_mf.flatMatHardMode(arr2)#viz_mf.flatMatHardMode(np.flipud(arr))

        p[0, :]    = n1 * (.25*np.sin(rtim/15)+.75)#b*n1
        p[1, :]    = n3 * (.25*np.sin(rtim/15+np.pi/3)+.75)#g*n1 #*(.5*np.sin(rtim/20)+.5)
        p[2, :]    = n2 * (.25*np.sin(rtim/15 + 2*np.pi/3)+.75)#r*n1
       
        #p_filt.update(p[:,0:500])
        #p_filt.update(p[:,500:])
        prev = p
        #p = (p+prev)/2
        #p += prev
        
        p = gaussian_filter1d(p, sigma=.3)
        cnt+=1

        return p


    




