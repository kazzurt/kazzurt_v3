import time
import numpy as np
import pyaudio
import config
import dsp

import cmdfun
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation 
from numpy.fft import fft, ifft



RATE =1000

def start_stream(callback):
    p = pyaudio.PyAudio()
    frames_per_buffer = 0  #int(config.MIC_RATE / config.FPS)#config.FPS
#     stream = p.open(format=pyaudio.paInt16,
#                     channels=1,
#                     rate=config.MIC_RATE,
#                     input=True,
#                     frames_per_buffer=frames_per_buffer)
    overflows = 0
    prev_ovf_time = time.time()
    prev = 0
    while True:
        try:
            y=5
            
            #y = #np.fromstring(stream.read(frames_per_buffer, exception_on_overflow=False), dtype=np.int16)
            #y = y.astype(np.float32)
            callback(y)
            
        except IOError:
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
                print('Audio buffer has overflowed {} times'.format(overflows))
    stream.stop_stream()
    stream.close()
    p.terminate()
