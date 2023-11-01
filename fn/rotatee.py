
import PIL
from PIL import Image, ImageOps
import numpy as np

def rotatee(array_input, angle, n):
    
    arx = len(array_input[0,:])
    ary = len(array_input[:,0])
    
    ar_in = np.array(array_input)
    ar_out = Image.fromarray(ar_in).resize(size=(n*ary,n*arx))
    ar_out = ar_out.rotate(angle)
    ar_out = ar_out.resize(size=(ary,arx))
    array_out = np.transpose(np.array((ar_out)))
    return array_out