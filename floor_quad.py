import numpy as np


# Define objects for Quadratize

import numpy as np





def floorQuads(p):

    for i in range(0,24):
        quad_1_flat(i)

    
    return np.concatenate((quad_1_flat ,quad_2_flat, quad_3_flat, quad_4_flat), axis = None)
    
