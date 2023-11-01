import numpy as np
from numpy import random as rn

class pallette:
    def pal(co):
        if co == 0:
            c = rn.randint(0,19)
        else:
            c = co
            
        watercolor       = np.array([[222, 226, 255], [251, 168, 250], [244, 136, 238], [63,  131, 238], [120,182,241], [162,215,226]])
        pastel           = np.array([[204, 224, 215], [215, 200, 223], [249, 217, 220], [225, 235, 244], [212, 214, 249]])
        endracism        = np.array([[21,  105, 184], [220, 215,  55], [75,  174, 126], [173, 36,   75], [207, 93,  134]])
        sunrise          = np.array([[24,   12,  92], [87,   21, 109], [180,  24, 105], [255, 245, 182], [254, 203, 145], [230, 46, 116]])
        instagram        = np.array([[244,  71,  77], [238, 220,  49], [127, 219, 106], [14,  104, 206]])
        candy            = np.array([[255, 235, 153], [80,  100, 135], [87,  202, 198], [188, 218, 110], [243, 113, 111], [211, 79, 89]])
        chromo           = np.array([[168,   8,   8], [254,  39,  18], [253,  83, 8],   [251, 153,   2], [198, 0, 144]])
        chemistry        = np.array([[202, 226, 190], [248, 243, 211], [254, 198, 194], [103, 169, 219], [88, 91, 154], [243, 236,153]])
        sunflower        = np.array([[183, 207,  12], [134, 189,  38], [46,   43, 34],  [248, 157,   2], [247, 187, 0], [249, 202, 0]])
        wipro            = np.array([[108, 189,  81], [254, 202,  29], [209,  35, 55],  [68,   80, 161], [0, 141, 206]])
        purps            = np.array([[147,  99, 186], [221, 152, 210], [118,  38, 127], [232, 217,  70], [65, 41, 115]])
        pointsettia      = np.array([[25,   37,  15], [53,   80,  29], [255, 222, 70],  [247,  52,  40], [190, 22, 22]])
        catchy           = np.array([[255,  87,  81], [255, 237, 107], [0,   224, 255], [16,    0, 255]])
        joy              = np.array([[242,  63, 254], [137,  55, 251], [225, 230,  47], [255, 255, 255]])
        recover          = np.array([[191,  85,  79], [75,   25,  56], [173, 113,  88], [222, 186, 107], [62,  67, 140]])
        valentine        = np.array([[85,  27,  140], [136,  43, 222], [237, 130, 237], [199,  21, 133]])
        vintage          = np.array([[195,  55,  64], [216, 162, 121], [226, 213, 184], [102, 148, 153], [105, 66, 57]])
        spaceinv         = np.array([[248,  59,  58], [235, 223, 100], [98,  222, 109], [219,  85, 221], [83,  88, 214], [66, 233, 244]]) 
        switch           = np.array([[250, 255,   0], [144,  13, 255], [255,   1, 129], [50,  219, 240]])
        kurtees          = np.array([[255, 140,   0], [0,   255, 250], [255,  20, 147], [70,  130, 180]])
        murica           = np.array([[179,  25,  66], [255, 255, 255], [10,   49,  97],[179,  25,  66], [255, 255, 255], [10,   49,  97]])
        bold             = np.array([[1,   185, 189], [37,  39,   89], [241, 196, 197], [70,  88, 232], [248, 179,  22], [244, 118,  27]])
        explain          = np.array([[177,  77, 133], [213, 119, 105], [220, 161, 98 ], [52, 134, 137], [87, 70, 131]])
        
        if c == 0:
            color = watercolor
        elif c == 1:
            color = pastel
        elif c == 2:
            color = endracism
        elif c == 3:
            color = sunrise
        elif c == 4:
            color = instagram
        elif c == 5:
            color = candy
        elif c == 6:
            color = chromo
        elif c== 7:
            color = chemistry
        elif c == 8:
            color = sunflower
        elif c == 9:
            color = wipro
        elif c == 10:
            color = purps
        elif c == 11:
            color = pointsettia
        elif c == 12:
            color = catchy
        elif c == 13:
            color = joy
        elif c == 14:
            color = recover
        elif c == 15:
            color = valentine
        elif c == 16:
            color = vintage
        elif c == 17:
            color = spaceinv
        elif c == 18:
            color = switch
        elif c == 19:
            color = kurtees
        elif c == 20:
            color = murica
        elif c == 21:
            color = bold
        elif c == 22:
            color = explain
        return color
        