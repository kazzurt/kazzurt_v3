
# Define objects for Quadratize

import numpy as np

funktions = ["bessel1","bessel2","colorwave01","colorwave02","radial_wave","colorwave26","radial_wave3","radial_wave5","radial_wave6","colorwave25","dantewave2"]

quad_odd_x = [i for i in range(0,26)]
quad_rev_y = [i for i in range(12,-1,-1)]

quad_1_x = []
quad_1_y = []

quad_2_x = []
quad_2_y = []

quad_3_x = []
quad_3_y = []

quad_4_x = []
quad_4_y = []

quad_top_x = []
quad_top_y = []
quad_btm_x = []
quad_btm_y = []


quad_1_x_t = []
quad_1_y_t = []

quad_2_x_t = []
quad_2_y_t = []

quad_3_x_t = []
quad_3_y_t = []

quad_4_x_t = []
quad_4_y_t = []

quad_left_x = []
quad_left_y = []
quad_right_x = []
quad_right_y = []

for x in quad_odd_x:
    if x % 2 == 1:
        y_max_top = 12
        y_max_bottom = 13
        
        y_max_left = 12
        y_max_right = 13
        
        incrementer_left = 1
        incrementer_rt = 0
    else:
        y_max_top = 13
        y_max_bottom = 12
        
        y_max_left = 13
        y_max_right = 12
        
        incrementer_left = 0
        incrementer_rt = 1
    for y in range(13):
        if y < y_max_top:
            quad_top_x.extend([x])
            quad_top_y.extend([y])
            
            quad_1_x.extend([x])
            quad_1_y.extend([y])
            
            quad_2_x.extend([x+26])
            quad_2_y.extend([y])
            
            
            quad_left_x.extend([x])
            quad_left_y.extend([y])
            
            quad_1_x_t.extend([x])
            quad_1_y_t.extend([y])
            
            quad_3_x_t.extend([x])
#             quad_3_y_t.extend([y + y_max_left + incrementer_left]) # For normal wired 3rd grid
            quad_3_y_t.extend([quad_rev_y[y + 13 - y_max_left] + y_max_left + incrementer_left]) # for reverse wired 3rd grid
            
        if y < y_max_bottom:
            quad_btm_x.extend([x])
            quad_btm_y.extend([y])
            
            quad_3_x.extend([x])
            quad_3_y.extend([y_max_top + y])
            
            quad_4_x.extend([x+26])
            quad_4_y.extend([y_max_top+y])
            
            
            quad_right_x.extend([x])
            quad_right_y.extend([y])
            
            quad_2_x_t.extend([x + 26])
            quad_2_y_t.extend([y])
            
            quad_4_x_t.extend([x + 26])
            quad_4_y_t.extend([y + y_max_right + incrementer_rt])
			
pop_ix_quad1 = [i for i in range(25,26*2*7+1,26*2)] + [i for i in range(26,26*2*6+1,26*2)]
pop_ix_quad1.sort()

def flatMatQuadMode(pixel_mat,grid_pos = 'top'):
    flattened_mat = []
    ref_dict = {}
    n_rows = pixel_mat.shape[0]
    n_cols = pixel_mat.shape[1]
    col_range = range(n_cols)
    keys = range(int(np.ceil(pixel_mat.shape[0]/2)))
    
    if grid_pos == 'top':
        key_comp = 0
    else:
        key_comp = 1

    max_val = 0
    for ikey in keys:
        if max_val+1 < n_rows:
            if ikey%2 == key_comp:
                dict_entry = [max_val,max_val+1]
            else:
                dict_entry = [max_val+1,max_val]
        else:
            dict_entry = [max_val]
        ref_dict[ikey] = dict_entry
        max_val = max_val + 2

    for key, value in ref_dict.items():
        modrem = key%2

        dict_entry_len = len(value)

        if dict_entry_len>1:

            init_row = value[0]
            zip_row  = value[1]
            row1_list = pixel_mat[init_row,:]
            row2_list = pixel_mat[zip_row,:]

            if modrem == 1:
                row1_list = list(reversed(row1_list))  #not using col1_list or col2_list
                row2_list = list(reversed(row2_list))

            for mat_col in col_range:
                flattened_mat.extend([row1_list[mat_col]])
                flattened_mat.extend([row2_list[mat_col]])
        else:
            flattened_mat.extend(pixel_mat[value[0],:])
                        
    return np.array(flattened_mat)

# <<<<<<< HEAD
# def flatMatQuads(pixel_mat):
#     false_mat_1 = np.zeros((26,13))
#     false_mat_1[quad_top_x,quad_top_y] = pixel_mat[quad_1_x,quad_1_y]
#     
#     false_mat_2 = np.zeros((26,13))
#     false_mat_2[quad_top_x,quad_top_y] = pixel_mat[quad_2_x,quad_2_y]
#     #false_mat_2 = np.flipud(false_mat_2)
#     
#     false_mat_3 = np.zeros((26,13))
#     false_mat_3[quad_btm_x,quad_btm_y] = pixel_mat[quad_3_x,quad_3_y]
#     #false_mat_3 = np.fliplr(np.flipud(false_mat_3))
#     
#     false_mat_4 = np.zeros((26,13))
#     false_mat_4[quad_btm_x,quad_btm_y] = pixel_mat[quad_4_x,quad_4_y]
# =======


# if layout = matrix, pixel_mat = np.zeros((52,25))
# if layout = mirrored, pixel_mat = np.zeros((52,26))

def flatMatQuads(pixel_mat, layout = 'mirror'):
# >>>>>>> 7fba80f130ccdfe9c2aaf9ac28e0d5541dcb1970
    
    if layout == 'matrix':
        false_mat_1 = np.zeros((26,13))
        false_mat_1[quad_top_x,quad_top_y] = pixel_mat[quad_1_x,quad_1_y]

        false_mat_2 = np.zeros((26,13))
        false_mat_2[quad_top_x,quad_top_y] = pixel_mat[quad_2_x,quad_2_y]

        false_mat_3 = np.zeros((26,13))
        false_mat_3[quad_btm_x,quad_btm_y] = pixel_mat[quad_3_x,quad_3_y]
        #false_mat_3 = np.fliplr(false_mat_3)
        false_mat_4 = np.zeros((26,13))
        false_mat_4[quad_btm_x,quad_btm_y] = pixel_mat[quad_4_x,quad_4_y]

        quad_1_flat = flatMatQuadMode(false_mat_1, grid_pos = 'top')
        quad_2_flat = flatMatQuadMode(false_mat_2, grid_pos = 'top')
        quad_3_flat = flatMatQuadMode(false_mat_3, grid_pos = 'bottom')
        quad_4_flat = flatMatQuadMode(false_mat_4, grid_pos = 'bottom')
    else:
        false_mat_1 = np.zeros((26,13))
        false_mat_1[quad_left_x,quad_left_y] = pixel_mat[quad_1_x_t,quad_1_y_t]

        false_mat_2 = np.zeros((26,13))
        false_mat_2[quad_right_x,quad_right_y] = pixel_mat[quad_2_x_t,quad_2_y_t]

        false_mat_3 = np.zeros((26,13))
        false_mat_3[quad_left_x,quad_left_y] = pixel_mat[quad_3_x_t,quad_3_y_t]
        
        false_mat_4 = np.zeros((26,13))
        false_mat_4[quad_right_x,quad_right_y] = pixel_mat[quad_4_x_t,quad_4_y_t]

        quad_1_flat = flatMatQuadMode(false_mat_1, grid_pos = 'top')
        quad_2_flat = flatMatQuadMode(false_mat_2, grid_pos = 'bottom')
        quad_3_flat = flatMatQuadMode(false_mat_3, grid_pos = 'top')
        quad_4_flat = flatMatQuadMode(false_mat_4, grid_pos = 'bottom')
        
    
    quad_1_flat = np.delete(quad_1_flat, pop_ix_quad1)
    quad_2_flat = np.delete(quad_2_flat, pop_ix_quad1)
    
    quad_3_flat = np.delete(quad_3_flat, pop_ix_quad1)
    quad_4_flat = np.delete(quad_4_flat, pop_ix_quad1)
    
    return np.concatenate((quad_1_flat ,quad_2_flat, quad_3_flat, quad_4_flat), axis = None)
    