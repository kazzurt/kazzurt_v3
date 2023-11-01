import pygame
import numpy as np

coms = np.zeros(322)

def pygrun():
    global coms
    
    for event in pygame.event.get():
            if event.type==pygame.QUIT:
                mainloop=False

            pressed = pygame.key.get_pressed()
            
            #This creates an array of 0s and 1s corresponding to the entire keyboard.
            #Thank god I didn't have to do this one by one
            for k,v in enumerate(pressed):
                if v: 
                    coms[k] = 1
                    print('Command {:.0f}'.format(k))
                
            #buttons = [pygame.key.name(k) for k,v in enumerate(pressed) if v]          
            #print(buttons)  # print list to console
           
#             if pressed[pygame.K_z]:
#                 coms[0] = 1
#             if pressed[pygame.K_x]:
#                 coms[1] = 1
#             if pressed[pygame.K_c]:
#                 coms[2] = 1
#             if pressed[pygame.K_v]:
#                 coms[3] = 1
#             if pressed[pygame.K_b]:
#                 coms[4] = 1
#             if pressed[pygame.K_SPACE]:
#                 coms[5] = 1
#             if pressed[pygame.K_TAB]:
#                 coms[6] = 1
#             if pressed[pygame.K_BACKQUOTE]:
#                 coms[6] = 0
#             if pressed[pygame.K_0]:
#                 coms[8] = 1
#             if pressed[pygame.K_n]:
#                 coms[9] = 1
            if pressed[pygame.K_q]:
                print("Q QUIT")
                pygame.quit()
    return coms
     