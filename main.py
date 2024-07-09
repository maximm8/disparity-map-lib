import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import disparitymaplib as dm

from numba import cuda

img1  = cv2.imread('data/left.png')
img2  = cv2.imread('data/right.png')

disp_range  = [0, 60]
wind_size   = [11, 11]

disparity_map = dm.disparity(img1, img2, disp_range, wind_size, dm.sad_cuda, dm.min_cost)
# disparity_map = dm.disparity(img1, img2, disp_range, wind_size, dm.ssd_cuda, dm.min_cost)
# disparity_map = dm.disparity(img1, img2, disp_range, wind_size, dm.ncc_cuda, dm.max_cost)
# disparity_map = dm.disparity(img1, img2, disp_range, wind_size, dm.zncc_cuda, dm.max_cost)

fig, ax = plt.subplots(1,2)
ax[0].imshow(img1[:,:,::-1])
ax[1].imshow(disparity_map, cmap='jet')
plt.show()

