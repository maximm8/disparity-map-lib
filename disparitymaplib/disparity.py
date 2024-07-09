import numpy as np
import math
from numba import njit,  cuda

@cuda.jit
def calc_cost(left, right, disparity_cost, disparity_range, window_size):
    
    img_size = left.shape    
    w = window_size
    y, x = cuda.grid(2)
    disp_min, disp_max = disparity_range

    if y+w[0]>=img_size[0] or x+w[1]>=img_size[1] or x-disp_max<0: return

    left_ = left[y:y+w[0], x:x+w[1],:]

    for d in range(disp_min, disp_max):
        right_ = right[y:y+w[0], x-d:x+w[1]-d,:]
        disparity_cost[y, x, d-disp_min] = cost_func(left_, right_)

def max_cost(disparity_cost):

    disparity_map = np.argmax(disparity_cost, axis=2)

    return disparity_map

def min_cost(disparity_cost):

    disparity_map = np.argmin(disparity_cost, axis=2)

    return disparity_map


def disparity(img1, img2, disp_range, wind_size, cost, cost_aggregator=min_cost, TPB1=16):

    global cost_func

    if len(img1.shape) == 2: img1 = img1[:,:,None]
    if len(img2.shape) == 2: img2 = img2[:,:,None]

    disp_range  = np.array(disp_range)
    wind_size   = np.array(wind_size)
    wind_size2  = wind_size//2

    rr, rc = img1.shape[0:2]
    blockspergrid = (math.ceil(rr/TPB1), math.ceil(rc/TPB1))

    # copy data to gpu 
    disparity_cost = np.zeros((img1.shape[0], img1.shape[1], disp_range[1]-disp_range[0]), dtype=np.float32)
    img1g, img2g, dispg = map(cuda.to_device, (img1.astype(np.float32), img2.astype(np.float32), disparity_cost))

    # set cost function
    cost_func = cost
    #calc cost ay every pixel
    calc_cost[blockspergrid, (TPB1, TPB1)](img1g, img2g, dispg,  disp_range,  wind_size)

    # calc disparity map
    disparity_cost = dispg.copy_to_host()
    disparity_map = cost_aggregator(disparity_cost)
    disparity_cost_min = np.min(disparity_cost, axis=2)

    # correct disparity location 
    disparity_map  = np.roll(disparity_map, wind_size2[0], axis=0)
    disparity_map  = np.roll(disparity_map, wind_size2[1], axis=1)

    return disparity_map