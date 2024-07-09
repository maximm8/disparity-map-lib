from numba import njit,  cuda
from math import sqrt

@cuda.jit(device=True)
def sad_cuda(left, right):
    cost = 0
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            for k in range(left.shape[2]):
                cost += abs(left[i,j,k]-right[i,j,k])

    return cost

@cuda.jit(device=True)
def ssd_cuda(left, right):
    cost = 0
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            for k in range(left.shape[2]):
                cost += (left[i,j,k]-right[i,j,k])**2

    return cost

# @cuda.jit(device=True)
# def ssd_cuda(left, right):
#     cost = 0
#     for i in range(left.shape[0]):
#         for j in range(left.shape[1]):
#             for k in range(left.shape[2]):
#                 cost += (left[i,j,k]-right[i,j,k])**2

#     return cost

@cuda.jit(device=True)
def ncc_cuda(left, right):

    cost = 0
    lr = 0
    llsum = 0
    rrsum = 0
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            for k in range(left.shape[2]):
                lr += (left[i,j,k]*right[i,j,k])
                llsum += (left[i,j,k]*left[i,j,k])
                rrsum += (right[i,j,k]*right[i,j,k])

    cost = lr/sqrt(llsum*rrsum)    

    return cost

# @njit
# def zncc(left, right):
#     lm = left.mean()
#     rm = right.mean()
#     cost = ((left - lm)*(right-rm)).sum()/np.sqrt(((left - lm)**2).sum()*((right - rm)**2).sum())

#     return cost

@cuda.jit(device=True)
def zncc_cuda(left, right):

    lm = 0
    rm = 0
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            for k in range(left.shape[2]):
                lm += left[i,j,k]
                rm += right[i,j,k]

    N = left.shape[0]*left.shape[1]*left.shape[2]
    lm /= N
    rm /= N
    
    cost = 0
    lr = 0
    llsum = 0
    rrsum = 0
    for i in range(left.shape[0]):
        for j in range(left.shape[1]):
            for k in range(left.shape[2]):
                lr += (left[i,j,k]-lm)*(right[i,j,k]-rm)
                llsum += (left[i,j,k]-lm)*(left[i,j,k]-lm)
                rrsum += (right[i,j,k]-rm)*(right[i,j,k]-rm)

    cost = lr/sqrt(llsum*rrsum)    

    return cost