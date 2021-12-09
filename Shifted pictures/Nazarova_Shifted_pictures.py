
import cv2
import numpy as np
from numba import jit
from mpi4py import MPI
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Initialization for parallelization
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Loag image
money_heist = cv2.imread('meme.jpg')
money_heist = cv2.cvtColor(money_heist, cv2.COLOR_BGR2RGB)

m = money_heist.shape[0]
n = money_heist.shape[1]

# I share which piece of the picture goes to which rank. 
#I can use the "Send" command here, but I think it will take longer. 
#I already know to which rank which piece will go
distr = n//(size)
list_of_distr = list(range(0, n+1, distr))
if list_of_distr[-1] != n:
    list_of_distr[-1] = n
    
list_to_send = []
for i in range(len(list_of_distr)-1):
    list_to_send.append([list_of_distr[i],list_of_distr[i+1]])
    
list_to_send = list_to_send[rank]


money_heist_split = money_heist[:, list_to_send[0]:list_to_send[1], :]

## Using Numpy
list_for_gif = []
for i in range(m):
    money_heist_split = np.roll(money_heist_split, 1, axis=0)

comm.Barrier()
Shifted = comm.gather(money_heist_split.copy(), root=0)

if rank == 0:
    Data = np.concatenate((Shifted[0], Shifted[1]), axis=1)
    
    for i in range(2,size):
        Data = np.concatenate((Data, Shifted[i]), axis=1)
        
    list_for_gif.append(Data)

## Algorithm with two loops

# for j in range(money_heist_split.shape[0]):
#     current = money_heist_split[0, :, :]
#     for i in range(money_heist_split.shape[0]):
#         try:
#             next_ = money_heist_split[i+1, :, :].copy()

#             money_heist_split[i+1, :, :] = current

#             current = next_

#         except:
#             money_heist_split[0, :, :] = current
            
    # comm.Barrier()
    # Shifted = comm.gather(money_heist_split.copy(), root=0)

    # if rank == 0:
    #     Data = np.concatenate((Shifted[0], Shifted[1]), axis=1)
        
    #     for i in range(2,size):
    #         Data = np.concatenate((Data, Shifted[i]), axis=1)
            
    #     list_for_gif.append(Data)
