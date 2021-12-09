
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

# Initialization for parallelization
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Функция, которая сдвигает позиции
def change_the_pos(i,j):
    array_of_neigbors = np.array([field[i-1,j], field[i-1,j-1], field[i-1,j+1], field[i+1,j], field[i+1,j-1], 
                             field[i+1,j+1], field[i,j-1], field[i,j+1]])
    
    sum_of_neigbors = sum(array_of_neigbors)
    
    if field[i,j] == 1:
        ##Любая живая клетка с менее чем двумя живыми соседями умирает, как если бы она была недостаточной
        ##Любая живая клетка с более чем тремя живыми соседями умирает, как будто от перенаселения
        if sum_of_neigbors < 2 or sum_of_neigbors > 3:
            pass

        ##Любая живая клетка с двумя или тремя живыми соседями доживает до следующего поколения
        elif sum_of_neigbors == 2 or sum_of_neigbors == 3:
            blank[i,j] = 1

    ##Любая мертвая клетка с ровно тремя живыми соседями становится живой клеткой, 
    #как бы в результате размножения
    if field[i,j] == 0 and sum_of_neigbors == 3:
        blank[i,j] = 1

glider_gun =\
[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
 [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

field = np.zeros((50, 70))
field[1:10,1:37] = glider_gun

n = field.shape[0]

distr = n//(size)
list_of_distr = list(range(0, n+1, distr))
if list_of_distr[-1] != n:
    list_of_distr[-1] = n
    
list_to_send = []
for i in range(len(list_of_distr)-1):
    list_to_send.append([list_of_distr[i],list_of_distr[i+1]])
    
list_to_send = list_to_send[rank]
print(rank,list_to_send)


for number_of_iter in range(100):
    if rank == (size-1):
        blank = np.zeros((field.shape[0], field.shape[1]))

        for i in range(list_to_send[0], list_to_send[1]-1):
            for j in range(0, blank.shape[1]-1):
                ##check 8 neigbors without borders 
                change_the_pos(i,j)

        #bottom right corner
        change_the_pos(-1,-1)

        #bottom
        for i in range(list_to_send[0]-1, list_to_send[1]):
            for j in range(0, blank.shape[1]-1):
                change_the_pos(-1,j)

        #right
        for j in range(blank.shape[1]-1, blank.shape[1]):
            for i in range(list_to_send[0], list_to_send[1]-1):
                change_the_pos(i,-1)


        blank = blank[list_to_send[0]:list_to_send[1], :]


    else:
        blank = np.zeros((field.shape[0], field.shape[1]))

        # print(blank.shape)

        for i in range(list_to_send[0], list_to_send[1]):
            for j in range(0, blank.shape[1]-1):
                ##check 8 neigbors without borders 
                change_the_pos(i,j)

        #right
        for j in range(blank.shape[1]-1, blank.shape[1]):
            for i in range(list_to_send[0], list_to_send[1]-1):
                change_the_pos(i,-1)

        blank = blank[list_to_send[0]:list_to_send[1], :]

    comm.Barrier()
    Shifted = comm.gather(blank.copy(), root=0)

    if rank == 0:
        Data = np.concatenate((Shifted[0], Shifted[1]), axis=0)

        for i in range(2,size):
            Data = np.concatenate((Data, Shifted[i]), axis=0)
            
        field = Data.copy()
        
        for i in range(1, size):
            comm.send(field, dest=i)
        
                  
    else:
        field = comm.recv(source=0)
        

    if rank==0:
        plt.title(f"{number_of_iter}")
        plt.imshow(field)
        plt.show()
