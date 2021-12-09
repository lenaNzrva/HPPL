
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# Initialization for parallelization
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Initialization for Bifurcation diagram
steps = 500
n = [500, 330, 250, 200, 160, 140, 120]
ran = np.linspace(0, 4, size*n[size-2])

def bifurcation_map(r, x):
    return r*x*(1-x)

# Initialization for who takes which piece of r
distr = len(ran)//(size)

list_of_distr = list(range(0, len(ran), distr))
list_of_distr.append(len(ran))

list_to_send = []
for i in range(len(list_of_distr)-1):
    list_to_send.append([list_of_distr[i],list_of_distr[i+1]])


# Parallelization itself
sendbuf = None
Rs_send = None
if rank == 0:
    sendbuf = np.empty([size*n[size-2], steps], dtype='f')
    sendbuf.T[:,:] = range(size*n[size-2])

    Rs_send = np.empty([size*n[size-2], steps], dtype='f')
    Rs_send.T[:,:] = range(size*n[size-2])

recvbuf = np.empty((n[size-2], steps), dtype='f')
comm.Scatter(sendbuf, recvbuf, root=0)

Rs_res = np.empty((n[size-2], steps), dtype='f')
comm.Scatter(Rs_send, Rs_res, root=0)

list_to_send = list_to_send[rank]

j = -1
for r in ran[list_to_send[0]:list_to_send[1]]:
    j += 1
    x = np.random.random()
    for i in range(steps):
        recvbuf[j, i] = x
        Rs_res[j, i] = r
        x = bifurcation_map(r, x)

comm.Barrier()

comm.Gather(recvbuf, sendbuf, root=0)
comm.Gather(Rs_res, Rs_send, root=0)

if rank == 0:
    Rs_send = Rs_send[:,Rs_send.shape[1]//2:]
    sendbuf = sendbuf[:,sendbuf.shape[1]//2:]
    Rs_send = Rs_send.reshape(Rs_send.shape[0]*Rs_send.shape[1])
    sendbuf = sendbuf.reshape(sendbuf.shape[0]*sendbuf.shape[1])
    plt.figure(figsize=(9, 8))
    plt.title('Bifurcation map')
    plt.xlabel("r")
    plt.ylabel("x")
    plt.plot(Rs_send, sendbuf, 'r.', markersize=1)
    plt.grid()
    plt.pause(1)
            
