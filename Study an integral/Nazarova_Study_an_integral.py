
from mpi4py import MPI
import numpy as np
import math

#Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def func(x):
    return 1/math.sqrt(1 + x**2)

a = 15
b = 21

N = 1000
h = (b-a)/N
S1 = 0

N_loc = math.ceil(N/size)
a_loc = rank*N_loc

#Main part
if rank != size-1:
    b_loc = (rank+1)*N_loc
else:
    b_loc = N-1

for k in range(a_loc,b_loc):
    x = a + k*h
    S1 += func(x)

S1 = comm.reduce(S1, op = MPI.SUM, root = 0)

if rank == 0:
    S0 = (func(a) + func(b))*h/2
    S = S0 + h*S1
